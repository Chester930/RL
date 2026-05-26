"""在低維狀態環境上訓練 State-based Dreamer（重跑版，2026-05-25）。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import pickle
import torch
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import StateDreamerAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

RESUME_DIR = "checkpoints/resume"


def train(config: dict) -> StateDreamerAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    env      = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])  # Pendulum: 2.0

    agent = StateDreamerAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        embed_dim=config["embed_dim"],
        deter_dim=config["deter_dim"],
        stoch_dim=config["stoch_dim"],
        gamma=config["gamma"],
        lambda_=config["lambda_"],
        imagine_horizon=config["imagine_horizon"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"dreamer_state_{config['env_id']}")
    global_step = 0
    best_eval   = -float("inf")
    os.makedirs("best_checkpoints", exist_ok=True)

    start_episode = 1

    # 自動偵測暫停點並繼續 (Auto-detect resume checkpoint)
    resume_meta_path = os.path.join(RESUME_DIR, "train_meta.pkl")
    resume_ckpt_path = os.path.join(RESUME_DIR, "dreamer_state.pt")
    if os.path.exists(resume_ckpt_path) and os.path.exists(resume_meta_path):
        agent.load_resume(RESUME_DIR)
        with open(resume_meta_path, "rb") as f:
            meta = pickle.load(f)
        start_episode = meta["episode"] + 1
        global_step   = meta["global_step"]
        best_eval     = meta["best_eval"]
        random.setstate(meta["random_state"])
        np.random.set_state(meta["np_state"])
        torch.set_rng_state(meta["torch_state"])
        print(f"[RESUME] 從集數 {meta['episode']}（step {global_step}）繼續，歷史最佳 {best_eval:.1f}")

    print(f"State-based Dreamer 重跑：{config['env_id']}，{config['n_episodes']} 集")
    print(f"  state_dim={state_dim}, action_dim={action_dim}, action_scale={action_scale}")

    for episode in range(start_episode, config["n_episodes"] + 1):
        obs, _ = env.reset()
        agent.reset_state()
        ep_return = ep_length = 0
        done = False

        while not done:
            if global_step < config["seed_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, done)
            obs = next_obs
            ep_return += reward
            ep_length += 1
            global_step += 1

            if global_step % config["update_every"] == 0 and global_step > config["seed_steps"]:
                for _ in range(config["update_steps"]):
                    metrics = agent.update()
                if metrics and np.isnan(metrics.get("critic_loss", 0)):
                    raise RuntimeError(f"NaN loss detected at step {global_step}, stopping training.")
                if metrics and global_step % (config["update_every"] * 20) == 0:
                    logger.log_scalars(metrics, global_step)

        logger.log_episode(ep_return, ep_length, global_step)

        if episode % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, global_step)
            print(f"Episode {episode:5d} | Step {global_step:8d} | Eval: {mean_r:.1f} ± {std_r:.1f}")
            if global_step > 10_000 and mean_r < best_eval * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_eval:.1f}")
            if mean_r > best_eval:
                best_eval = mean_r
                agent.save("best_checkpoints")
                print(f"  ** 新最佳 {best_eval:.1f}，已儲存 checkpoint **")

        if episode % config["save_freq"] == 0:
            agent.save(f"checkpoints/dreamer_ep{episode}")
            agent.save_resume(RESUME_DIR)
            meta = {
                "episode": episode,
                "global_step": global_step,
                "best_eval": best_eval,
                "random_state": random.getstate(),
                "np_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
            }
            with open(os.path.join(RESUME_DIR, "train_meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            print(f"  [RESUME] 暫停點已儲存至 {RESUME_DIR}（集數 {episode}）")

    logger.close()
    env.close()
    eval_env.close()
    print(f"\n訓練完成。最佳 eval: {best_eval:.1f}")
    return agent


if __name__ == "__main__":
    config = {
        "env_id":         "Pendulum-v1",
        "n_episodes":     500,
        "embed_dim":      64,
        "deter_dim":      128,
        "stoch_dim":      20,
        "gamma":          0.99,
        "lambda_":        0.95,
        "imagine_horizon": 15,
        "seed_steps":     1000,
        "update_every":   20,
        "update_steps":   4,
        "eval_freq":      25,
        "save_freq":      100,
        "device":         "cuda" if torch.cuda.is_available() else "cpu",
        "seed":           42,
    }
    train(config)
