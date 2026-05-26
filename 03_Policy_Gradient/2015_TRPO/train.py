"""在 CartPole-v1 上訓練 TRPO。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym
from agent import TRPOAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> TRPOAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = TRPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr_critic=config["lr_critic"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        max_kl=config["max_kl"],
        damping=config["damping"],
        cg_iters=config["cg_iters"],
        line_search_iters=config.get("line_search_iters", 10),
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"trpo_{config['env_id']}")
    global_step = 0
    best_eval = -float("inf")
    os.makedirs("best_checkpoints", exist_ok=True)

    print(f"TRPO 重跑：max_kl={config['max_kl']}, n_episodes={config['n_episodes']}")

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        ep_return = ep_length = 0
        done = False

        while not done:
            for _ in range(config["n_steps"]):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store(obs, action, reward, done)
                obs = next_obs
                ep_return += reward
                ep_length += 1
                global_step += 1
                if done:
                    break

            metrics = agent.update(next_state=None if done else obs, last_done=done)
            if metrics and np.isnan(metrics.get("critic_loss", 0)):
                raise RuntimeError(f"NaN loss detected at step {global_step}, stopping training.")

        logger.log_episode(ep_return, ep_length, global_step)

        if episode % config["eval_freq_ep"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, global_step)
            print(f"Episode {episode:5d} | Step {global_step:8d} | Eval: {mean_r:.1f} ± {std_r:.1f}")
            if global_step > 10_000 and mean_r < best_eval * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_eval:.1f}")
            if mean_r > best_eval:
                best_eval = mean_r
                agent.save("best_checkpoints")
                print(f"  ** 新最佳 {best_eval:.1f}，已儲存 checkpoint **")

    logger.close()
    env.close()
    eval_env.close()
    print(f"\n訓練完成。最佳 eval: {best_eval:.1f}")
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "CartPole-v1",
        "n_episodes": 1000,
        "lr_critic": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "max_kl": 0.01,
        "damping": 0.1,
        "cg_iters": 10,
        "line_search_iters": 20,
        "n_steps": 2048,
        "eval_freq_ep": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
