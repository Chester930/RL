"""
在稀疏獎勵環境上訓練 ICM (PPO + 內在好奇心模組 / Intrinsic Curiosity Module)。

參考文獻：
    Pathak, D., et al. (2017). Curiosity-driven Exploration by Self-Supervised Prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import pickle
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import ICMAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

RESUME_DIR = "checkpoints/resume"


def train(config: dict) -> ICMAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    best_return = -float("inf")

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ICMAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        eta=config["eta"],
        beta=config["beta"],
        lr=config["lr"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_eps=config["clip_eps"],
        n_epochs=config["n_epochs"],
        rollout_steps=config["rollout_steps"],
        batch_size=config["batch_size"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"icm_{config['env_id']}")

    obs, _ = env.reset()
    ep_return = 0.0
    ep_length = 0
    total_steps = 0

    # 自動偵測暫停點並繼續 (Auto-detect resume checkpoint)
    resume_meta_path = os.path.join(RESUME_DIR, "train_meta.pkl")
    resume_ckpt_path = os.path.join(RESUME_DIR, "icm_checkpoint.pt")
    if os.path.exists(resume_ckpt_path) and os.path.exists(resume_meta_path):
        agent.load_resume(RESUME_DIR)
        with open(resume_meta_path, "rb") as f:
            meta = pickle.load(f)
        total_steps = meta["total_steps"]
        ep_return   = meta["ep_return"]
        ep_length   = meta["ep_length"]
        best_return = meta["best_return"]
        obs         = meta["obs"]
        random.setstate(meta["random_state"])
        np.random.set_state(meta["np_state"])
        torch.set_rng_state(meta["torch_state"])
        print(f"[RESUME] 從步數 {total_steps} 繼續訓練，歷史最佳 {best_return:.1f}")

    print(f"正在 {config['env_id']} 環境上進行 ICM+PPO 訓練，總步數為 {config['total_steps']} 步...")

    while total_steps < config["total_steps"]:
        # 收集取樣資料 (Collect rollout)
        for _ in range(config["rollout_steps"]):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(next_obs, reward, done)
            obs = next_obs
            ep_return += reward
            ep_length += 1

            if done:
                logger.log_episode(ep_return, ep_length, total_steps)
                obs, _ = env.reset()
                ep_return = 0.0
                ep_length = 0

        # 取樣結束後執行更新 (Update after rollout)
        metrics = agent.update()
        if metrics and np.isnan(metrics.get("policy_loss", 0)):
            raise RuntimeError(f"NaN loss detected at step {total_steps}, stopping training.")
        total_steps += config["rollout_steps"]

        if metrics:
            logger.log_scalars(metrics, total_steps)

        if total_steps // config["eval_freq"] > (total_steps - config["rollout_steps"]) // config["eval_freq"]:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=5)
            logger.log_scalar("eval/mean_return", mean_r, total_steps)
            print(f"步數 {total_steps:8d}  評估回報: {mean_r:.1f} ± {std_r:.1f}  "
                  f"內在獎勵: {metrics.get('mean_intr_reward', 0):.4f}")
            if total_steps > 10_000 and mean_r < best_return * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_return:.1f}")
            if mean_r > best_return:
                best_return = mean_r
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳：{mean_r:.1f}，已儲存")

        if total_steps // config["save_freq"] > (total_steps - config["rollout_steps"]) // config["save_freq"]:
            agent.save(f"checkpoints/icm_step{total_steps}")
            agent.save_resume(RESUME_DIR)
            meta = {
                "total_steps": total_steps,
                "ep_return": ep_return,
                "ep_length": ep_length,
                "best_return": best_return,
                "obs": obs,
                "random_state": random.getstate(),
                "np_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
            }
            with open(os.path.join(RESUME_DIR, "train_meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            print(f"  [RESUME] 暫停點已儲存至 {RESUME_DIR}（步數 {total_steps}）")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "MountainCar-v0",  # 經典的稀疏獎勵環境
        "total_steps": 1_000_000,
        "feature_dim": 256,
        "hidden_dim": 256,
        "eta": 1.0,        # 內在獎勵縮放因子 (Intrinsic reward scale)
        "beta": 0.2,       # 前向 vs 逆向損失權重 (Forward vs Inverse loss weight)
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "n_epochs": 4,
        "rollout_steps": 2048,
        "batch_size": 64,
        "eval_freq": 20_000,
        "save_freq": 100_000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
