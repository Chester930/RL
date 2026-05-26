"""
在 CartPole-v1 或 Atari 上訓練 C51 (分類式 DQN / Categorical DQN)。

參考文獻：
    Bellemare, M. G., Dabney, W., & Munos, R. (2017).
    A Distributional Perspective on Reinforcement Learning. ICML 2017.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import C51Agent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> C51Agent:
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

    agent = C51Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_atoms=config["n_atoms"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        lr=config["lr"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        target_update=config["target_update"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_steps=config["epsilon_steps"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"c51_{config['env_id']}")

    obs, _ = env.reset()
    ep_return = 0.0
    ep_length = 0

    print(f"正在 {config['env_id']} 環境上進行 C51 訓練，總步數為 {config['total_steps']} 步...")

    for step in range(1, config["total_steps"] + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_return += reward
        ep_length += 1

        if done:
            logger.log_episode(ep_return, ep_length, step)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_length = 0

        if step >= config["learning_starts"]:
            metrics = agent.update()
            if metrics and np.isnan(metrics.get("loss", 0)):
                raise RuntimeError(f"NaN loss detected at step {step}, stopping training.")
            if metrics and step % config["log_freq"] == 0:
                logger.log_scalars(metrics, step)

        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=10)
            logger.log_scalar("eval/mean_return", mean_r, step)
            print(f"步數 {step:8d}  評估回報: {mean_r:.1f} ± {std_r:.1f}")
            if step > 10_000 and mean_r < best_return * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_return:.1f}")
            if mean_r > best_return:
                best_return = mean_r
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳：{mean_r:.1f}，已儲存")

        if step % config["save_freq"] == 0:
            agent.save(f"checkpoints/c51_step{step}")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "CartPole-v1",
        "total_steps": 300_000,
        "n_atoms": 51,
        "v_min": 0.0,
        "v_max": 500.0,
        "lr": 1e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 32,
        "target_update": 500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_steps": 150_000,
        "learning_starts": 1000,
        "log_freq": 1000,
        "eval_freq": 10_000,
        "save_freq": 50_000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
