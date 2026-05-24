"""
A2C 訓練迴圈（4 個平行環境，VecEnv 版本）。

使用 gymnasium.vector.SyncVectorEnv 同時收集 4 個環境的資料，
相較於單環境版本，梯度估計的方差更小、訓練更穩定。

參考文獻：
    Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning.
    A2C 是由 OpenAI 推廣的同步版本（A3C 的同步替代方案）。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import A2CAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> A2CAgent:
    n_envs  = config["n_envs"]
    n_steps = config["n_steps"]
    env_id  = config["env_id"]

    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_id) for _ in range(n_envs)]
    )
    eval_env = gym.make(env_id)

    state_dim  = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        n_steps=n_steps,
        gae_lambda=config["gae_lambda"],
        c_v=config["c_v"],
        c_e=config["c_e"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"a2c_vecenv_{env_id}")

    obs, _ = env.reset()
    ep_returns = np.zeros(n_envs)
    ep_lengths = np.zeros(n_envs, dtype=int)
    global_step = 0
    n_updates = config["total_steps"] // (n_envs * n_steps)

    print(f"A2C VecEnv 訓練：{env_id}，{n_envs} 個平行環境，"
          f"總步數 {config['total_steps']:,}，共 {n_updates:,} 次更新")

    for update in range(1, n_updates + 1):
        for _ in range(n_steps):
            actions = agent.select_actions(obs)
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated
            agent.store_reward_done(rewards, dones)

            ep_returns += rewards
            ep_lengths += 1
            global_step += n_envs

            for i in np.where(dones)[0]:
                logger.log_episode(ep_returns[i], ep_lengths[i], global_step)
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

            obs = next_obs

        metrics = agent.update(obs, dones)
        if metrics and global_step % config["log_freq"] == 0:
            logger.log_scalars(metrics, global_step)

        if update % config["eval_freq_updates"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, global_step)
            print(f"更新 {update:6d} | 步數 {global_step:8d} | 評估: {mean_r:.1f} ± {std_r:.1f}")

        if update % config["save_freq_updates"] == 0:
            agent.save(f"checkpoints/a2c_vecenv_step{global_step}")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id":            "CartPole-v1",
        "n_envs":            4,
        "total_steps":       200_000,
        "lr":                7e-4,
        "gamma":             0.99,
        "n_steps":           5,
        "gae_lambda":        0.95,
        "c_v":               0.5,
        "c_e":               0.01,
        "log_freq":          2_000,
        "eval_freq_updates": 500,   # 每 500 次更新 ≈ 每 10k 步
        "save_freq_updates": 2_500, # 每 2500 次更新 ≈ 每 50k 步
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    train(config)
