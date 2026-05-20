"""
在具備目標條件 (Goal-conditioned) 的環境上訓練 HER (DDPG + 事後經驗重播)。

使用 Gymnasium Robotics 或 FetchReach 風格的 GoalEnv 介面。

參考文獻：
    Andrychowicz, M., et al. (2017). Hindsight Experience Replay. NeurIPS 2017.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import HERAgent
from common.utils.logger import Logger


def evaluate_goal_env(agent: HERAgent, env, n_episodes: int = 10) -> float:
    """評估成功率（實際達到的目標 achieved_goal 與預期目標 desired_goal 的距離在閾值內）。"""
    successes = 0
    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        done = False
        while not done:
            obs = obs_dict["observation"]
            goal = obs_dict["desired_goal"]
            action = agent.select_action(obs, goal, evaluate=True)
            obs_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("is_success", False):
            successes += 1
    return successes / n_episodes


def train(config: dict) -> HERAgent:
    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    # 從 GoalEnv 觀測空間中提取維度資訊 (Extract dimensions)
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["observation"].shape[0]
    goal_dim = obs_dict["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = HERAgent(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        her_ratio=config["her_ratio"],
        noise_std=config["noise_std"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"her_{config['env_id']}")

    print(f"正在 {config['env_id']} 環境上進行 HER 訓練，共 {config['n_epochs']} 個週期 (Epochs)...")
    print(f"  obs_dim={obs_dim}, goal_dim={goal_dim}, action_dim={action_dim}")

    for epoch in range(1, config["n_epochs"] + 1):
        # 收集回合資料 (Collect episodes)
        for _ in range(config["n_episodes_per_epoch"]):
            obs_dict, _ = env.reset()
            done = False

            while not done:
                obs = obs_dict["observation"]
                goal = obs_dict["desired_goal"]
                achieved = obs_dict["achieved_goal"]

                action = agent.select_action(obs, goal)
                next_obs_dict, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.buffer.store_transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs_dict["observation"],
                    done=done,
                    goal=goal,
                    achieved_goal=achieved,
                )
                obs_dict = next_obs_dict

        # 在每個 Epoch 的取樣結束後執行梯度更新 (Gradient updates)
        for _ in range(config["updates_per_epoch"]):
            metrics = agent.update()

        # 評估 (Evaluate)
        if epoch % config["eval_freq"] == 0:
            success_rate = evaluate_goal_env(agent, eval_env, n_episodes=10)
            logger.log_scalar("eval/success_rate", success_rate, epoch)
            print(f"週期 {epoch:5d}  成功率: {success_rate:.2%}")

        if epoch % config["save_freq"] == 0:
            agent.save(f"checkpoints/her_epoch{epoch}")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "FetchReach-v4",
        "n_epochs": 200,
        "n_episodes_per_epoch": 16,
        "updates_per_epoch": 40,
        "lr": 1e-3,
        "gamma": 0.98,
        "tau": 0.05,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "her_ratio": 0.8,
        "noise_std": 0.2,
        "eval_freq": 10,
        "save_freq": 50,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }
    train(config)
