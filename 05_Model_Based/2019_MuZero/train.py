"""在 CartPole-v1 環境上訓練 MuZero。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import MuZeroAgent
from common.utils.logger import Logger


def train(config: dict) -> MuZeroAgent:
    env = gym.make(config["env_id"])

    agent = MuZeroAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        num_simulations=config["num_simulations"],
        gamma=config["gamma"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"muzero_{config['env_id']}")

    print("注意：這是一個 MuZero 的骨架實作版本 (Skeleton implementation)。")
    print("如需完整的實作版本，請參考：")
    print("  https://github.com/werner-duvaud/muzero-general")

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        game_history = []
        ep_return = ep_length = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            game_history.append({
                "obs": obs, "action": action,
                "reward": reward, "done": done,
            })
            ep_return += reward
            ep_length += 1
            obs = next_obs

        # 儲存完整的對局歷史記錄 (Full game history)
        agent.store(game_history)

        # 訓練更新 (Training update)
        metrics = agent.update()
        logger.log_episode(ep_return, ep_length, episode)

        if episode % 50 == 0:
            print(f"集數 {episode:5d}  回報: {ep_return:.1f}  緩衝區: {len(agent.replay_buffer)}")

    logger.close()
    env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "CartPole-v1",
        "n_episodes": 500,
        "hidden_dim": 64,
        "lr": 1e-3,
        "num_simulations": 25,
        "gamma": 0.997,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    train(config)
