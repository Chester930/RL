"""訓練 Dyna-Q 並比較不同的規劃步數。"""

import sys
import os
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import DynaQAgent


def train(config: dict) -> DynaQAgent:
    env = gym.make(config["env_id"])

    agent = DynaQAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        n_planning=config["n_planning"],
    )

    ep_returns = []

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            ep_return += reward
            obs = next_obs

        ep_returns.append(ep_return)

        if episode % config["log_freq"] == 0:
            avg = np.mean(ep_returns[-config["log_freq"]:])
            print(
                f"[K={config['n_planning']:3d}] "
                f"集數 {episode:5d}  平均回報: {avg:.3f}"
            )

    env.close()
    return agent


if __name__ == "__main__":
    base_config = {
        "env_id": "FrozenLake-v1",
        "n_episodes": 5000,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 0.1,
        "log_freq": 1000,
    }

    # 比較不同規劃步數 (Planning steps) 的效果
    for k in [0, 5, 10, 50]:
        print(f"\n--- Dyna-Q (規劃步數 K={k}) ---")
        config = {**base_config, "n_planning": k}
        train(config)

    print("\n訓練完成。")
