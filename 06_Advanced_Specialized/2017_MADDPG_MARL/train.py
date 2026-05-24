"""
在多代理人環境上訓練 MADDPG。

使用 PettingZoo 風格的並行環境 (Parallel env) API 或簡單的自定義多代理人環境。

參考文獻：
    Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np

from agent import MADDPGAgent
from common.utils.logger import Logger


class SimpleCoopEnv:
    """
    用於測試的極簡 2-代理人合作環境。

    兩個代理人都觀察同一個共享狀態，並接收合作型獎勵。
    TODO: 替換為 PettingZoo 或正式的自定義多代理人環境。
    """

    def __init__(self, n_agents: int = 2, obs_dim: int = 4, action_dim: int = 2):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.t = 0
        self.max_steps = 50

    def reset(self):
        self.state = np.random.randn(self.obs_dim).astype(np.float32)
        self.t = 0
        return [self.state.copy() for _ in range(self.n_agents)]

    def step(self, actions):
        self.t += 1
        joint_action = np.concatenate(actions)
        self.state = np.clip(self.state + joint_action[:self.obs_dim] * 0.1,
                             -1.0, 1.0).astype(np.float32)
        # 合作獎勵：所有代理人根據與原點的距離受罰 (Cooperative reward)
        reward = -float(np.linalg.norm(self.state))
        rewards = [reward for _ in range(self.n_agents)]
        done = self.t >= self.max_steps
        dones = [done for _ in range(self.n_agents)]
        next_obs = [self.state.copy() for _ in range(self.n_agents)]
        return next_obs, rewards, dones, {}


def train(config: dict) -> MADDPGAgent:
    n_agents = config["n_agents"]
    obs_dims = [config["obs_dim"]] * n_agents
    action_dims = [config["action_dim"]] * n_agents

    env = SimpleCoopEnv(n_agents=n_agents,
                        obs_dim=config["obs_dim"],
                        action_dim=config["action_dim"])

    agent = MADDPGAgent(
        obs_dims=obs_dims,
        action_dims=action_dims,
        hidden_dim=config["hidden_dim"],
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        gamma=config["gamma"],
        tau=config["tau"],
        noise_std=config["noise_std"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name="maddpg")

    print(f"正在為 {n_agents} 個代理人進行 MADDPG 訓練，共 {config['total_episodes']} 回合...")

    for ep in range(1, config["total_episodes"] + 1):
        obs_list = env.reset()
        ep_rewards = [0.0] * n_agents
        done = False

        while not done:
            actions = agent.select_actions(obs_list)
            next_obs_list, rewards, dones, _ = env.step(actions)

            agent.buffer.push(obs_list, actions, rewards, next_obs_list, dones)
            obs_list = next_obs_list

            for i in range(n_agents):
                ep_rewards[i] += rewards[i]
            done = all(dones)

            metrics = agent.update()

        if ep % config["log_freq"] == 0:
            mean_reward = np.mean(ep_rewards)
            logger.log_scalar("train/mean_reward", mean_reward, ep)
            print(f"回合 {ep:6d}  平均獎勵: {mean_reward:.2f}")

        if ep % config["save_freq"] == 0:
            agent.save(f"checkpoints/maddpg_ep{ep}")

    logger.close()
    return agent


if __name__ == "__main__":
    config = {
        "n_agents": 2,
        "obs_dim": 4,
        "action_dim": 2,
        "total_episodes": 50_000,
        "hidden_dim": 128,
        "lr_actor": 1e-4,
        "lr_critic": 1e-3,
        "gamma": 0.95,
        "tau": 0.01,
        "noise_std": 0.2,
        "buffer_size": 100_000,
        "batch_size": 256,
        "log_freq": 200,
        "save_freq": 2000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }
    train(config)
