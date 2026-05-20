"""
在合作型多代理人環境中訓練 MAPPO。

參考文獻：
    Yu et al. (2021). The Surprising Effectiveness of PPO in Cooperative MARL.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np

from agent import MAPPOAgent
from common.utils.logger import Logger


class SimpleCoopEnv:
    """
    用於測試的極簡合作型多代理人環境。

    代理人觀察共享狀態與自身的代理人索引，並接收聯合獎勵 (Joint reward)。
    TODO: 研究用途請替換為 PettingZoo StarCraft 或 SMACv2。

    全域性狀態 (Global state) = 完整的共享狀態向量。
    區域性觀測 (Local obs)    = 狀態向量 + 獨熱編碼 (one-hot) 的代理人 ID。
    """

    def __init__(self, n_agents: int = 3, state_dim: int = 6, action_dim: int = 4,
                 max_steps: int = 50):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.t = 0

    @property
    def obs_dim(self):
        return self.state_dim + self.n_agents  # state + one-hot agent ID

    def reset(self):
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        self.t = 0
        return self._get_obs(), self.state.copy()

    def _get_obs(self):
        obs_list = []
        for i in range(self.n_agents):
            agent_id = np.zeros(self.n_agents, dtype=np.float32)
            agent_id[i] = 1.0
            obs_list.append(np.concatenate([self.state, agent_id]))
        return obs_list

    def step(self, actions):
        self.t += 1
        # 代理人透過「投票」來決定將狀態移向原點的方向 (Vote to move toward origin)
        vote = np.zeros(self.state_dim, dtype=np.float32)
        for a in actions:
            # 將離散動作對映至方向 (Map discrete action to direction)
            direction = np.zeros(self.state_dim, dtype=np.float32)
            direction[a % self.state_dim] = 1.0 if a < self.state_dim else -1.0
            vote += direction

        self.state = np.clip(self.state + 0.1 * vote / self.n_agents, -1.0, 1.0)
        joint_reward = -float(np.linalg.norm(self.state))
        rewards = [joint_reward] * self.n_agents
        done = self.t >= self.max_steps
        dones = [done] * self.n_agents
        return self._get_obs(), self.state.copy(), rewards, dones


def train(config: dict) -> MAPPOAgent:
    n_agents = config["n_agents"]
    env = SimpleCoopEnv(
        n_agents=n_agents,
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        max_steps=config["episode_len"],
    )

    obs_dims = [env.obs_dim] * n_agents
    global_state_dim = config["state_dim"]
    action_dims = [config["action_dim"]] * n_agents

    agent = MAPPOAgent(
        n_agents=n_agents,
        obs_dims=obs_dims,
        global_state_dim=global_state_dim,
        action_dims=action_dims,
        hidden_dim=config["hidden_dim"],
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_eps=config["clip_eps"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        entropy_coef=config["entropy_coef"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name="mappo")
    print(f"正在為 {n_agents} 個代理人訓練 MAPPO，總步數為 {config['total_steps']} 步...")

    step = 0
    episode = 0
    rollout_steps = config["rollout_steps"]

    while step < config["total_steps"]:
        agent.init_rollout(rollout_steps)
        ep_rewards = [0.0] * n_agents

        obs_list, global_state = env.reset()

        for t in range(rollout_steps):
            values = agent.get_values([global_state] * n_agents)
            actions, log_probs = agent.select_actions(obs_list)
            next_obs_list, global_state, rewards, dones = env.step(actions)

            agent.store_step(
                obs_list, global_state, actions, log_probs, rewards, values, dones
            )

            obs_list = next_obs_list
            for i in range(n_agents):
                ep_rewards[i] += rewards[i]
            step += 1

            if all(dones):
                episode += 1
                logger.log_scalar("train/mean_reward", np.mean(ep_rewards), step)
                ep_rewards = [0.0] * n_agents
                obs_list, global_state = env.reset()

        # 使用最後的價值進行引導 (Bootstrap last values)
        last_values = agent.get_values([global_state] * n_agents)
        metrics = agent.update(last_values)

        if step % config["log_freq"] == 0:
            logger.log_scalars(metrics, step)
            mean_rl = np.mean([metrics.get(f"agent{i}_actor_loss", 0)
                               for i in range(n_agents)])
            print(f"步數 {step:8d}  演員損失: {mean_rl:.4f}")

        if step % config["save_freq"] == 0:
            agent.save(f"checkpoints/mappo_step{step}")

    logger.close()
    return agent


if __name__ == "__main__":
    config = {
        "n_agents": 3,
        "state_dim": 6,
        "action_dim": 4,
        "episode_len": 50,
        "total_steps": 500_000,
        "rollout_steps": 400,  # 每次 PPO 更新前收集 400 步取樣資料
        "hidden_dim": 256,
        "lr_actor": 5e-4,
        "lr_critic": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "n_epochs": 10,
        "batch_size": 64,
        "entropy_coef": 0.01,
        "log_freq": 10_000,
        "save_freq": 50_000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }
    train(config)
