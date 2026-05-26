"""
MADDPG — 多代理人深度確定性策略梯度 (Multi-Agent Deep Deterministic Policy Gradient)。

核心思想：集中式訓練，分散式執行 (Centralized Training, Decentralized Execution, CTDE)。
    - 每個代理人都有自己的演員 (Actor)，執行時僅能看到區域性觀測 (Local obs)。
    - 每個代理人的評論家 (Critic) 在訓練時可以看到「所有」代理人的觀測 + 動作 (集中式)。
    - 測試或部署時，僅使用演員網路 (分散式)。

參考文獻：
    Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017).
    Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
    NeurIPS 2017. arXiv:1706.02275.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from network import AgentActor, CentralizedCritic


class MultiAgentReplayBuffer:
    """
    用於多代理人環境的重播緩衝區。

    儲存完整的聯合轉換資料 (Joint transitions)：包含所有代理人的觀測、動作、獎勵、下一狀態與結束標記。
    """

    def __init__(self, capacity: int, n_agents: int):
        self.buffer = deque(maxlen=capacity)
        self.n_agents = n_agents

    def push(self, obs_list, action_list, reward_list, next_obs_list, done_list):
        """
        引數：
            obs_list:      n_agents 個觀測陣列的列表。
            action_list:   n_agents 個動作陣列的列表。
            reward_list:   n_agents 個標量獎勵的列表。
            next_obs_list: n_agents 個下一觀測陣列的列表。
            done_list:     n_agents 個結束標記 (Done flags) 的列表。
        """
        self.buffer.append((
            [o.copy() for o in obs_list],
            [a.copy() for a in action_list],
            list(reward_list),
            [o.copy() for o in next_obs_list],
            list(done_list),
        ))

    def sample(self, batch_size: int) -> list:
        """回傳每個代理人的資料字典列表。"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs_b = [[] for _ in range(self.n_agents)]
        act_b = [[] for _ in range(self.n_agents)]
        rew_b = [[] for _ in range(self.n_agents)]
        nobs_b = [[] for _ in range(self.n_agents)]
        done_b = [[] for _ in range(self.n_agents)]

        for obs_l, act_l, rew_l, nobs_l, done_l in batch:
            for i in range(self.n_agents):
                obs_b[i].append(obs_l[i])
                act_b[i].append(act_l[i])
                rew_b[i].append(rew_l[i])
                nobs_b[i].append(nobs_l[i])
                done_b[i].append(done_l[i])

        return [
            {
                "obs": np.array(obs_b[i], dtype=np.float32),
                "actions": np.array(act_b[i], dtype=np.float32),
                "rewards": np.array(rew_b[i], dtype=np.float32),
                "next_obs": np.array(nobs_b[i], dtype=np.float32),
                "dones": np.array(done_b[i], dtype=np.float32),
            }
            for i in range(self.n_agents)
        ]

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


class MADDPGAgent:
    """
    MADDPG：管理所有代理人的演員 (Actors) 與評論家 (Critics)。

    引數：
        obs_dims:    每個代理人的觀測維度列表。
        action_dims: 每個代理人的動作維度列表。
        hidden_dim:  MLP 隱藏層維度。
        lr_actor:    演員學習率。
        lr_critic:   評論家學習率。
        gamma:       折扣因子。
        tau:         目標網路軟更新率。
        noise_std:   探索雜訊標準差。
        buffer_size: 重播緩衝區容量。
        batch_size:  小批次 (Batch) 維度。
        device:      "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        obs_dims: list,
        action_dims: list,
        hidden_dim: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        noise_std: float = 0.2,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.n_agents = len(obs_dims)
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.device = device

        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)

        # 每個代理人的演員與目標網路 (Per-agent actor + target)
        self.actors = nn.ModuleList([
            AgentActor(obs_dims[i], action_dims[i], hidden_dim).to(device)
            for i in range(self.n_agents)
        ])
        self.actors_target = nn.ModuleList([
            AgentActor(obs_dims[i], action_dims[i], hidden_dim).to(device)
            for i in range(self.n_agents)
        ])
        for i in range(self.n_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

        # 每個代理人的集中式評論家與目標網路 (Per-agent centralized critic + target)
        self.critics = nn.ModuleList([
            CentralizedCritic(total_obs_dim, total_action_dim, hidden_dim).to(device)
            for _ in range(self.n_agents)
        ])
        self.critics_target = nn.ModuleList([
            CentralizedCritic(total_obs_dim, total_action_dim, hidden_dim).to(device)
            for _ in range(self.n_agents)
        ])
        for i in range(self.n_agents):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.actor_optimizers = [
            optim.Adam(self.actors[i].parameters(), lr=lr_actor)
            for i in range(self.n_agents)
        ]
        self.critic_optimizers = [
            optim.Adam(self.critics[i].parameters(), lr=lr_critic)
            for i in range(self.n_agents)
        ]

        self.buffer = MultiAgentReplayBuffer(buffer_size, self.n_agents)

    # --- 執行動作 (Acting) ---

    def select_actions(self, obs_list: list, evaluate: bool = False) -> list:
        """
        給定每個代理人的區域性觀測，選擇其對應的動作。

        引數：
            obs_list: 包含 n_agents 個 numpy 陣列的列表。
        回傳：
            包含 n_agents 個動作陣列的列表。
        """
        actions = []
        for i, obs in enumerate(obs_list):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actors[i](obs_t).squeeze(0).cpu().numpy()
            if not evaluate:
                action += np.random.randn(*action.shape) * self.noise_std
                action = np.clip(action, -1.0, 1.0)
            actions.append(action)
        return actions

    # --- 學習更新 (Learning) ---

    def update(self) -> dict:
        """
        為所有代理人執行一步 MADDPG 梯度更新。

        評論家 (Critic)：使用集中式資訊（所有人的觀測 + 所有人的動作）進行訓練。
        演員 (Actor)：從自己的評論家獲取梯度，同時將其他人的動作視為固定常數。
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batches = self.buffer.sample(self.batch_size)
        B = self.batch_size

        # 轉換為張量 (Convert to tensors)
        all_obs = [
            torch.FloatTensor(batches[i]["obs"]).to(self.device)
            for i in range(self.n_agents)
        ]
        all_acts = [
            torch.FloatTensor(batches[i]["actions"]).to(self.device)
            for i in range(self.n_agents)
        ]
        all_rews = [
            torch.FloatTensor(batches[i]["rewards"]).to(self.device)
            for i in range(self.n_agents)
        ]
        all_nobs = [
            torch.FloatTensor(batches[i]["next_obs"]).to(self.device)
            for i in range(self.n_agents)
        ]
        all_dones = [
            torch.FloatTensor(batches[i]["dones"]).to(self.device)
            for i in range(self.n_agents)
        ]

        # 聯合張量 (Joint tensors)
        joint_obs = torch.cat(all_obs, dim=-1)        # (B, total_obs)
        joint_acts = torch.cat(all_acts, dim=-1)      # (B, total_act)
        joint_nobs = torch.cat(all_nobs, dim=-1)

        # 來自目標演員網路的目標動作 (Target actions from target actors)
        with torch.no_grad():
            target_acts = [
                self.actors_target[i](all_nobs[i])
                for i in range(self.n_agents)
            ]
            joint_target_acts = torch.cat(target_acts, dim=-1)

        metrics = {}
        for i in range(self.n_agents):
            # --- 評論家 (Critic) 更新 ---
            with torch.no_grad():
                q_next = self.critics_target[i](joint_nobs, joint_target_acts)
                q_target = all_rews[i] + self.gamma * (1 - all_dones[i]) * q_next

            q_pred = self.critics[i](joint_obs, joint_acts)
            critic_loss = nn.functional.mse_loss(q_pred, q_target)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), max_norm=0.5)
            self.critic_optimizers[i].step()

            # --- 演員 (Actor) 更新 ---
            # 固定其他代理人的動作；僅最佳化目前代理人 i 的演員網路
            current_acts = []
            for j in range(self.n_agents):
                if j == i:
                    current_acts.append(self.actors[i](all_obs[i]))
                else:
                    current_acts.append(all_acts[j].detach())
            joint_current = torch.cat(current_acts, dim=-1)

            actor_loss = -self.critics[i](joint_obs, joint_current).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=0.5)
            self.actor_optimizers[i].step()

            metrics[f"agent{i}_critic_loss"] = float(critic_loss.item())
            metrics[f"agent{i}_actor_loss"] = float(actor_loss.item())

        # --- 目標網路軟更新 (Soft target updates) ---
        for i in range(self.n_agents):
            for p, tp in zip(self.actors[i].parameters(),
                             self.actors_target[i].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.critics[i].parameters(),
                             self.critics_target[i].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return metrics

    # --- 持久化 (Persistence) ---

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
        }, os.path.join(path, "maddpg_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "maddpg_checkpoint.pt"), map_location=self.device)
        for i, sd in enumerate(ckpt["actors"]):
            self.actors[i].load_state_dict(sd)
            self.actors_target[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["critics"]):
            self.critics[i].load_state_dict(sd)
            self.critics_target[i].load_state_dict(sd)

    def save_resume(self, path: str) -> None:
        """儲存暫停點，供關機後續跑使用。"""
        self.save(path)

    def load_resume(self, path: str) -> None:
        """載入暫停點。"""
        self.load(path)
