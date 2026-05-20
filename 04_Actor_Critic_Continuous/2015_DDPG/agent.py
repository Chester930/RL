"""
DDPG 代理人 — 深度確定性策略梯度 (Deep Deterministic Policy Gradient)。

參考文獻：
    Lillicrap, T. P., et al. (2015). Continuous Control with Deep
    Reinforcement Learning. ICLR 2016. arXiv:1509.02971.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from common.base_agent import BaseAgent
from common.buffers.replay_buffer import ReplayBuffer
from network import ActorNetwork, CriticNetwork


class OUNoise:
    """
    Ornstein-Uhlenbeck (OU) 雜訊過程，用於具有時間相關性的探索。

    OU 雜訊公式：dx = theta * (mu - x) * dt + sigma * dW
    提供平滑變化的雜訊，非常適合物理系統。
    """

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(action_dim)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state.copy()


class DDPGAgent(BaseAgent):
    """
    DDPG：使用確定性演員 (Deterministic actor) 將 DQN 擴充套件到連續動作空間。

    核心思想：
    - 確定性演員 pi(s) -> a（無取樣過程，無須計算對數機率）
    - 透過貝爾曼更新估計評論家 Q(s, a)
    - 演員與評論家均具備目標網路（採軟更新：tau << 1）
    - 使用 OU 雜訊進行探索（疊加於確定性動作之上）
    - 具備經驗回放緩衝區 (Experience replay buffer)

    引數：
        state_dim:    連續狀態維度。
        action_dim:   連續動作維度。
        action_scale: tanh 輸出的縮放因子（動作的最大幅值）。
        tau:          軟目標更新速率 (target = tau*online + (1-tau)*target)。
        noise_sigma:  OU 雜訊的標準差。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        noise_sigma: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # 演員網路 (Actor networks，線上 + 目標)
        self.actor = ActorNetwork(state_dim, action_dim, action_scale=action_scale).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.eval()

        # 評論家網路 (Critic networks，線上 + 目標)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.noise = OUNoise(action_dim, sigma=noise_sigma)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """指數移動平均更新：target = tau * source + (1.0 - tau) * target。"""
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * t_param.data)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        回傳確定性動作 + 探索雜訊。
        評估期間不加入雜訊。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if not evaluate:
            # 加入 OU 雜訊進行探索
            action = action + self.noise.sample()

        return np.clip(action, -self.action_scale, self.action_scale)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        DDPG 更新流程：
        1. 評論家：最小化 (Q(s,a) - y)^2，其中 y = r + gamma * Q_target(s', pi_target(s'))
        2. 演員：最大化 Q(s, pi(s))（確定性策略梯度）
        3. 對兩套目標網路執行軟更新 (Soft update)
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # --- 評論家更新 (Critic update) ---
        with torch.no_grad():
            # TODO: y = r + gamma * Q_target(s', pi_target(s'))
            next_actions = self.target_actor(next_states)
            next_q = self.target_critic(next_states, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.critic(states, actions).squeeze(1)
        critic_loss = nn.functional.mse_loss(current_q, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 演員更新 (Actor update) ---
        # TODO: 最大化 E[Q(s, pi(s))] = 最小化 -E[Q(s, pi(s))]
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 執行目標網路軟更新 (Soft update targets) ---
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        self.total_steps += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, os.path.join(path, "ddpg.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "ddpg.pt"), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
