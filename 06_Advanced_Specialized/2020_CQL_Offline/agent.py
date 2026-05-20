"""
CQL 代理人 — 離線強化學習 (Offline RL) 中的保守 Q 學習 (Conservative Q-Learning)。

核心思想：在標準貝爾曼更新 (Bellman update) 中加入保守懲罰項 (Conservative penalty)，
防止策略選擇那些在資料集中未出現過、且可能被錯誤估計為高 Q 值的「分佈外 (OOD)」動作。

CQL 損失 = 標準 SAC 損失 + alpha * CQL 懲罰項
CQL 懲罰項 = E_{a~pi}[Q(s,a)] - E_{a~dataset}[Q(s,a)]

這能確保 Q 值估計是策略下真實價值的「下界 (Lower bound)」，僅讓資料集中出現過的動作維持較高的 Q 值。

參考文獻：
    Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020).
    Conservative Q-Learning for Offline RL. NeurIPS 2020.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.base_agent import BaseAgent
from common.buffers.replay_buffer import ReplayBuffer
from network import TwinQNetwork, GaussianPolicy


class CQLAgent(BaseAgent):
    """
    搭配 SAC 風格策略的保守 Q 學習 (CQL)。

    專為離線強化學習 (Offline RL) 設計：完全在固定的資料集上訓練，
    在訓練過程中不與環境進行任何互動。

    引數：
        state_dim:      狀態維度。
        action_dim:     動作維度。
        hidden_dim:     MLP 隱藏層維度。
        lr:             Q 網路與策略的學習率。
        gamma:          折扣因子。
        tau:            目標網路軟更新係數。
        alpha:          SAC 溫度引數（熵正規化）。
        auto_alpha:     自動調整 alpha。
        cql_alpha:      CQL 正規化權重（論文中的 alpha）。
        cql_n_actions:  用於估計 CQL 懲罰項的隨機動作數量。
        cql_lagrange:   使用拉格朗日乘子 (Lagrange multiplier) 自動調整 cql_alpha。
        cql_target_action_gap: 拉格朗日調整的目標動作間隙 (Target action gap)。
        device:         "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        cql_alpha: float = 5.0,
        cql_n_actions: int = 10,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.cql_lagrange = cql_lagrange

        # 網路結構 (Networks)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # 自動調整 Alpha / SAC 溫度引數 (Auto-alpha)
        self.auto_alpha = auto_alpha
        self.target_entropy = -float(action_dim)
        if auto_alpha:
            self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # CQL 的拉格朗日乘子（選用，用於自動調整保守權重）
        if cql_lagrange:
            self.log_cql_alpha = nn.Parameter(torch.zeros(1, device=device))
            self.cql_alpha_optimizer = optim.Adam([self.log_cql_alpha], lr=lr)
            self.cql_target_action_gap = cql_target_action_gap

        # 離線資料集緩衝區（由 load_dataset 填充）
        self.buffer = ReplayBuffer(capacity=2_000_000)

    def load_dataset(self, dataset: dict) -> None:
        """
        載入離線資料集（例如來自 D4RL）。

        引數：
            dataset: 包含鍵值 "observations", "actions", "rewards",
                     "next_observations", "terminals" 的字典。
        """
        obs = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        next_obs = dataset["next_observations"]
        dones = dataset["terminals"]

        for i in range(len(obs)):
            self.buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

        print(f"已載入 {len(obs)} 筆轉換資料至離線緩衝區。")

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> np.ndarray:
        """評估時使用確定性動作；資料收集時使用隨機動作。"""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                return self.actor.get_action(s).squeeze(0).cpu().numpy()
            action, _ = self.actor.sample(s)
            return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # CQL penalty computation
    # ------------------------------------------------------------------

    def _cql_penalty(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        CQL 懲罰項 = logsumexp(隨機動作下的 Q 值) - Q(資料集內的動作)。

        Logsumexp 是透過均勻分佈的重要性取樣 (Importance sampling) 來估計的：
            logsumexp Q(s, a~uniform) ≈ log mean exp Q(s, a_i)  i=1..N

        包含兩個項：
            (1) E_{a ~ pi(a|s)}[Q(s,a)]     <- 會被不良的分佈外 (OOD) 動作極大化
            (2) E_{a ~ dataset}[Q(s,a)]      <- 應該保持較高的 Q 值
        懲罰項 = (1) - (2)  ->  效果是壓低 (1)，推高 (2)
        """
        B = states.shape[0]

        # 從目前策略取樣動作 (Sample actions from policy)
        repeated_states = states.unsqueeze(1).repeat(1, self.cql_n_actions, 1).view(
            B * self.cql_n_actions, -1
        )
        policy_actions, policy_log_probs = self.actor.sample(repeated_states)
        policy_log_probs = policy_log_probs.view(B, self.cql_n_actions)

        # 從均勻分佈取樣隨機動作 (Sample random actions)
        rand_actions = torch.FloatTensor(
            B * self.cql_n_actions, self.action_dim
        ).uniform_(-1, 1).to(self.device)
        rand_log_probs = torch.log(
            torch.ones(B, self.cql_n_actions, device=self.device)
            / (2.0 ** self.action_dim)
        )

        # 計算策略動作與隨機動作的 Q 值 (Q-values)
        q1_policy, q2_policy = self.critic(repeated_states, policy_actions)
        q1_policy = q1_policy.view(B, self.cql_n_actions)
        q2_policy = q2_policy.view(B, self.cql_n_actions)

        q1_rand, q2_rand = self.critic(repeated_states, rand_actions)
        q1_rand = q1_rand.view(B, self.cql_n_actions)
        q2_rand = q2_rand.view(B, self.cql_n_actions)

        # 重要性加權的 logsumexp (Importance-weighted)
        q1_is = torch.cat([
            q1_policy - policy_log_probs,
            q1_rand - rand_log_probs,
        ], dim=1)
        q2_is = torch.cat([
            q2_policy - policy_log_probs,
            q2_rand - rand_log_probs,
        ], dim=1)

        # 在增強後的集合上進行 logsumexp 計算
        cql_q1 = torch.logsumexp(q1_is, dim=1)
        cql_q2 = torch.logsumexp(q2_is, dim=1)

        # 扣除資料集中的 Q 值 (Subtract dataset Q-values)
        q1_data, q2_data = self.critic(states, actions)
        cql_penalty_q1 = (cql_q1 - q1_data).mean()
        cql_penalty_q2 = (cql_q2 - q2_data).mean()

        return cql_penalty_q1, cql_penalty_q2

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, batch_size: int = 256) -> dict:
        """
        一步 CQL 梯度更新。

        總評論家損失 = 貝爾曼誤差 (Bellman error) + cql_alpha * CQL 懲罰項
        """
        if not self.buffer.is_ready(batch_size):
            return {}

        batch = self.buffer.sample(batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # --- 標準 SAC 貝爾曼目標 (Standard SAC Bellman target) ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q_next = self.critic_target.q_min(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_probs)

        q1, q2 = self.critic(states, actions)
        bellman_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)

        # --- CQL 懲罰項 (CQL penalty) ---
        cql_p1, cql_p2 = self._cql_penalty(states, actions)

        if self.cql_lagrange:
            cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=0.0, max=1e6).item()
        else:
            cql_alpha = self.cql_alpha

        critic_loss = bellman_loss + cql_alpha * (cql_p1 + cql_p2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 拉格朗日自動調整 (Lagrange update) ---
        if self.cql_lagrange:
            action_gap = (cql_p1 + cql_p2).detach() / 2.0
            cql_alpha_loss = self.log_cql_alpha * (
                action_gap - self.cql_target_action_gap
            )
            self.cql_alpha_optimizer.zero_grad()
            (-cql_alpha_loss).backward()
            self.cql_alpha_optimizer.step()

        # --- 演員更新 (Actor update - SAC 風格) ---
        new_actions, log_probs = self.actor.sample(states)
        q_actor = self.critic.q_min(states, new_actions)
        actor_loss = (self.alpha * log_probs - q_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 溫度引數更新 (Alpha update) ---
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- 目標網路軟更新 (Soft target update) ---
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.total_steps += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "cql_penalty": float((cql_p1 + cql_p2).item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": self.alpha,
        }

    # --- 持久化 (Persistence) ---

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "cql_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "cql_checkpoint.pt"), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.total_steps = ckpt["total_steps"]
