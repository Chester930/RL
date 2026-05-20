"""
IQL 代理人 — 離線強化學習 (Offline RL) 中的隱式 Q 學習 (Implicit Q-Learning)。

核心洞察：在訓練期間「從不」查詢分佈外 (OOD) 動作的 Q 值。
相反地，透過對 V(s) 執行期望分位數回歸 (Expectile regression) 來隱式地學習
資料集策略下的最大 Q 值，最後透過優勢加權行為複製 (Advantage-weighted BC) 提取策略。

包含三個獨立的損失函式：
    1. V 網路：期望分位數回歸 L_tau(Q(s,a) - V(s))
    2. Q 網路：使用 V(s') 進行貝爾曼備份 — 完全不需要 OOD 動作！
    3. 演員網路：優勢加權行為複製 (Advantage-weighted behavior cloning)

參考文獻：
    Kostrikov, I., Nair, A., & Levine, S. (2021).
    Offline RL with Implicit Q-Learning. ICLR 2022.
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
from network import ValueNetwork, TwinQNetwork, GaussianActor


def expectile_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """
    期望分位數回歸損失 (Expectile regression loss)。

    L_tau(u) = |tau - I(u<0)| * u^2

    當 tau > 0.5 時：對正殘差 (Residuals) 施加更多懲罰（上期望分位數）。
    隨著 tau 趨近於 1，期望分位數會趨向最大值，實現隱式最大 Q 學習。

    引數：
        u:   殘差 (目標值 - 預測值)。
        tau: (0, 1) 之間的期望分位數。論文預設使用 0.7。
    """
    weight = torch.where(u > 0, tau * torch.ones_like(u), (1 - tau) * torch.ones_like(u))
    return (weight * u.pow(2)).mean()


class IQLAgent(BaseAgent):
    """
    IQL：離線強化學習中的隱式 Q 學習。

    訓練時不需要任何線上環境互動。
    在呼叫 update() 之前，請先透過 load_dataset() 載入靜態資料集。

    引數：
        state_dim:     狀態維度。
        action_dim:    動作維度。
        hidden_dim:    MLP 隱藏層維度。
        lr:            學習率。
        gamma:         折扣因子。
        tau:           目標網路軟更新係數。
        expectile:     V 網路訓練的期望分位數（論文中的 tau，預設 0.7）。
        temperature:   優勢加權的逆溫度引數（論文中的 beta）。
        clip_score:    截斷優勢權重以防止訓練不穩定。
        device:        "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        temperature: float = 3.0,
        clip_score: float = 100.0,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.clip_score = clip_score

        # Networks
        self.vf = ValueNetwork(state_dim, hidden_dim).to(device)
        self.qf = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.qf_target = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.qf_target.load_state_dict(self.qf.state_dict())
        self.qf_target.eval()

        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(device)

        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=lr)
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.buffer = ReplayBuffer(capacity=2_000_000)

    def load_dataset(self, dataset: dict) -> None:
        """Load offline dataset (D4RL format or compatible dict)."""
        obs = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        next_obs = dataset["next_observations"]
        dones = dataset["terminals"]

        for i in range(len(obs)):
            self.buffer.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
        print(f"已載入 {len(obs)} 筆轉換資料至離線緩衝區。")

    # --- 執行動作 (Acting) ---

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.actor.get_action(s).squeeze(0).cpu().numpy()

    # --- 學習更新 (Learning) ---

    def update(self, batch_size: int = 256) -> dict:
        """
        一步 IQL 梯度更新 — 包含三個獨立的網路更新。

        V-更新： 對 Q - V 殘差進行期望分位數回歸。
        Q-更新： 使用 V(s') 作為目標進行貝爾曼備份（不涉及 OOD 動作！）。
        A-更新： 優勢加權行為複製。
        """
        if not self.buffer.is_ready(batch_size):
            return {}

        batch = self.buffer.sample(batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # --- V 網路更新：期望分位數回歸 (Expectile regression) ---
        with torch.no_grad():
            q1, q2 = self.qf_target(states, actions)
            q_min = torch.min(q1, q2)

        v = self.vf(states)
        vf_loss = expectile_loss(q_min - v, self.expectile)

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        # --- Q 網路更新：透過 V(s') 執行貝爾曼備份 ---
        with torch.no_grad():
            v_next = self.vf(next_states)
            q_target = rewards + self.gamma * (1 - dones) * v_next

        q1, q2 = self.qf(states, actions)
        qf_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # --- 演員網路更新：優勢加權行為複製 (Advantage-weighted BC) ---
        with torch.no_grad():
            v = self.vf(states)
            q1, q2 = self.qf_target(states, actions)
            adv = torch.min(q1, q2) - v
            # 優勢權重：exp(溫度 * 優勢)，並進行截斷以確保穩定
            weights = torch.exp(self.temperature * adv).clamp(max=self.clip_score)
            weights = weights / weights.mean()  # 歸一化權重 (Normalize)

        log_probs = self.actor.log_prob(states, actions)
        actor_loss = -(weights * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 目標網路軟更新 (Soft target update) ---
        for p, tp in zip(self.qf.parameters(), self.qf_target.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.total_steps += 1

        return {
            "vf_loss": float(vf_loss.item()),
            "qf_loss": float(qf_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "mean_v": float(v.mean().item()),
            "mean_adv": float(adv.mean().item()),
        }

    # --- 持久化 (Persistence) ---

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "vf": self.vf.state_dict(),
            "qf": self.qf.state_dict(),
            "actor": self.actor.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "iql_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "iql_checkpoint.pt"), map_location=self.device)
        self.vf.load_state_dict(ckpt["vf"])
        self.qf.load_state_dict(ckpt["qf"])
        self.actor.load_state_dict(ckpt["actor"])
        self.total_steps = ckpt["total_steps"]
