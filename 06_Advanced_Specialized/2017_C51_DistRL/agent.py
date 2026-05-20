"""
C51 代理人 — 分類式 (Categorical) 分散式 DQN。

核心思想：將分散式貝爾曼目標投影至固定的原子支撐集 (Atom support) 上，
接著最小化預測分佈與投影分佈之間的交叉熵 (Cross-entropy)。

參考文獻：
    Bellemare, M. G., Dabney, W., & Munos, R. (2017).
    A Distributional Perspective on Reinforcement Learning. ICML 2017.
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
from common.utils.scheduler import LinearSchedule
from network import CategoricalQNetwork


class C51Agent(BaseAgent):
    """
    分類式 DQN (C51) 代理人。

    與一般 DQN 的主要區別：
        - 預測的是回報分佈 (Return distribution)，而非純量 Q 值。
        - 貝爾曼更新：將 T_z (貝爾曼原子) 投影至固定的支撐集上。
        - 損失函式：交叉熵 (projected_target, predicted_distribution)。

    引數：
        state_dim:     狀態維度。
        action_dim:    離散動作數量。
        n_atoms:       分類原子的數量（預設為 51）。
        v_min:         原子支撐集的回報最小值。
        v_max:         原子支撐集的回報最大值。
        lr:            Adam 學習率。
        gamma:         折扣因子。
        buffer_size:   重播緩衝區容量。
        batch_size:    小批次 (Batch) 維度。
        target_update: 目標網路同步頻率（步數）。
        epsilon_start: 初始探索率 (Epsilon)。
        epsilon_end:   最終探索率。
        epsilon_steps: 探索率衰減時程。
        device:        "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        lr: float = 6.25e-5,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        target_update: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_steps: int = 100_000,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # 網路結構 (Networks)
        self.online_net = CategoricalQNetwork(
            state_dim, action_dim, n_atoms, v_min, v_max
        ).to(device)
        self.target_net = CategoricalQNetwork(
            state_dim, action_dim, n_atoms, v_min, v_max
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=1.5e-4)
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.epsilon_schedule = LinearSchedule(
            start=epsilon_start, end=epsilon_end, total_steps=epsilon_steps
        )

        # 原子支撐集 (Atom support)
        self.atoms = torch.linspace(v_min, v_max, n_atoms, device=device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

    # --- 執行動作 (Acting) ---

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        epsilon = 0.0 if evaluate else self.epsilon_schedule.get(self.total_steps)

        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_dim))

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net.get_q_values(s)
        return int(q_values.argmax(dim=1).item())

    # --- 分散式貝爾曼投影 (Distributional Bellman projection) ---

    @torch.no_grad()
    def _project_distribution(
        self,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        將貝爾曼更新後的分佈投影至固定的原子支撐集上。

        演演算法流程：
            1. 針對每個原子 z_j 計算 T_z_j = r + gamma*(1-done)*z_j。
            2. 將 T_z_j 限制在 [v_min, v_max] 範圍內。
            3. 計算 b_j = (T_z_j - v_min) / delta_z (小數索引 / Fractional index)。
            4. 按比例將機率質量分配給 floor(b_j) 與 ceil(b_j)。

        回傳：
            m: (batch, n_atoms) 投影後的目標分佈。
        """
        batch_size = next_states.shape[0]

        # 從線上網路選擇貪婪動作（Double-C51 風格）
        next_q = self.online_net.get_q_values(next_states)
        next_actions = next_q.argmax(dim=1)  # (batch,)

        # 目標網路針對所選動作的分佈機率 (Target probs)
        target_probs = self.target_net(next_states)  # (batch, A, N)
        target_probs = target_probs[range(batch_size), next_actions]  # (batch, N)

        # 計算投影後的原子：T_z_j = r + gamma*(1-d)*z_j
        atoms = self.atoms.unsqueeze(0)                    # (1, N)
        rewards = rewards.unsqueeze(1)                     # (batch, 1)
        dones = dones.unsqueeze(1)                         # (batch, 1)
        Tz = rewards + self.gamma * (1 - dones) * atoms   # (batch, N)
        Tz = Tz.clamp(self.v_min, self.v_max)

        # 小數索引 (Fractional indices)
        b = (Tz - self.v_min) / self.delta_z              # (batch, N)
        lower = b.floor().long()
        upper = b.ceil().long()

        # 將索引限制在有效範圍內 (Clamp indices)
        lower = lower.clamp(0, self.n_atoms - 1)
        upper = upper.clamp(0, self.n_atoms - 1)

        # 分配機率質量 (Distribute mass)
        m = torch.zeros(batch_size, self.n_atoms, device=self.device)

        # 分配給下方桶位：p_j * (u - b)
        # 分配給上方桶位：p_j * (b - l)
        offset = (
            torch.arange(batch_size, device=self.device).unsqueeze(1) * self.n_atoms
        )  # (batch, 1)

        m.view(-1).scatter_add_(
            0,
            (lower + offset).view(-1),
            (target_probs * (upper.float() - b)).view(-1),
        )
        m.view(-1).scatter_add_(
            0,
            (upper + offset).view(-1),
            (target_probs * (b - lower.float())).view(-1),
        )

        return m

    # --- 學習更新 (Learning) ---

    def update(self) -> dict:
        """
        使用分散式貝爾曼交叉熵損失 (Cross-entropy loss) 執行一步梯度更新。

        損失函式 = -sum_j m_j * log p_j(s, a)
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # 投影後的目標分佈 (Projected target distribution)
        m = self._project_distribution(next_states, rewards, dones)  # (B, N)

        # 所選動作的對數機率 (Log-probs for chosen actions)
        log_p = self.online_net.log_probs(states)          # (B, A, N)
        log_p = log_p[range(self.batch_size), actions]     # (B, N)

        # 交叉熵損失：-sum_j m_j * log p_j
        loss = -(m * log_p).sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1

        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "loss": float(loss.item()),
            "epsilon": self.epsilon_schedule.get(self.total_steps),
        }

    # --- 持久化 (Persistence) ---

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "c51_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "c51_checkpoint.pt"), map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
