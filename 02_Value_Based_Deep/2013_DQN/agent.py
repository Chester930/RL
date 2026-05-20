"""
DQN 代理人 — 深度 Q 網路 (Deep Q-Network)。

相對於表格型 Q-Learning 的兩大核心創新：
1. 經驗回放 (Experience Replay)：透過隨機取樣迷你批次 (Mini-batches) 來打破樣本間的相關性。
2. 目標網路 (Target Network)：透過定期更新的副本來穩定訓練過程。

參考文獻：
    Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning.
    Mnih et al. (2015). Human-level control through deep reinforcement learning.
    Nature 518, 529-533.
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
from network import QNetwork


class DQNAgent(BaseAgent):
    """
    具備經驗回放與目標網路的 DQN。

    引數：
        state_dim:     狀態維度（或 CNN 的影格數）。
        action_dim:    離散動作的數量。
        lr:            Adam 學習率。
        gamma:         折扣因子 (Discount factor)。
        buffer_size:   經驗回放池容量。
        batch_size:    每次梯度更新的迷你批次大小。
        target_update: 每隔 N 步同步一次目標網路。
        epsilon_start: 初始 Epsilon。
        epsilon_end:   最終 Epsilon。
        epsilon_steps: Epsilon 衰減的總步數。
        use_cnn:       使用 NatureCNN 骨幹網路（Atari 環境設為 True）。
        device:        "cpu" 或 "cuda"（顯示卡加速）。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        target_update: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_steps: int = 100_000,
        use_cnn: bool = False,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # --- Networks ---
        self.online_net = QNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # 目標網路從不直接參與訓練

        # --- Optimizer ---
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(capacity=buffer_size)

        # --- Epsilon schedule ---
        self.epsilon_schedule = LinearSchedule(
            start=epsilon_start, end=epsilon_end, total_steps=epsilon_steps
        )

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Epsilon-greedy 動作選擇。

        在評估期間 (evaluate=True) 始終選擇貪婪動作。
        """
        epsilon = 0.0 if evaluate else self.epsilon_schedule.get(self.total_steps)

        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        取樣一個迷你批次並執行一次梯度更新。

        DQN TD 目標：
            y = r + gamma * max_{a'} Q_target(s', a')   (若未結束)
            y = r                                         (若已結束)

        損失函式 (Loss)：
            L = mean( (y - Q_online(s, a))^2 )

        回傳：
            指標字典，包含 "loss" (損失值) 與 "mean_q" (平均 Q 值)。
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        # --- 計算 TD 目標 ---
        with torch.no_grad():
            # TODO: target = r + gamma * max_{a'} Q_target(s', a') * (1 - done)
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        # --- 當前採取的動作之 Q 值 ---
        # TODO: Q_online(s, a) selected by action index
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Huber 損失 (平滑 L1 損失) ---
        loss = nn.functional.smooth_l1_loss(current_q, targets)

        # --- 梯度更新步驟 ---
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (有助於穩定性)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1

        # --- 同步目標網路 ---
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "loss": float(loss.item()),
            "mean_q": float(current_q.mean().item()),
            "epsilon": self.epsilon_schedule.get(self.total_steps),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "dqn_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "dqn_checkpoint.pt"), map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
