"""
Dueling DQN 代理人。

使用獨立的價值流 (Value) 與優勢流 (Advantage) 來更好地估計 Q(s,a)。

參考文獻：
    Wang, Z., et al. (2016). Dueling Network Architectures for Deep
    Reinforcement Learning. ICML 2016. arXiv:1511.06581.
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
from network import DuelingQNetwork


class DuelingDQNAgent(BaseAgent):
    """
    Dueling DQN 結合了：
    - Dueling 網路架構 (本檔案)
    - Double DQN 目標計算 (解耦選擇與評估)
    - 經驗回放 + 目標網路

    網路架構的變動是相對於 Double DQN 的唯一修改。
    其餘部分 (緩衝區、目標網路、更新規則) 皆完全相同。
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

        # 使用 Dueling 網路而非普通的 Q 網路 (QNetwork)
        self.online_net = DuelingQNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.epsilon_schedule = LinearSchedule(
            start=epsilon_start, end=epsilon_end, total_steps=epsilon_steps
        )

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        epsilon = 0.0 if evaluate else self.epsilon_schedule.get(self.total_steps)
        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_dim))
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online_net(state_t).argmax(dim=1).item())

    def update(self) -> dict:
        """
        結合 Dueling 網路的 Double DQN 更新。

        網路輸出為 Q(s,a) = V(s) + A(s,a) - mean(A(s,:))，
        但其更新規則與 Double DQN 完全相同。
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        with torch.no_grad():
            # Double DQN 目標計算 (線上網路選擇，目標網路評估)
            best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "loss": float(loss.item()),
            "mean_q": float(current_q.mean().item()),
            "epsilon": self.epsilon_schedule.get(self.total_steps),
        }

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "dueling_dqn.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "dueling_dqn.pt"), map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
