"""
Double DQN 代理人。

透過將「動作選擇」與「價值估計」解耦，修正了 DQN 的高估偏差：
- 線上網路選擇最佳動作：a* = argmax_a Q_online(s', a)
- 目標網路評估該動作：  y  = r + gamma * Q_target(s', a*)

參考文獻：
    van Hasselt, H., Guez, A., & Silver, D. (2015). Deep Reinforcement Learning
    with Double Q-Learning. arXiv:1509.06461. AAAI 2016.
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


class DoubleDQNAgent(BaseAgent):
    """
    Double DQN — 除了目標計算方式外，其餘與 DQN 相同。

    唯一的變動之處：
        DQN:       y = r + gamma * max_{a'} Q_target(s', a')
        DoubleDQN: y = r + gamma * Q_target(s', argmax_{a'} Q_online(s', a'))

    當 max 運算元同時使用同一組數值來進行選擇與評估動作時，會產生高估問題，此方法可有效防止之。
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

        self.online_net = QNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
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
        Double DQN 更新步驟。

        與 DQN 在目標計算上的核心差異：
            a* = argmax_{a'} Q_online(s', a')     # 線上網路選擇動作
            y  = r + gamma * Q_target(s', a*)      # 目標網路評估價值
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
            # TODO: Double DQN 目標計算
            # 步驟 1: 使用 online_net 選擇最佳的下一動作
            best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # 步驟 2: 使用 target_net 評估該動作的價值 (已解耦！)
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
        }, os.path.join(path, "double_dqn.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "double_dqn.pt"), map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
