"""
PER-DQN 代理人 — 結合優先經驗回放的 DQN。

參考文獻：
    Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015).
    Prioritized Experience Replay. ICLR 2016. arXiv:1511.05952.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.base_agent import BaseAgent
from common.buffers.priority_buffer import PrioritizedReplayBuffer
from common.utils.scheduler import LinearSchedule
from network import QNetwork


class PERDQNAgent(BaseAgent):
    """
    使用優先經驗回放 (PER) 增強的 DQN。

    相對於標準 DQN 的核心改動：
    1. 將均勻經驗回放池替換為優先經驗回放池 (基於 SumTree)。
    2. 根據 |TD 誤差|^alpha 的比例來取樣轉移資料。
    3. 使用重要性取樣 (IS) 權重對梯度更新進行加權。
    4. 在每次迷你批次更新後，同步更新轉移資料的優先權。

    IS 權重用於補償非均勻取樣分佈所產生的偏差，確保更新在極限情況下（beta -> 1）能保持無偏。
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
        alpha: float = 0.6,          # PER 優先權指數
        beta_start: float = 0.4,     # IS 權重指數 (逐漸增加至 1.0)
        beta_anneal_steps: int = 100_000,  # beta 線性退火的總步數
        use_cnn: bool = False,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.beta_anneal_steps = beta_anneal_steps

        self.online_net = QNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # 使用 PER 緩衝區而非均勻緩衝區
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=alpha, beta=beta_start
        )

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
        結合重要性取樣權重的 PER-DQN 更新步驟。

        步驟：
        1. 取樣具優先權的迷你批次 (包含索引與 IS 權重)
        2. 計算 Double DQN 目標
        3. 計算每個樣本的 TD 誤差
        4. 根據 IS 權重對損失進行加權
        5. 使用新的 TD 誤差更新 PER 優先權
        """
        if len(self.buffer) < self.batch_size:
            return {}

        # 將 beta 逐漸趨近 1.0 (在訓練過程中消除 IS 偏差)
        self.buffer.anneal_beta(self.total_steps, total_steps=self.beta_anneal_steps)

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        # 來自 PER 的 IS 修正權重
        weights = torch.FloatTensor(batch["weights"]).to(self.device)
        indices = batch["indices"]

        with torch.no_grad():
            # Double DQN target
            best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 每個樣本的 TD 誤差 (用於更新優先權)
        td_errors = (targets - current_q).detach().abs().cpu().numpy()

        # TODO: 加權 Huber 損失: mean(weights * huber(current_q, targets))
        elementwise_loss = nn.functional.smooth_l1_loss(
            current_q, targets, reduction="none"
        )
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # TODO: 使用新的 |TD 誤差| 更新優先權 (Priorities)
        self.buffer.update_priorities(indices, td_errors)

        self.total_steps += 1
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "loss": float(loss.item()),
            "mean_td_error": float(td_errors.mean()),
            "beta": self.buffer.beta,
            "epsilon": self.epsilon_schedule.get(self.total_steps),
        }

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "per_dqn.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "per_dqn.pt"), map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
