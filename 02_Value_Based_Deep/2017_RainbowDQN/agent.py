"""
Rainbow DQN 代理人 — 整合了相對於原始 DQN 的 6 項重要改進。

參考文獻：
    Hessel, M., et al. (2017). Rainbow: Combining Improvements in Deep
    Reinforcement Learning. AAAI 2018. arXiv:1710.02298.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.base_agent import BaseAgent
from common.buffers.priority_buffer import PrioritizedReplayBuffer
from network import RainbowNetwork


class RainbowAgent(BaseAgent):
    """
    Rainbow = Double DQN + PER + Dueling + NoisyNet + C51 + N-步回報。

    探索 (Exploration)：NoisyNet (網路權重內建雜訊 -> 不需要 epsilon-greedy)
    經驗 (Experience)： 優先經驗回放池 (PER)
    目標 (Target)：      Double DQN (線上網路選擇，目標網路評估)
    網路 (Network)：     Dueling + 分散式 (C51 原子)
    預見 (Lookahead)：   N-步回報

    引數：
        n_atoms:    C51 的支撐原子數量 (論文中為 51)。
        v_min/max:  價值支撐範圍。
        n_step:     N-步回報的展望長度。
        alpha/beta: PER 引數。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 6.25e-5,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        target_update: int = 8000,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        n_step: int = 3,
        alpha: float = 0.5,
        beta: float = 0.4,
        beta_anneal_steps: int = 100_000,
        use_cnn: bool = False,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.n_atoms = n_atoms
        self.n_step = n_step

        self.online_net = RainbowNetwork(
            state_dim, action_dim, n_atoms, v_min, v_max, use_cnn
        ).to(self.device)
        self.target_net = RainbowNetwork(
            state_dim, action_dim, n_atoms, v_min, v_max, use_cnn
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=1.5e-4)

        # PER 緩衝區 (改進項 #2)
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=alpha, beta=beta
        )
        self.beta_anneal_steps = beta_anneal_steps

        # N-步回報緩衝區 (改進項 #6)
        self._n_step_buffer = []
        self._gamma_n = gamma ** n_step

        # 將支撐集註冊為緩衝區參考 (Buffer reference)
        self.support = self.online_net.support

    # ------------------------------------------------------------------
    # 採取動作 (無須 Epsilon — 由 NoisyNet 處理探索)
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """對期望 Q 值採取貪婪策略；NoisyNet 提供探索能力。"""
        if evaluate:
            self.online_net.eval()
        else:
            self.online_net.train()
            self.online_net.sample_noise()

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net.get_q_values(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # N-step buffer
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done):
        """存入 n-步緩衝區；當累積滿 n 步時，將其推入 PER 池中。"""
        self._n_step_buffer.append((state, action, reward, next_state, done))
        if len(self._n_step_buffer) < self.n_step and not done:
            return

        # 計算 n-步回報 G = r_1 + gamma*r_2 + ... + gamma^{n-1}*r_n
        G = 0.0
        s0, a0 = self._n_step_buffer[0][0], self._n_step_buffer[0][1]
        for i, (_, _, r, ns, d) in enumerate(self._n_step_buffer):
            G += (self.gamma ** i) * r
            if d:
                next_state = ns
                done = True
                break
            next_state = ns

        self.buffer.push(s0, a0, G, next_state, done)
        self._n_step_buffer.pop(0)

    # ------------------------------------------------------------------
    # 學習 — 分散式 (C51) 損失
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        結合分散式貝爾曼投影的 C51 交叉熵損失。

        步驟：
        1. 計算分散式目標 (投影至原子位置)
        2. 計算線上分佈與投影目標分佈之間的交叉熵損失
        3. 使用 PER 的 IS 權重進行加權
        4. 更新優先權

        回傳：
            包含 "loss" 的指標字典。
        """
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        weights = torch.FloatTensor(batch["weights"]).to(self.device)
        indices = batch["indices"]

        self.online_net.sample_noise()
        self.target_net.sample_noise()

        with torch.no_grad():
            # TODO: C51 分散式貝爾曼投影 (Distributional Bellman Projection)
            # 1. Double DQN: 使用線上網路選擇最佳動作
            next_q = self.online_net.get_q_values(next_states)
            best_actions = next_q.argmax(dim=1)

            # 2. 使用目標網路的分佈進行評估
            next_probs = self.target_net(next_states)  # (batch, A, n_atoms)
            next_probs = next_probs[range(self.batch_size), best_actions]  # (batch, n_atoms)

            # 3. 將原子投影至目標分佈
            #    z' = r + gamma_n * z (限制在 [v_min, v_max] 區間)
            gamma_n = self._gamma_n
            z = self.support.unsqueeze(0)          # (1, n_atoms)
            rewards_ = rewards.unsqueeze(1)         # (batch, 1)
            dones_ = dones.unsqueeze(1)             # (batch, 1)

            Tz = rewards_ + gamma_n * (1 - dones_) * z
            Tz = Tz.clamp(self.online_net.v_min, self.online_net.v_max)

            delta_z = (self.online_net.v_max - self.online_net.v_min) / (self.n_atoms - 1)
            b = (Tz - self.online_net.v_min) / delta_z
            l = b.floor().long().clamp(0, self.n_atoms - 1)
            u = b.ceil().long().clamp(0, self.n_atoms - 1)

            # 分配機率質量 (Probability mass)
            m = torch.zeros(self.batch_size, self.n_atoms, device=self.device)
            for i in range(self.n_atoms):
                m.scatter_add_(1, l[:, i:i+1], next_probs[:, i:i+1] * (u[:, i] - b[:, i]).unsqueeze(1))
                m.scatter_add_(1, u[:, i:i+1], next_probs[:, i:i+1] * (b[:, i] - l[:, i]).unsqueeze(1))

        # 線上網路對所採取動作的對數機率 (Log-probs)
        log_probs = torch.log(self.online_net(states) + 1e-8)  # (batch, A, n_atoms)
        log_probs = log_probs[range(self.batch_size), actions]  # (batch, n_atoms)

        # 交叉熵損失 (使用 IS 權重加權)
        loss_per_sample = -(m * log_probs).sum(dim=1)
        loss = (loss_per_sample * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新 PER 優先權
        self.buffer.update_priorities(indices, loss_per_sample.detach().cpu().numpy())

        self.total_steps += 1
        self.buffer.anneal_beta(self.total_steps, total_steps=self.beta_anneal_steps)
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        with torch.no_grad():
            mean_q = self.online_net.get_q_values(states).max(dim=1)[0].mean().item()

        return {
            "loss":   float(loss.item()),
            "mean_q": float(mean_q),
            "beta":   float(self.buffer.beta),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "rainbow.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "rainbow.pt"), map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
