"""
REINFORCE 代理人 — 蒙特卡羅策略梯度 (Monte Carlo Policy Gradient)。

參考文獻：
    Williams, R. J. (1992). Simple statistical gradient-following algorithms
    for connectionist reinforcement learning. Machine Learning, 8(3–4), 229–256.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from common.base_agent import BaseAgent
from network import PolicyNetwork


class REINFORCEAgent(BaseAgent):
    """
    具備可選基準值 (Baseline) 的 REINFORCE（使用價值函式基準來縮減方差）。

    REINFORCE 更新規則：
        theta += alpha * G_t * grad log pi(a_t | s_t; theta)

    加入基準值 b(s_t)（在不引入偏差的情況下縮減方差）：
        theta += alpha * (G_t - b(s_t)) * grad log pi(a_t | s_t; theta)

    梯度僅在每個集數結束時進行計算（蒙特卡羅方法）。
    這是最簡單的策略梯度演演算法，但缺點是方差較高。

    引數：
        state_dim:   輸入狀態維度。
        action_dim:  離散動作的數量。
        lr:          學習率。
        gamma:       折扣因子。
        use_baseline: 若為 True，則減去平均回報作為縮減方差的基準值。
        normalize_returns: 若為 True，則將回報歸一化為零均值與單位方差。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True,
        normalize_returns: bool = True,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.normalize_returns = normalize_returns

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 集數儲存區 (Episode storage)
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        從策略分佈中取樣。

        在評估期間，使用貪婪動作（logits 的最大值）。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                logits = self.policy.net(state_t)
            return int(logits.argmax(dim=1).item())

        dist = self.policy.forward(state_t)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return int(action.item())

    def store_reward(self, reward: float) -> None:
        """儲存當前時步的獎勵。"""
        self._rewards.append(reward)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        計算 REINFORCE 梯度並更新策略。

        在每個完整集數結束時呼叫一次。

        步驟：
        1. 反向計算整個集數的折扣回報 G_t
        2. (選用) 回報歸一化
        3. 計算損失函式：L = -sum_t G_t * log pi(a_t | s_t)
        4. 執行梯度更新步驟
        """
        if not self._rewards:
            return {}

        T = len(self._rewards)

        # --- 步驟 1：計算折扣回報 (Discounted returns) ---
        # TODO: G_t = r_t + gamma * G_{t+1} (反向累積計算)
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G

        # --- 步驟 2：回報歸一化 (方差縮減) ---
        if self.normalize_returns and T > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # --- 步驟 3：減去基準值 (Baseline subtraction) ---
        if self.use_baseline:
            baseline = returns.mean()
            advantages = returns - baseline
        else:
            advantages = returns

        # --- 步驟 4：策略梯度損失 (Policy gradient loss) ---
        # TODO: L = -sum_t advantage_t * log_prob_t
        log_probs = torch.stack(self._log_probs)   # (T,)
        loss = -(log_probs * advantages).sum()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += T
        self.episodes_done += 1

        # 清除集數緩衝區 (Episode buffers)
        self._log_probs.clear()
        self._rewards.clear()

        return {"loss": float(loss.item()), "mean_return": float(returns.mean().item())}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, os.path.join(path, "reinforce.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "reinforce.pt"), map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
