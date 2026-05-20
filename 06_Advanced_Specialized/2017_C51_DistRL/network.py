"""
C51 (分類式 DQN / Categorical DQN) 網路 — 分散式強化學習 (Distributional RL)。

不直接預測 Q(s,a) 的期望值，而是將完整的回報分佈 (Return distribution)
預測為在 N 個固定原子 (Atoms) {z_1, ..., z_N} 上的分類分佈。

參考文獻：
    Bellemare, M. G., Dabney, W., & Munos, R. (2017).
    A Distributional Perspective on Reinforcement Learning.
    ICML 2017. arXiv:1707.06887.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.networks.mlp import MLP


class CategoricalQNetwork(nn.Module):
    """
    C51 分散式 Q 網路。

    輸出：在 N 個原子上的機率分佈 p(s, a)
    Q(s, a) = sum_i z_i * p_i(s, a)

    引數：
        state_dim:   輸入狀態維度。
        action_dim:  離散動作數量。
        n_atoms:     分佈支撐集 (Support) 原子數量（預設為 51）。
        v_min:       回報最小值。
        v_max:       回報最大值。
        hidden_dims: 隱藏層維度。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dims=(256, 256),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # 原子支撐集數值（固定）(Atom support values)
        self.register_buffer(
            "atoms",
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # 共享特徵提取器 (Shared feature extractor)
        self.backbone = MLP(state_dim, list(hidden_dims), hidden_dims[-1])
        # 輸出：每個（動作, 原子）對的 Logits
        self.head = nn.Linear(hidden_dims[-1], action_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        回傳：
            probs: (batch, action_dim, n_atoms) — Softmax 機率分佈。
        """
        features = self.backbone(x)
        logits = self.head(features)                      # (B, A*N)
        logits = logits.view(-1, self.action_dim, self.n_atoms)  # (B, A, N)
        return F.softmax(logits, dim=-1)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        期望 Q 值：Q(s,a) = sum_i z_i * p_i(s,a)。

        回傳：
            q_values: (batch, action_dim)
        """
        probs = self.forward(x)                           # (B, A, N)
        return (probs * self.atoms.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, A)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        用於交叉熵損失的對數機率 (Log-probabilities)。

        回傳：
            log_p: (batch, action_dim, n_atoms)
        """
        features = self.backbone(x)
        logits = self.head(features).view(-1, self.action_dim, self.n_atoms)
        return F.log_softmax(logits, dim=-1)
