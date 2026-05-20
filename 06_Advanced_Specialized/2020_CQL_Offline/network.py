"""
CQL 網路元件：用於離線強化學習 (Offline RL) 的 Q 網路與策略網路。

參考文獻：
    Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020).
    Conservative Q-Learning for Offline Reinforcement Learning.
    NeurIPS 2020. arXiv:2006.04779.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwinQNetwork(nn.Module):
    """
    雙 Q 網路（SAC 風格），用以減少目標價值的過度高估。

    在 CQL 中，兩個 Q 網路頭都會受到保守正規化項 (Conservative regularizer) 的懲罰。

    引數：
        state_dim:  狀態維度。
        action_dim: 動作維度。
        hidden_dim: MLP 隱藏層維度。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class GaussianPolicy(nn.Module):
    """
    用於 CQL 的 SAC 風格隨機高斯策略（適用於連續動作空間）。

    輸出：透過 tanh 函式歸一化到 (-1, 1) 的重引數化動作。
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        """執行重引數化取樣並計算對數機率 (Log-prob)，包含 tanh Jacobian 修正。"""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(std)
        raw = mu + std * eps
        action = torch.tanh(raw)
        log_prob = (
            torch.distributions.Normal(mu, std).log_prob(raw)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1)
        return action, log_prob

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """評估時使用的確定性動作（取高斯分佈均值）。"""
        mu, _ = self.forward(state)
        return torch.tanh(mu)
