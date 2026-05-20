"""
IQL 網路元件：用於離線強化學習 (Offline RL) 的 V 網路、Q 網路與演員網路。

參考文獻：
    Kostrikov, I., Nair, A., & Levine, S. (2021).
    Offline Reinforcement Learning with Implicit Q-Learning.
    ICLR 2022. arXiv:2110.06169.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    狀態價值函式 V(s)。

    IQL 使用獨立的 V 網路將策略提取與 Q 學習更新解耦，
    從而避免在計算貝爾曼目標時需要查詢 OOD 動作。

    引數：
        state_dim:  狀態維度。
        hidden_dim: MLP 隱藏層維度。
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class TwinQNetwork(nn.Module):
    """
    雙 Q 網路 Q(s, a)。

    IQL 透過使用 V(s') 代替 max_a' Q(s', a') 的貝爾曼備份來訓練 Q，
    藉此完全避免了對分佈外 (OOD) 動作的查詢。

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


class GaussianActor(nn.Module):
    """
    用於 IQL 的高斯策略網路。

    訓練目標為極大化優勢加權對數概似度 (Advantage-weighted log-likelihood)：
        L_actor = E[exp(tau * A(s,a)) * log pi(a|s)]

    其中 A(s,a) = Q(s,a) - V(s) 為優勢函式。
    指數權重會賦予具備正優勢的動作更高的學習權重。

    引數：
        state_dim:  狀態維度。
        action_dim: 動作維度。
        hidden_dim: MLP 隱藏層維度。
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

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        計算動作在目前策略下的對數機率 (Log probability)。

        為了簡化，假設動作處於原始空間（尚未經過 tanh 截斷）。
        """
        mu, log_std = self.forward(state)
        dist = torch.distributions.Normal(mu, log_std.exp())
        return dist.log_prob(action).sum(dim=-1)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(state)
        return torch.tanh(mu)
