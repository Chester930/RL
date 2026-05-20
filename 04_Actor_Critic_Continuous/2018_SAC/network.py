"""SAC 的策略與 Q 網路。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class PolicyNetwork(nn.Module):
    """
    高斯策略：輸出均值 (mean) 與對數標準差 (log_std)。
    動作透過重引數化技巧 (Reparameterization trick) 進行取樣：
        a = tanh(mu + sigma * epsilon)   其中 epsilon ~ N(0, I)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256),
                 action_scale: float = 1.0):
        super().__init__()
        self.action_scale = action_scale
        self.net = MLP(state_dim, list(hidden_dims), hidden_dims[-1])
        self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state: torch.Tensor):
        features = self.net(state)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_action(self, state: torch.Tensor):
        """
        使用重引數化技巧與 tanh 壓縮功能進行動作取樣。

        回傳：
            action:   取樣並壓縮後的動作，範圍在 [-action_scale, action_scale]
            log_prob: 包含壓縮修正後的對數機率值
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        # Reparameterization: z = mu + std * epsilon
        z = dist.rsample()

        # Squash with tanh and scale
        action = torch.tanh(z) * self.action_scale

        # TODO: 包含 tanh 壓縮 Jacobian 修正的對數機率計算
        # log pi(a|s) = log N(z|mu,sigma) - sum log(1 - tanh^2(z))
        log_prob = dist.log_prob(z).sum(dim=-1)
        # 修正項：- sum log(1 - tanh(z)^2)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - z - nn.functional.softplus(-2 * z))).sum(dim=-1)

        return action, log_prob

    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """用於評估：回傳 tanh(均值) 作為確定性動作。"""
        mu, _ = self.forward(state)
        return torch.tanh(mu) * self.action_scale


class TwinQNetwork(nn.Module):
    """雙 Q 網路 Q1 與 Q2，用於減少高估問題（與 TD3 類似）。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, list(hidden_dims), 1)
        self.q2 = MLP(state_dim + action_dim, list(hidden_dims), 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
