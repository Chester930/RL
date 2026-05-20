"""REINFORCE 的策略網路。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class PolicyNetwork(nn.Module):
    """適用於離散動作空間的隨機策略 pi(a|s)。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        self.net = MLP(state_dim, list(hidden_dims), action_dim)

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        """回傳動作上的類別分佈 (Categorical distribution)。"""
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, x: torch.Tensor):
        dist = self.forward(x)
        action = dist.sample()
        return action, dist.log_prob(action)
