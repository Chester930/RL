"""PPO 的演員-評論家網路。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class ActorCriticNetwork(nn.Module):
    """分離的演員與評論家網路（在 PPO 中通常比共享網路表現略好）。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(64, 64)):
        super().__init__()
        self.actor = nn.Sequential(
            MLP(state_dim, list(hidden_dims), hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], action_dim),
        )
        self.critic = nn.Sequential(
            MLP(state_dim, list(hidden_dims), hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 1),
        )
        # 使用較小的權重初始化輸出層
        for layer in [self.actor[-1], self.critic[-1]]:
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        value = self.critic(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
