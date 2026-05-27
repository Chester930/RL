"""CPO 網路：與 PPO-Lagrangian 相同的三頭架構（獨立副本）。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class SafeActorCriticNetwork(nn.Module):
    """
    高斯策略 (Actor) + 獎勵評論家 + 代價評論家（三頭獨立網路）。

    CPO 和 PPO-Lagrangian 使用相同的網路架構，
    差異在於更新規則（自然梯度 vs. 梯度剪裁）。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(64, 64)):
        super().__init__()
        self.actor_trunk = MLP(state_dim, list(hidden_dims), hidden_dims[-1])
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.reward_critic = nn.Sequential(
            MLP(state_dim, list(hidden_dims), hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 1),
        )
        self.cost_critic = nn.Sequential(
            MLP(state_dim, list(hidden_dims), hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 1),
        )

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        for net in [self.reward_critic, self.cost_critic]:
            nn.init.orthogonal_(net[-1].weight, gain=1.0)
            nn.init.zeros_(net[-1].bias)

    def get_values(self, x):
        return self.reward_critic(x).squeeze(-1), self.cost_critic(x).squeeze(-1)

    def get_mean_action(self, x):
        return self.actor_mean(self.actor_trunk(x))

    def get_dist(self, x):
        feat = self.actor_trunk(x)
        mean = self.actor_mean(feat)
        std = self.actor_log_std.clamp(-4.0, 0.5).exp()
        return torch.distributions.Normal(mean, std)

    def get_action_and_value(self, x, action=None):
        dist = self.get_dist(x)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        rv, cv = self.get_values(x)
        return action, log_prob, entropy, rv, cv
