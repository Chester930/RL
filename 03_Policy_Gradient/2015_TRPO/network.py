"""TRPO 的演員-評論家 (Actor-Critic) 網路。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64)):
        super().__init__()
        self.actor = MLP(state_dim, list(hidden_dims), action_dim)
        self.critic = MLP(state_dim, list(hidden_dims), 1)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_policy(self, x):
        return torch.distributions.Categorical(logits=self.actor(x))

    def get_value(self, x):
        return self.critic(x)
