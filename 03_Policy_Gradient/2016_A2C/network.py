"""A2C 的演員-評論家網路 (架構與 A3C 相同)。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class ActorCriticNetwork(nn.Module):
    """共享骨幹的演員-評論家：共享 MLP -> 演員輸出層 + 評論家輸出層。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.shared = MLP(state_dim, list(hidden_dims), hidden_dims[-1])
        self.actor_head = nn.Linear(hidden_dims[-1], action_dim)
        self.critic_head = nn.Linear(hidden_dims[-1], 1)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        return self.actor_head(features), self.critic_head(features)

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
