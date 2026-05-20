"""TD3 的演員與雙評論家網路。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(MLP(state_dim, list(hidden_dims), action_dim), nn.Tanh())

    def forward(self, s):
        return self.net(s) * self.action_scale


class TwinCriticNetwork(nn.Module):
    """兩個獨立的 Q 網路，用於減少高估偏差 (Clipped Double Q)。"""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, list(hidden_dims), 1)
        self.q2 = MLP(state_dim + action_dim, list(hidden_dims), 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1))
