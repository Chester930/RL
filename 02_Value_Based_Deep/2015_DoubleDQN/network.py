"""
Double DQN 的 Q 網路 (與 DQN 的架構相同)。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP
from common.networks.cnn import NatureCNN


class QNetwork(nn.Module):
    """標準 Q 網路 — 與 DQN 版本完全相同。"""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), use_cnn=False):
        super().__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.backbone = NatureCNN(n_input_channels=state_dim)
            self.head = nn.Linear(512, action_dim)
        else:
            self.backbone = MLP(state_dim, list(hidden_dims), hidden_dims[-1])
            self.head = nn.Linear(hidden_dims[-1], action_dim)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
