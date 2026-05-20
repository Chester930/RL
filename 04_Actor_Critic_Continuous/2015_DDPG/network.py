"""DDPG 的演員與評論家網路。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class ActorNetwork(nn.Module):
    """
    確定性策略：將狀態對映至 [-1, 1] 區間內的動作值。
    使用 tanh 輸出層以確保動作在合理範圍內。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(400, 300),
                 action_scale: float = 1.0):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            MLP(state_dim, list(hidden_dims), action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.action_scale


class CriticNetwork(nn.Module):
    """
    Q(s, a)：將 (狀態, 動作) 對映至標量 Q 值。
    動作值通常與狀態串聯後作為輸入。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(400, 300)):
        super().__init__()
        # 輸入 = 狀態 + 動作
        self.net = MLP(state_dim + action_dim, list(hidden_dims), 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
