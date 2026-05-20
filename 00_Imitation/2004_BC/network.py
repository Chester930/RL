"""BC policy network: state → bounded continuous action."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class BCNetwork(nn.Module):
    """
    Behavioral Cloning 的確定性策略網路。

    架構：state → MLP → tanh → 縮放至動作範圍

    與 SAC PolicyNetwork 的差異：
        SAC: 輸出高斯分佈的均值與標準差（隨機性策略，支援熵正則化）
        BC:  直接輸出確定性動作（直接模仿專家的平均行為）

    tanh 確保輸出落在 [-1, 1]，再乘以 action_scale 對應環境的動作範圍。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple = (256, 256),
        action_scale: float = 2.0,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.net = MLP(state_dim, list(hidden_dims), action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """回傳縮放後的確定性動作，範圍在 [-action_scale, action_scale]。"""
        return torch.tanh(self.net(state)) * self.action_scale
