"""
DQN 的 Q 網路 (Q-Network)。

可使用 MLP (用於低維度狀態空間) 或 CNN (用於畫素級觀察值)。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP
from common.networks.cnn import NatureCNN


class QNetwork(nn.Module):
    """
    將狀態對映至所有動作的 Q(s, a)。

    引數：
        state_dim:  針對 MLP：輸入特徵的維度。
                    針對 CNN：堆疊影格的數量 (預設為 4)。
        action_dim: 離散動作的數量。
        hidden_dims: 隱藏層大小 (僅限 MLP)。
        use_cnn:    若為 True，則使用 NatureCNN 骨幹網路 (用於 Atari)。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
        use_cnn: bool = False,
    ):
        super().__init__()
        self.use_cnn = use_cnn

        if use_cnn:
            self.backbone = NatureCNN(n_input_channels=state_dim, features_dim=512)
            self.head = nn.Linear(512, action_dim)
        else:
            self.backbone = MLP(state_dim, list(hidden_dims), hidden_dims[-1])
            self.head = nn.Linear(hidden_dims[-1], action_dim)

        # 使用較小的權重初始化輸出層 (強化學習中的常見做法)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引數：
            x: 狀態張量，形狀為 (batch, state_dim) 或 (batch, C, H, W)。

        回傳：
            q_values: 形狀為 (batch, action_dim)。
        """
        features = self.backbone(x)
        return self.head(features)
