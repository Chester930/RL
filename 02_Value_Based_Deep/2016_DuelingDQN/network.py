"""
Dueling Q 網路 (Dueling Q-Network)。

將 Q(s,a) 分解為 V(s) + A(s,a)，並減去平均優勢。

參考文獻：
    Wang, Z., et al. (2016). Dueling Network Architectures for Deep
    Reinforcement Learning. ICML 2016.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP
from common.networks.cnn import NatureCNN


class DuelingQNetwork(nn.Module):
    """
    具有獨立價值流與優勢流的 Dueling 網路。

    Q(s, a) = V(s) + A(s, a) - mean_{a'} A(s, a')

    減去平均優勢可使分解過程具備唯一性與穩定性。
    網路可以精確地估計 V(s)，而不需要單獨評估每個動作。

    架構：
        共享骨幹網路 -> 分割為：
            價值流 (Value stream)：     全連線層 -> V(s)   [純量]
            優勢流 (Advantage stream)： 全連線層 -> A(s,a) [動作向量]
        合併：Q = V + A - mean(A)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
        use_cnn: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim

        # 共享骨幹網路 (Shared backbone)
        if use_cnn:
            self.backbone = NatureCNN(n_input_channels=state_dim, features_dim=512)
            feature_dim = 512
        else:
            # 共享層 (除了最後一個隱藏層外的所有層)
            shared_dims = list(hidden_dims[:-1]) if len(hidden_dims) > 1 else [128]
            self.backbone = MLP(state_dim, shared_dims, shared_dims[-1])
            feature_dim = shared_dims[-1]

        stream_dim = hidden_dims[-1] if hidden_dims else 128

        # 價值流 (Value stream)：V(s) -> 純量 (scalar)
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, stream_dim),
            nn.ReLU(),
            nn.Linear(stream_dim, 1),
        )

        # 優勢流 (Advantage stream)：A(s, a) -> 向量 (vector)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, stream_dim),
            nn.ReLU(),
            nn.Linear(stream_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            q_values: shape (batch, action_dim)
        """
        features = self.backbone(x)

        # TODO: V(s) + A(s,a) - mean_a A(s,a)
        V = self.value_stream(features)               # (batch, 1)
        A = self.advantage_stream(features)           # (batch, n_actions)

        # 減去平均優勢以確保可識別性 (Identifiability)
        Q = V + A - A.mean(dim=1, keepdim=True)       # (batch, n_actions)
        return Q
