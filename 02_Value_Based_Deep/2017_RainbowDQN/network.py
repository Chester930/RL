"""
Rainbow 網路 — 整合了 Dueling + 分散式 (C51) + 雜訊網路 (NoisyNet)。

參考文獻：
    Hessel, M., et al. (2017). Rainbow: Combining Improvements in Deep
    Reinforcement Learning. AAAI 2018. arXiv:1710.02298.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.networks.mlp import NoisyLinear
from common.networks.cnn import NatureCNN


class RainbowNetwork(nn.Module):
    """
    Rainbow 整合了 6 項改進：
    1. Double DQN (解耦目標計算)
    2. 優先經驗回放 (PER, 見 priority_buffer.py)
    3. Dueling 網路 (V + A 分解)
    4. N-步回報 (多步 TD)
    5. 分散式 RL / C51 (此網路輸出原子機率)
    6. 雜訊網路 (NoisyNet, 使用 NoisyLinear 層取代 epsilon-greedy)

    此網路實作了第 3 項 (Dueling)、第 5 項 (C51 原子) 與第 6 項 (NoisyNet)。

    輸出：每個動作在 n_atoms 上的 logits (經 softmax 轉換為機率分佈)

    引數：
        state_dim:  輸入狀態維度 (或 CNN 的通道數)。
        action_dim: 離散動作的數量。
        n_atoms:    C51 分佈的原子數量 (預設為 51)。
        v_min:      支撐集 (Support) 的最小值。
        v_max:      支撐集 (Support) 的最大值。
        use_cnn:    使用 NatureCNN 骨幹網路。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        use_cnn: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # 支撐集 (固定的原子位置)
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, n_atoms),
        )

        # 共享骨幹網路 (Shared backbone)
        if use_cnn:
            self.backbone = NatureCNN(n_input_channels=state_dim, features_dim=512)
            feature_dim = 512
        else:
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
            )
            feature_dim = 256

        # Dueling 輸出流 — 使用 NoisyLinear 進行探索
        # 價值流 (Value stream)：特徵 -> NoisyLinear(256) -> NoisyLinear(n_atoms)
        self.value_hidden = NoisyLinear(feature_dim, 256)
        self.value_out = NoisyLinear(256, n_atoms)

        # 優勢流 (Advantage stream)：特徵 -> NoisyLinear(256) -> NoisyLinear(action_dim * n_atoms)
        self.advantage_hidden = NoisyLinear(feature_dim, 256)
        self.advantage_out = NoisyLinear(256, action_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        回傳：
            probs: (batch, action_dim, n_atoms) — 原子上的 softmax 機率
        """
        features = self.backbone(x)

        # Value stream
        V = F.relu(self.value_hidden(features))
        V = self.value_out(V).view(-1, 1, self.n_atoms)          # (batch, 1, n_atoms)

        # Advantage stream
        A = F.relu(self.advantage_hidden(features))
        A = self.advantage_out(A).view(-1, self.action_dim, self.n_atoms)  # (batch, A, n_atoms)

        # TODO: Dueling 結合 (在原子空間中，softmax 之前)
        #   Q_logits = V + A - mean(A)
        Q_logits = V + A - A.mean(dim=1, keepdim=True)            # (batch, A, n_atoms)

        # 在原子上進行 softmax 以獲得每個動作的機率分佈
        return F.softmax(Q_logits, dim=-1)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算純量 Q 值（取期望值）：Q(s,a) = sum_z p(z|s,a) * z

        回傳：
            q_values: (batch, action_dim)
        """
        probs = self.forward(x)                                    # (batch, A, n_atoms)
        return (probs * self.support.view(1, 1, -1)).sum(dim=-1)  # (batch, A)

    def sample_noise(self) -> None:
        """重新取樣 NoisyLinear 的雜訊（訓練時每次前向傳遞呼叫一次）。"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.sample_noise()
