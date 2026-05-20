"""
MADDPG 網路元件：每個代理人的演員 (Actor) 與集中式評論家 (Centralized Critic)。

參考文獻：
    Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017).
    Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
    NeurIPS 2017. arXiv:1706.02275.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn


class AgentActor(nn.Module):
    """
    代理人演員網路：分散式執行 (Decentralized execution)。

    僅接收該代理人的區域性觀測 (Local observation)；輸出動作。

    引數：
        obs_dim:    該代理人的觀測維度。
        action_dim: 該代理人的動作維度。
        hidden_dim: MLP 隱藏層維度。
        continuous: 若為 True，透過 tanh 輸出連續動作。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        continuous: bool = True,
    ):
        super().__init__()
        self.continuous = continuous
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        if self.continuous:
            return torch.tanh(out)
        return out  # 離散動作的 Logits (在 MADDPG 中通常配合 Gumbel-softmax 使用)


class CentralizedCritic(nn.Module):
    """
    集中式評論家網路：使用完整資訊進行訓練。

    接收所有代理人的觀測與動作；輸出該代理人的 Q 值。
    這是 MADDPG 的關鍵創新 — 評論家在訓練期間可以看到「全域性」狀態與所有人的行為。

    引數：
        total_obs_dim:    所有代理人觀測維度的總和。
        total_action_dim: 所有代理人動作維度的總和。
        hidden_dim:       MLP 隱藏層維度。
    """

    def __init__(
        self,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """
        引數：
            all_obs:     (batch, total_obs_dim)   — 所有代理人的觀測連線而成的張量。
            all_actions: (batch, total_action_dim) — 所有代理人的動作連線而成的張量。
        回傳：
            q_value: (batch,)
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x).squeeze(-1)
