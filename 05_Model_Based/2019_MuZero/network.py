"""
MuZero 網路元件：表示網路 (Representation)、動態網路 (Dynamics) 與預測網路 (Prediction)。

參考文獻：
    Schrittwieser, J., et al. (2019). Mastering Atari, Go, Chess and Shogi by
    Planning with a Learned Model. Nature 588, 604-609. arXiv:1911.08265.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.networks.mlp import MLP


class RepresentationNetwork(nn.Module):
    """
    h(o_1..t) -> s_t
    將觀測影像（歷史記錄）對映至初始隱藏狀態 s_0。
    棋盤遊戲：將棋盤狀態對映至潛在嵌入向量 (Latent embedding)。
    Atari 影像：將堆疊的影像幀對映至潛在嵌入向量。
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(obs_dim, [256, 256], hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """回傳潛在狀態 s。"""
        s = self.net(obs)
        # 正規化至 [-1, 1] 範圍（MuZero 的慣例）
        return s / (s.abs().max(dim=-1, keepdim=True)[0] + 1e-5)


class DynamicsNetwork(nn.Module):
    """
    g(s_t, a_t) -> (s_{t+1}, r_t)
    給定目前狀態與動作，預測下一個隱藏狀態與即時獎勵。
    這是學習到的動態模型 — 規劃時無需真實環境參與。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 support_size: int = 601):
        super().__init__()
        # 狀態轉移：(狀態 + one-hot 動作) -> 下一個狀態
        self.state_net = MLP(state_dim + action_dim, [256, 256], hidden_dim)

        # 獎勵預測：將純量獎勵表示為支撐集 (Support) 上的分佈
        # 支撐集：[-300, 300]，共 601 個原子 (Atoms)（分散式 MuZero）
        self.reward_net = nn.Linear(hidden_dim, support_size)
        self.support_size = support_size

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        引數：
            state:  (batch, state_dim)
            action: (batch, action_dim)  one-hot 編碼
        回傳：
            next_state: (batch, state_dim)
            reward_logits: (batch, support_size)
        """
        sa = torch.cat([state, action], dim=-1)
        next_state = self.state_net(sa)
        next_state = next_state / (next_state.abs().max(dim=-1, keepdim=True)[0] + 1e-5)
        reward_logits = self.reward_net(next_state)
        return next_state, reward_logits


class PredictionNetwork(nn.Module):
    """
    f(s_t) -> (pi_t, v_t)
    從隱藏狀態預測策略與價值。
    同時用於 MCTS 的選擇（策略先驗）與回傳（價值評估）。
    """

    def __init__(self, state_dim: int, action_dim: int, support_size: int = 601):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        # 價值表示為支撐集上的分佈（分散式 MuZero）
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, support_size),
        )
        self.support_size = support_size

    def forward(self, state: torch.Tensor) -> tuple:
        """
        回傳：
            policy_logits: (batch, action_dim) — MCTS 先驗
            value_logits:  (batch, support_size) — 價值分佈
        """
        policy_logits = self.policy_head(state)
        value_logits = self.value_head(state)
        return policy_logits, value_logits

    @staticmethod
    def support_to_scalar(logits: torch.Tensor, support_size: int = 601) -> torch.Tensor:
        """將分散式輸出轉換為純量值。"""
        probs = F.softmax(logits, dim=-1)
        support = torch.linspace(-300, 300, support_size, device=logits.device)
        return (probs * support).sum(dim=-1)
