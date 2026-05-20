"""
MAPPO 網路元件：各代理人的演員網路 (Actor) 與集中式評論家網路 (Critic)。

參考文獻：
    Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., & Wu, Y. (2021).
    The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
    arXiv:2103.01955.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentActor(nn.Module):
    """
    用於分散式執行的各代理人演員網路。

    支援輸出分類分佈 (Categorical, 離散動作) 或高斯分佈 (Gaussian, 連續動作) 策略。

    引數：
        obs_dim:    該代理人的區域性觀測維度。
        action_dim: 離散動作數量（或連續動作維度）。
        hidden_dim: MLP 隱藏層維度。
        continuous: 若為 True 則使用高斯策略，False 則使用分類策略。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        if continuous:
            # 用於連續策略的可學習對數標準差 (Learnable log_std)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        if self.continuous:
            mu = self.policy_head(h)
            std = self.log_std.exp().expand_as(mu)
            return torch.distributions.Normal(mu, std)
        else:
            logits = self.policy_head(h)
            return torch.distributions.Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.forward(obs)
        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob


class CentralizedCritic(nn.Module):
    """
    用於 MAPPO 訓練的集中式價值函式 (Centralized value function)。

    在 MAPPO 中，評論家接收的是全域性狀態（而非僅是區域性觀測），
    這使得它比分散式評論家能提供更精準的價值估計。

    兩種變體：
        MAPPO (全域性)：評論家觀察全域性環境狀態。
        IPPO  (區域性)：評論家僅觀察該代理人的區域性觀測 (Independent PPO)。

    引數：
        state_dim:  全域性狀態維度（若為 IPPO 則為代理人區域性觀測維度）。
        hidden_dim: MLP 隱藏層維度。
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
