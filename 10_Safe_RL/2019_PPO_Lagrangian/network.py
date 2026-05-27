"""PPO-Lagrangian 三頭網路：高斯策略 + 獎勵評論家 + 代價評論家。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from common.networks.mlp import MLP


class SafeActorCriticNetwork(nn.Module):
    """
    連續動作空間的三頭 Actor-Critic 網路。

    Actor：高斯策略，輸出動作均值；log_std 為可學習的全域參數。
    Reward Critic：估計獎勵值函式 V^r(s)。
    Cost Critic：估計代價值函式 V^c(s)，用於安全約束。

    三個網路各自獨立（不共享主幹），避免梯度衝突。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(64, 64)):
        super().__init__()

        # Actor：主幹 + 均值輸出 + 可學習 log_std
        self.actor_trunk = MLP(state_dim, list(hidden_dims), hidden_dims[-1])
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # 獎勵評論家
        self.reward_critic = nn.Sequential(
            MLP(state_dim, list(hidden_dims), hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 1),
        )

        # 代價評論家
        self.cost_critic = nn.Sequential(
            MLP(state_dim, list(hidden_dims), hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 1),
        )

        # 小權重初始化輸出層（初始策略近似均勻分佈）
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        for net in [self.reward_critic, self.cost_critic]:
            nn.init.orthogonal_(net[-1].weight, gain=1.0)
            nn.init.zeros_(net[-1].bias)

    def get_values(self, x: torch.Tensor):
        """回傳 (reward_value, cost_value)，shape 均為 (batch,)。"""
        rv = self.reward_critic(x).squeeze(-1)
        cv = self.cost_critic(x).squeeze(-1)
        return rv, cv

    def get_mean_action(self, x: torch.Tensor) -> torch.Tensor:
        """評估模式：回傳確定性均值動作（不採樣）。"""
        feat = self.actor_trunk(x)
        return self.actor_mean(feat)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """
        採樣（或評估）動作，同時計算 log_prob、熵、獎勵值、代價值。

        引數：
            x:      狀態張量，shape (batch, state_dim)
            action: 若不為 None，使用給定動作計算 log_prob（update 時使用）

        回傳：
            (action, log_prob, entropy, reward_value, cost_value)
        """
        feat = self.actor_trunk(x)
        mean = self.actor_mean(feat)
        # 限制 log_std 範圍，避免 std 過大或過小
        log_std = self.actor_log_std.clamp(-4.0, 0.5)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        rv, cv = self.get_values(x)
        return action, log_prob, entropy, rv, cv
