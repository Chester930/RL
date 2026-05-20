"""
A3C 的演員-評論家網路 (Actor-Critic Network)。

共享骨幹網路，並設有獨立的演員 (策略) 與評論家 (價值) 輸出層。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.networks.mlp import MLP


class ActorCriticNetwork(nn.Module):
    """
    共享骨幹的演員-評論家網路。

    架構：
        共享 MLP -> 分割為：
            演員輸出層 (Actor head)： 全連線層 -> logits -> softmax (離散策略)
            評論家輸出層 (Critic head)：全連線層 -> V(s) 純量

    引數：
        state_dim:   輸入狀態維度。
        action_dim:  離散動作的數量。
        hidden_dims: 共享隱藏層的大小。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()

        # 共享特徵提取器 (Shared feature extractor)
        self.shared = MLP(state_dim, list(hidden_dims), hidden_dims[-1])

        # 演員輸出層：輸出動作的 logits (Actor head)
        self.actor_head = nn.Linear(hidden_dims[-1], action_dim)

        # 評論家輸出層：輸出狀態價值 V(s) 純量 (Critic head)
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # 演員輸出層使用較小的初始權重（穩定訓練初期）
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, x: torch.Tensor):
        """
        引數：
            x: 狀態張量 (batch, state_dim)

        回傳：
            logits: (batch, action_dim) — 原始動作 logits
            value:  (batch, 1)          — V(s) 估計值
        """
        features = self.shared(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action(self, x: torch.Tensor):
        """
        從策略分佈中取樣動作。

        回傳：
            action:     取樣到的動作 (純量)
            log_prob:   該動作的對數機率 (log probability)
            entropy:    策略分佈的熵 (Entropy)
            value:      V(s) 估計值
        """
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """
        評估給定「狀態-動作」對的對數機率與熵。
        用於更新步驟中。
        """
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value
