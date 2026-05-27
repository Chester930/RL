"""RL² 網路：GRU 策略 + 價值頭。"""

import torch
import torch.nn as nn


class RL2Network(nn.Module):
    """
    RL² 的循環演員-評論家網路。

    核心設計：GRU 作為「記憶」，讓策略能從同一任務的歷史互動中學習。
    每個時步的輸入為 [prev_action_onehot, prev_reward, prev_done]，
    觀察已隱含在先前動作與獎勵序列中（無狀態的 Bandit 情境）。

    GRU 隱藏狀態在任務內跨集數保留；僅在新任務開始時重置。
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        """
        引數：
            x:      shape (batch, seq_len, input_dim)
            hidden: shape (1, batch, hidden_dim)
        回傳：
            logits: shape (batch, seq_len, action_dim)
            values: shape (batch, seq_len)
            hidden: shape (1, batch, hidden_dim) — 更新後的隱藏狀態
        """
        out, hidden = self.gru(x, hidden)
        logits = self.actor(out)
        values = self.critic(out).squeeze(-1)
        return logits, values, hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """回傳零初始化的 GRU 隱藏狀態。"""
        return torch.zeros(1, batch_size, self.hidden_dim)
