"""
MBPO 網路元件：整合動態模型 (Ensemble Dynamics Model)、策略網路與 Q 網路。

參考文獻：
    Janner, M., Fu, J., Zhang, M., & Levine, S. (2019).
    When to Trust Your Model: Model-Based Policy Optimization.
    NeurIPS 2019. arXiv:1906.08253.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.networks.mlp import MLP


class EnsembleDynamicsModel(nn.Module):
    """
    機率式整合動態模型 (Probabilistic ensemble of dynamics models)。

    每個成員皆預測 (下一個狀態 - 目前狀態, 獎勵) 的高斯分佈：
        mu, log_var = f_i(state, action)

    整合模型提供了認知不確定性 (Epistemic uncertainty) 的估計，用於決定何時信賴該模型（從真實轉換資料進行分支取樣時）。

    引數：
        state_dim:   狀態維度。
        action_dim:  動作維度。
        hidden_dim:  每個整合成員的隱藏層維度。
        n_members:   整合成員的數量（預設為 7，並使用其中 5 個精銳成員）。
        max_log_var: 預測對數變異數 (Log-variance) 的上限。
        min_log_var: 預測對數變異數的下限。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 200,
        n_members: int = 7,
        max_log_var: float = 0.5,
        min_log_var: float = -10.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_members = n_members

        # 每個整合成員皆為獨立的 MLP (Separate MLP)
        in_dim = state_dim + action_dim
        out_dim = (state_dim + 1) * 2  # (delta_s, r) 的均值 mu 與對數變異數 log_var

        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            for _ in range(n_members)
        ])

        # 可學習的對數變異數邊界 (Learnable bounds)
        self.max_log_var = nn.Parameter(torch.ones(1, state_dim + 1) * max_log_var)
        self.min_log_var = nn.Parameter(torch.ones(1, state_dim + 1) * min_log_var)

    def forward(self, state: torch.Tensor, action: torch.Tensor, member_idx: int = 0):
        """
        單一整合成員的前向傳播。

        引數：
            state:      (batch, state_dim)
            action:     (batch, action_dim)
            member_idx: 指定使用哪一個整合成員。
        回傳：
            mu:      (batch, state_dim+1)  [狀態變化 delta_state | 獎勵 reward]
            log_var: (batch, state_dim+1)
        """
        x = torch.cat([state, action], dim=-1)
        out = self.members[member_idx](x)

        half = out.shape[-1] // 2
        mu = out[..., :half]
        raw_log_var = out[..., half:]

        # 在可學習邊界內對對數變異數進行軟截斷 (Soft clamp)
        log_var = self.max_log_var - F.softplus(self.max_log_var - raw_log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)

        return mu, log_var

    def predict_all(self, state: torch.Tensor, action: torch.Tensor):
        """執行所有整合成員；回傳堆疊後的結果 (n_members, batch, out)。"""
        mus, log_vars = [], []
        for i in range(self.n_members):
            mu, lv = self.forward(state, action, i)
            mus.append(mu)
            log_vars.append(lv)
        return torch.stack(mus), torch.stack(log_vars)

    def nll_loss(self, state: torch.Tensor, action: torch.Tensor,
                 delta_state: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        在所有整合成員上取平均的 Gaussian NLL 損失函式。

        目標值 target = [狀態變化 delta_state | 獎勵 reward]，形狀為：(batch, state_dim+1)
        """
        target = torch.cat([delta_state, reward.unsqueeze(-1)], dim=-1)
        total_loss = torch.tensor(0.0, device=state.device)

        for i in range(self.n_members):
            mu, log_var = self.forward(state, action, i)
            inv_var = (-log_var).exp()
            mse = (mu - target).pow(2)
            nll = (mse * inv_var + log_var).sum(dim=-1).mean()
            total_loss = total_loss + nll

        # 針對對數變異數邊界的正規化 (Regularization)
        reg = self.max_log_var.sum() - self.min_log_var.sum()

        return total_loss / self.n_members + 0.01 * reg

    @torch.no_grad()
    def sample_next_state(self, state: torch.Tensor, action: torch.Tensor,
                          elite_indices: list) -> tuple:
        """
        從隨機選擇的精銳成員 (Elite member) 中取樣下一個狀態與獎勵。

        回傳：
            next_state: (batch, state_dim)
            reward:     (batch,)
        """
        idx = elite_indices[np.random.randint(len(elite_indices))]
        mu, log_var = self.forward(state, action, idx)
        std = (0.5 * log_var).exp()
        sample = mu + std * torch.randn_like(std)

        delta_state = sample[..., :self.state_dim]
        reward = sample[..., -1]
        next_state = state + delta_state
        return next_state, reward


class PolicyNetwork(nn.Module):
    """
    隨機高斯策略 (Stochastic Gaussian policy)，與 SAC 共享。

    輸出：均值與對數標準差 -> 透過 tanh 將重引數化後的動作對映至 (-1, 1)。
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        """
        進行重引數化取樣並計算對數機率（包含 tanh 的 Jacobian 修正）。

        回傳：
            action:   (batch, action_dim)，範圍在 (-1, 1)
            log_prob: (batch,)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(std)
        raw = mu + std * eps

        action = torch.tanh(raw)
        log_prob = (
            torch.distributions.Normal(mu, std).log_prob(raw)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1)

        return action, log_prob


class TwinQNetwork(nn.Module):
    """用於 SAC 式評論家的雙 Q 網路（減少高估現象）。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
