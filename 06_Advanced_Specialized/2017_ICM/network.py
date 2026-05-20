"""
ICM 網路元件：特徵編碼器 (Feature encoder)、逆向模型 (Inverse model)、前向模型 (Forward model)。

參考文獻：
    Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
    Curiosity-driven Exploration by Self-Supervised Prediction.
    ICML 2017. arXiv:1705.05363.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """
    將原始狀態對映為精簡的特徵表示 phi(s)。

    透過逆向模型訓練，使特徵空間對於環境中無關的因素（如背景、光影）
    具備不變性 (Invariant)，僅保留受動作影響的部分。

    引數：
        state_dim:   輸入狀態維度（畫素輸入則使用 CNN）。
        feature_dim: 輸出特徵維度。
    """

    def __init__(self, state_dim: int, feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ELU(),
            nn.Linear(256, feature_dim), nn.ELU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class InverseModel(nn.Module):
    """
    根據 (phi(s_t), phi(s_{t+1})) 預測所採取的動作。

    強迫特徵編碼器僅捕捉環境中「可控 (Controllable)」的部分。

    引數：
        feature_dim: 編碼器的特徵維度。
        action_dim:  離散動作數量（用於分類）或動作維度（用於回歸）。
        discrete:    若為 True，預測離散動作的 Logits。
    """

    def __init__(self, feature_dim: int, action_dim: int, discrete: bool = True):
        super().__init__()
        self.discrete = discrete
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, phi_s: torch.Tensor, phi_s_next: torch.Tensor) -> torch.Tensor:
        """
        回傳：
            離散動作：動作 Logits (batch, action_dim)。
            連續動作：預測的動作 (batch, action_dim)。
        """
        x = torch.cat([phi_s, phi_s_next], dim=-1)
        return self.net(x)


class ForwardModel(nn.Module):
    """
    根據 (phi(s_t), a_t) 預測下一個狀態的特徵表示。

    預測誤差 (Prediction error) = 內在好奇心獎勵。
    高預測誤差 -> 新奇或出乎意料的狀態 -> 激發更多探索。

    引數：
        feature_dim: 編碼器的特徵維度。
        action_dim:  動作維度（離散為 One-hot，連續為原始數值）。
    """

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, phi_s: torch.Tensor, action_encoded: torch.Tensor) -> torch.Tensor:
        """
        引數：
            phi_s:          (batch, feature_dim) 編碼後的目前狀態。
            action_encoded: (batch, action_dim) One-hot 或連續動作。
        回傳：
            phi_s_next_pred: (batch, feature_dim) 預測的下一狀態特徵。
        """
        x = torch.cat([phi_s, action_encoded], dim=-1)
        return self.net(x)


class ICMModule(nn.Module):
    """
    完整的 ICM 模組，封裝了編碼器、逆向模型與前向模型。

    計算：
        內在獎勵 = eta * ||phi(s') - phi_hat(s')||^2

    引數：
        state_dim:   環境狀態維度。
        action_dim:  動作維度。
        feature_dim: 潛在特徵維度。
        discrete:    若為 True，使用離散動作空間。
        eta:         內在獎勵縮放因子。
        beta:        前向損失 (beta) 與逆向損失 (1-beta) 的平衡係數。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        discrete: bool = True,
        eta: float = 0.01,
        beta: float = 0.2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.discrete = discrete
        self.eta = eta
        self.beta = beta

        self.encoder = FeatureEncoder(state_dim, feature_dim)
        self.inverse = InverseModel(feature_dim, action_dim, discrete)
        self.forward_model = ForwardModel(feature_dim, action_dim)

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """對離散動作進行 One-hot 編碼；連續動作則直接透過。"""
        if self.discrete:
            return F.one_hot(action.long(), self.action_dim).float()
        return action

    def forward(self, state: torch.Tensor, next_state: torch.Tensor,
                action: torch.Tensor):
        """
        計算內在獎勵與 ICM 損失。

        引數：
            state:      (batch, state_dim)
            next_state: (batch, state_dim)
            action:     (batch,) 離散動作索引，或 (batch, action_dim) 連續動作

        回傳：
            intrinsic_reward: (batch,) 好奇心獎勵訊號。
            forward_loss:     純量 — 前向模型預測誤差。
            inverse_loss:     純量 — 逆向模型預測誤差。
        """
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)

        action_enc = self._encode_action(action)

        # 逆向模型：根據特徵對預測動作 (Predict action from features)
        action_pred = self.inverse(phi_s, phi_s_next)

        if self.discrete:
            inverse_loss = F.cross_entropy(action_pred, action.long())
        else:
            inverse_loss = F.mse_loss(action_pred, action)

        # 前向模型：根據目前特徵與動作預測下一個特徵 (Predict next feature)
        phi_s_next_pred = self.forward_model(phi_s.detach(), action_enc)
        forward_loss = 0.5 * F.mse_loss(phi_s_next_pred, phi_s_next.detach(),
                                          reduction="none").sum(dim=-1)

        # 內在獎勵 = 縮放後的前向預測誤差 (Scaled forward error)
        intrinsic_reward = self.eta * forward_loss.detach()

        # 組合後的 ICM 損失 (Combined ICM loss)
        icm_loss = (
            self.beta * forward_loss.mean()
            + (1 - self.beta) * inverse_loss
        )

        return intrinsic_reward, icm_loss, {
            "forward_loss": float(forward_loss.mean().item()),
            "inverse_loss": float(inverse_loss.item()),
        }
