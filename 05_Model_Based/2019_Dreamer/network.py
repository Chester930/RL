"""
Dreamer 網路元件：RSSM 世界模型、影像編碼器/解碼器、演員網路與價值網路。

參考文獻：
    Hafner, D., et al. (2019). Dream to Control: Learning Behaviors by
    Latent Imagination. ICLR 2020. arXiv:1912.01603.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ImageEncoder(nn.Module):
    """
    CNN 編碼器：觀測影像 -> 潛在特徵向量。
    輸入：(batch, C, H, W) 影像
    輸出：(batch, embed_dim) 特徵向量
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 48, 4, stride=2), nn.ELU(),
            nn.Conv2d(48, 96, 4, stride=2), nn.ELU(),
            nn.Conv2d(96, 192, 4, stride=2), nn.ELU(),
            nn.Conv2d(192, 384, 4, stride=2), nn.ELU(),
            nn.Flatten(),
        )
        # 計算輸出維度 (Compute output size)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            cnn_out = self.cnn(dummy).shape[1]
        self.linear = nn.Linear(cnn_out, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


class ImageDecoder(nn.Module):
    """
    CNN 解碼器：潛在狀態 -> 重建影像。
    輸入：(batch, latent_dim) 狀態
    輸出：(batch, C, H, W) 影像（高斯分佈的均值）
    """

    def __init__(self, latent_dim: int, out_channels: int = 3):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 32)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 192, 5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(192, 96, 5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(96, 48, 6, stride=2), nn.ELU(),
            nn.ConvTranspose2d(48, out_channels, 6, stride=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.linear(z).view(-1, 32, 1, 1)
        return self.deconv(x)


class RSSM(nn.Module):
    """
    迴圈狀態空間模型 (Recurrent State Space Model / RSSM) — Dreamer 的核心。

    維護兩個狀態元件：
    - h_t: 確定性迴圈狀態 (GRU 隱藏狀態)
    - z_t: 隨機性潛在向量（從後驗/先驗分佈中取樣）

    演員與評論家網路皆是以結合後的狀態 (h_t, z_t) 為輸入條件。

    核心元件：
    - 序列模型：h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  (使用 GRU)
    - 表示模型：z_t ~ q(z_t | h_t, x_t)       (後驗分佈，使用觀測影像)
    - 狀態轉移預測器：z_t ~ p(z_t | h_t)      (先驗分佈，不使用觀測影像)

    引數：
        deter_dim:   確定性狀態 h 的維度。
        stoch_dim:   隨機性狀態 z 的維度。
        embed_dim:   觀測影像嵌入向量的維度。
        action_dim:  動作維度。
    """

    def __init__(
        self,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        embed_dim: int = 1024,
        action_dim: int = 4,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # 用於處理確定性狀態的 GRU (Deterministic state)
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        # 表示模型（後驗分佈，Posterior）：q(z | h, x)
        self.repr_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 256), nn.ELU(),
            nn.Linear(256, 2 * stoch_dim),  # 均值 (mean) 與對數標準差 (log_std)
        )

        # 狀態轉移預測器（先驗分佈，Prior）：p(z | h)
        self.trans_net = nn.Sequential(
            nn.Linear(deter_dim, 256), nn.ELU(),
            nn.Linear(256, 2 * stoch_dim),  # 均值 (mean) 與對數標準差 (log_std)
        )

    def obs_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        包含觀測影像的單一步驟（用於收集真實經驗時）。

        回傳：
            h_next: 新的確定性狀態
            z_next: 後驗隨機性狀態（使用觀測影像）
            post_mean, post_std: 後驗分佈引數
        """
        # 確定性步進：h_t = GRU(z_{t-1}, a_{t-1}, h_{t-1})
        h_next = self.gru(torch.cat([z, a], dim=-1), h)

        # 後驗分佈：z_t ~ q(z | h, embed)
        post_params = self.repr_net(torch.cat([h_next, embed], dim=-1))
        post_mean, post_log_std = post_params.chunk(2, dim=-1)
        post_std = F.softplus(post_log_std) + 0.1
        z_next = post_mean + post_std * torch.randn_like(post_mean)

        return h_next, z_next, post_mean, post_std

    def img_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        不含觀測影像的單一步驟（用於想像 / 規劃時）。

        回傳：
            h_next, z_next, 先驗均值, 先驗標準差
        """
        h_next = self.gru(torch.cat([z, a], dim=-1), h)

        # 先驗分佈：z_t ~ p(z | h)
        prior_params = self.trans_net(h_next)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.1
        z_next = prior_mean + prior_std * torch.randn_like(prior_mean)

        return h_next, z_next, prior_mean, prior_std

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """回傳以零初始化的 (h, z) 狀態。"""
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.deter_dim, device=device),
            torch.zeros(batch_size, self.stoch_dim, device=device),
        )


class ActorNetwork(nn.Module):
    """以 RSSM 潛在狀態為條件的連續動作策略 (Policy)。"""

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 2 * action_dim),  # mean + log_std
        )
        self.action_dim = action_dim

    def forward(self, latent: torch.Tensor):
        out = self.net(latent)
        mu, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-4
        dist = torch.distributions.Normal(mu, std)
        action = torch.tanh(dist.rsample())
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class ValueNetwork(nn.Module):
    """以 RSSM 潛在狀態為條件的價值函式 (Value function)。"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class StateEncoder(nn.Module):
    """MLP 編碼器：低維狀態向量 -> 潛在嵌入。"""

    def __init__(self, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateDecoder(nn.Module):
    """MLP 解碼器：潛在狀態 -> 重建狀態向量。"""

    def __init__(self, latent_dim: int, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
