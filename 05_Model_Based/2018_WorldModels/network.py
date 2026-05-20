"""
World Models 網路元件：VAE、MDN-RNN 以及控制器 (Controller)。

參考文獻：
    Ha, D., & Schmidhuber, J. (2018). World Models.
    arXiv:1803.10122. NeurIPS 2018.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAE(nn.Module):
    """
    變分自編碼器 (Variational Autoencoder) — 視覺元件 (Visual component / V)。

    將每一幀影像壓縮為低維潛在向量 z。
    透過重建 z 的觀測值來進行訓練。

    網路架構（適用於 64x64 影像）：
        編碼器 (Encoder)：卷積層 -> mu, log_var
        解碼器 (Decoder)：全連線層 + 反摺積層 -> 重建影像
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 32, img_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2), nn.ReLU(),   # 31
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),            # 14
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),           # 6
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),          # 2
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            enc_out_dim = self.encoder(dummy).shape[1]

        self.mu_head = nn.Linear(enc_out_dim, latent_dim)
        self.log_var_head = nn.Linear(enc_out_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 6, stride=2), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_head(h), self.log_var_head(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * epsilon, 其中 epsilon ~ N(0,I)"""
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z).view(-1, 1024, 1, 1)
        return self.decoder(x)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var, z


class MDNRNN(nn.Module):
    """
    混合密度網路 RNN (Mixture Density Network RNN) — 記憶元件 (Memory component / M)。

    在給定 (z_t, a_t, h_t) 的條件下，使用混合高斯分佈預測下一個潛在狀態 (z_{t+1}) 的分佈。

    提供時間記憶以及隨機性的未來預測。

    引數：
        z_dim:      VAE 潛在維度。
        action_dim: 環境動作維度。
        hidden_dim: LSTM 隱藏層維度。
        n_mixtures: 高斯混合成分的數量。
    """

    def __init__(self, z_dim: int = 32, action_dim: int = 3,
                 hidden_dim: int = 256, n_mixtures: int = 5):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures

        self.lstm = nn.LSTM(z_dim + action_dim, hidden_dim, batch_first=True)

        # MDN 輸出：Logits（混合權重）、均值 (mus)、對數標準差 (log_sigmas) 以及結束機率
        out_dim = n_mixtures * (2 * z_dim + 1) + 1  # (pi, mu, sigma) * n_mix + done
        self.output_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor,
                hidden=None):
        """
        引數：
            z: (batch, seq, z_dim)
            a: (batch, seq, action_dim)
        回傳：
            pi, mu, sigma: 混合引數，形狀為 (batch, seq, n_mix, *)
            done_prob: (batch, seq)
            hidden: LSTM 狀態
        """
        x = torch.cat([z, a], dim=-1)
        out, hidden = self.lstm(x, hidden)
        params = self.output_head(out)

        # 分割輸出內容 (Split outputs)
        n = self.n_mixtures
        zd = self.z_dim

        log_pi = params[..., :n]
        mu = params[..., n:n + n * zd].view(*params.shape[:-1], n, zd)
        log_sigma = params[..., n + n * zd:n + 2 * n * zd].view(*params.shape[:-1], n, zd)
        done_logit = params[..., -1]

        pi = F.softmax(log_pi, dim=-1)
        sigma = log_sigma.exp()
        done_prob = torch.sigmoid(done_logit)

        return pi, mu, sigma, done_prob, hidden


class Controller(nn.Module):
    """
    線性控制器 (Linear Controller) — 控制器元件 (Controller component / C)。

    一個簡單的線性模型（沒有隱藏層！），負責將 (z, h) 對映至動作。
    由 CMA-ES（演化策略）訓練，而非梯度下降法。
    這種極致的簡化是刻意設計的：系統的複雜性主要集中在 V (視覺) 與 M (記憶) 元件。

    引數：
        z_dim:      VAE 潛在維度。
        h_dim:      MDN-RNN 隱藏層維度。
        action_dim: 動作數量（輸出維度）。
    """

    def __init__(self, z_dim: int = 32, h_dim: int = 256, action_dim: int = 3):
        super().__init__()
        self.linear = nn.Linear(z_dim + h_dim, action_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        引數：
            z: VAE 潛在向量 (batch, z_dim)
            h: MDN-RNN 隱藏狀態 (batch, h_dim)
        回傳：
            action: (batch, action_dim)，經過 tanh 處理以適用於連續動作
        """
        return torch.tanh(self.linear(torch.cat([z, h], dim=-1)))
