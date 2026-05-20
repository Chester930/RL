"""
World Models 代理人 — 整合了 VAE (V)、MDN-RNN (M) 以及控制器 (C)。

三階段訓練流程：
    第一階段：在隨機取樣的影像幀上訓練 VAE。
    第二階段：在 (z_t, a_t) -> z_{t+1} 序列上訓練 MDN-RNN。
    第三階段：在「夢境環境 (Dream environment)」中透過 CMA-ES 最佳化控制器權重。

參考文獻：
    Ha, D., & Schmidhuber, J. (2018). World Models.
    arXiv:1803.10122. NeurIPS 2018.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.base_agent import BaseAgent
from network import VAE, MDNRNN, Controller


class WorldModelsAgent(BaseAgent):
    """
    結合了 V (VAE)、M (MDN-RNN) 與 C (控制器) 的 World Models 代理人。

    推理過程：
        1. 透過 VAE 將觀測影像編碼為潛在向量 z。
        2. 透過控制器將 [z, h] 對映至動作。
        3. 透過 MDN-RNN(z, a) 更新 LSTM 的隱藏狀態 h。

    引數：
        obs_channels:  影像通道數（RGB 為 3）。
        img_size:      影像尺寸（CarRacing 環境為 64）。
        latent_dim:    VAE 的潛在維度 (z_dim)。
        hidden_dim:    MDN-RNN 中 LSTM 的隱藏層維度。
        action_dim:    環境的動作維度。
        n_mixtures:    MDN 中的高斯混合成分數量。
        device:        "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        obs_channels: int = 3,
        img_size: int = 64,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        action_dim: int = 3,
        n_mixtures: int = 5,
        device: str = "cpu",
    ):
        super().__init__(latent_dim, action_dim, device)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # --- V：視覺元件 (Visual component) ---
        self.vae = VAE(
            in_channels=obs_channels,
            latent_dim=latent_dim,
            img_size=img_size,
        ).to(self.device)

        # --- M：記憶元件 (Memory component) ---
        self.mdnrnn = MDNRNN(
            z_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_mixtures=n_mixtures,
        ).to(self.device)

        # --- C：控制器元件 (Controller component，由 CMA-ES 訓練) ---
        self.controller = Controller(
            z_dim=latent_dim,
            h_dim=hidden_dim,
            action_dim=action_dim,
        ).to(self.device)

        # 執行時的 MDN-RNN 隱藏狀態 (Runtime hidden state)
        self._hidden = None

    # ------------------------------------------------------------------
    # 推理輔助函式 (Inference helpers)
    # ------------------------------------------------------------------

    def reset_hidden(self, batch_size: int = 1):
        """在每一集開始前重置 LSTM 的隱藏狀態。"""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        self._hidden = (h, c)

    @torch.no_grad()
    def encode_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        將單一觀測影像編碼為潛在向量 z（評估時取 mu）。

        引數：
            obs: (H, W, C) uint8 型別的 numpy 陣列。
        回傳：
            z: (1, latent_dim) 維度的張量 (Tensor)。
        """
        self.vae.eval()
        x = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        mu, _ = self.vae.encode(x)
        return mu

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        將觀測影像編碼為 z，輸入 [z, h] 至控制器取得動作。
        利用 (z, action) 更新 MDN-RNN 的隱藏狀態。

        引數：
            obs: (H, W, C) 原始畫素觀測影像。
        回傳：
            action: (action_dim,) numpy 陣列，數值在 [-1, 1] 之間。
        """
        if self._hidden is None:
            self.reset_hidden()

        z = self.encode_obs(obs)  # (1, z_dim)

        # 從 LSTM 隱藏狀態中提取 h
        h = self._hidden[0].squeeze(0)  # (1, hidden_dim)

        action = self.controller(z, h)  # (1, action_dim)

        # 更新 MDN-RNN 的隱藏狀態 (Update MDN-RNN hidden state)
        with torch.no_grad():
            z_seq = z.unsqueeze(1)       # (1, 1, z_dim)
            a_seq = action.unsqueeze(1)  # (1, 1, action_dim)
            _, _, _, _, self._hidden = self.mdnrnn(z_seq, a_seq, self._hidden)

        return action.squeeze(0).cpu().detach().numpy()

    # ------------------------------------------------------------------
    # VAE 訓練 (VAE training)
    # ------------------------------------------------------------------

    def vae_loss(self, x: torch.Tensor, recon: torch.Tensor,
                 mu: torch.Tensor, log_var: torch.Tensor,
                 kl_weight: float = 1.0) -> torch.Tensor:
        """
        ELBO 損失：重建損失 (BCE) + KL 散度。

        公式：KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        """
        # 畫素級重建損失 (Pixel-wise reconstruction loss)
        recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
        recon_loss = recon_loss / x.shape[0]  # 每個樣本的平均值 (per-sample average)

        # KL 散度（高斯分佈的閉式解，KL divergence）
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]

        return recon_loss + kl_weight * kl

    def update_vae(self, frames: np.ndarray, optimizer: optim.Optimizer,
                   kl_weight: float = 1.0) -> dict:
        """
        對一批影像幀執行一次梯度更新。

        引數：
            frames: (batch, H, W, C) uint8 型別的陣列。
        回傳：
            包含 "vae_loss" 的指標字典。
        """
        self.vae.train()
        x = torch.FloatTensor(frames).permute(0, 3, 1, 2).to(self.device) / 255.0

        recon, mu, log_var, _ = self.vae(x)
        loss = self.vae_loss(x, recon, mu, log_var, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
        optimizer.step()

        return {"vae_loss": float(loss.item())}

    # ------------------------------------------------------------------
    # MDN-RNN 訓練 (MDN-RNN training)
    # ------------------------------------------------------------------

    def mdnrnn_loss(self, pi, mu, sigma, done_prob,
                    z_next: torch.Tensor, done_target: torch.Tensor) -> torch.Tensor:
        """
        在混合高斯分佈下 z_{t+1} 的負對數似然 (NLL)。

        公式：NLL = -log sum_k pi_k * N(z_{t+1}; mu_k, sigma_k^2)
        """
        # z_next: (batch, seq, z_dim) -> 為混合維度增加維度 (Unsqueeze)
        z_expanded = z_next.unsqueeze(-2)  # (batch, seq, 1, z_dim)

        # 各個成分下的對數機率 (Log-prob under each component)
        log_probs = (
            -0.5 * ((z_expanded - mu) / sigma).pow(2)
            - sigma.log()
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1)  # (batch, seq, n_mix)

        # 在對數空間進行加權求和（使用 log-sum-exp 技巧）
        log_pi = pi.log().clamp(min=-1e8)
        log_mix = torch.logsumexp(log_pi + log_probs, dim=-1)  # (batch, seq)
        mdn_loss = -log_mix.mean()

        # 終結預測（二元交叉熵損失，Binary cross-entropy）
        done_loss = nn.functional.binary_cross_entropy(
            done_prob, done_target.float()
        )

        return mdn_loss + done_loss

    def update_mdnrnn(self, z_seq: torch.Tensor, a_seq: torch.Tensor,
                      z_next_seq: torch.Tensor, done_seq: torch.Tensor,
                      optimizer: optim.Optimizer) -> dict:
        """
        對一個序列批次執行一次梯度更新。

        引數：
            z_seq:      (batch, seq, z_dim) 當前的潛在向量。
            a_seq:      (batch, seq, action_dim) 動作。
            z_next_seq: (batch, seq, z_dim) 下一個潛在向量（目標）。
            done_seq:   (batch, seq) 結束標記。
        """
        self.mdnrnn.train()
        z_seq = z_seq.to(self.device)
        a_seq = a_seq.to(self.device)
        z_next_seq = z_next_seq.to(self.device)
        done_seq = done_seq.to(self.device)

        pi, mu, sigma, done_prob, _ = self.mdnrnn(z_seq, a_seq)
        loss = self.mdnrnn_loss(pi, mu, sigma, done_prob, z_next_seq, done_seq)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.mdnrnn.parameters(), max_norm=1.0)
        optimizer.step()

        return {"mdnrnn_loss": float(loss.item())}

    # ------------------------------------------------------------------
    # 控制器 (CMA-ES 介面)
    # ------------------------------------------------------------------

    def get_controller_params(self) -> np.ndarray:
        """將控制器所有的引數攤平為 1-D numpy 陣列。"""
        return np.concatenate([
            p.data.cpu().numpy().ravel()
            for p in self.controller.parameters()
        ])

    def set_controller_params(self, flat_params: np.ndarray) -> None:
        """將攤平的引數向量載入至控制器。"""
        idx = 0
        for p in self.controller.parameters():
            size = p.numel()
            p.data.copy_(
                torch.FloatTensor(flat_params[idx: idx + size]).view(p.shape)
            )
            idx += size

    def update(self) -> dict:
        """不直接使用；VAE、MDN-RNN 與控制器各自擁有獨立的更新路徑。"""
        return {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "vae": self.vae.state_dict(),
            "mdnrnn": self.mdnrnn.state_dict(),
            "controller": self.controller.state_dict(),
        }, os.path.join(path, "world_models.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(
            os.path.join(path, "world_models.pt"), map_location=self.device
        )
        self.vae.load_state_dict(ckpt["vae"])
        self.mdnrnn.load_state_dict(ckpt["mdnrnn"])
        self.controller.load_state_dict(ckpt["controller"])
