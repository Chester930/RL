"""
Dreamer 代理人 — 從潛在空間想像中學習行為。

參考文獻：
    Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019).
    Dream to Control: Learning Behaviors by Latent Imagination.
    ICLR 2020. arXiv:1912.01603.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

from common.base_agent import BaseAgent
from network import RSSM, ImageEncoder, ImageDecoder, ActorNetwork, ValueNetwork, StateEncoder, StateDecoder


class DreamerAgent(BaseAgent):
    """
    Dreamer 將世界模型訓練與行為學習分開：

    階段 1 — 世界模型訓練（使用真實資料）：
        - 編碼器：影像 -> 嵌入向量 (Embed)
        - RSSM 後驗分佈：(h, 嵌入向量) -> z (後驗)
        - RSSM 先驗分佈：h -> z_prior (無影像輔助)
        - 解碼器：(h, z) -> 影像（重建）
        - 獎勵預測器：(h, z) -> 獎勵
        - 損失函式 = 重建損失 + KL(後驗 || 先驗) + 獎勵預測損失

    階段 2 — 行為學習（在潛在想像中）：
        - 使用 RSSM 先驗分佈想像 H 步的取樣（無需與環境互動！）
        - 訓練演員以極大化想像的獎勵與價值引導 (Value bootstrapping)
        - 訓練價值函式以預測 lambda-回報 (Lambda-returns)

    引數：
        obs_channels:  影像通道數（灰階為 1，RGB 為 3）。
        action_dim:    動作維度。
        deter_dim:     RSSM 確定性狀態維度。
        stoch_dim:     RSSM 隨機性狀態維度。
        embed_dim:     影像編碼器輸出維度。
        imagine_horizon: 想像步數 H。
        kl_scale:      KL 散度損失的權重。
        gamma:         折扣因子。
        lambda_:       想像回報使用的 TD-lambda 引數。
    """

    def __init__(
        self,
        state_dim: int,       # Not used for image-based Dreamer
        action_dim: int,
        obs_channels: int = 3,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        embed_dim: int = 1024,
        lr_model: float = 6e-4,
        lr_actor: float = 8e-5,
        lr_critic: float = 8e-5,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        imagine_horizon: int = 15,
        kl_scale: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.imagine_horizon = imagine_horizon
        self.kl_scale = kl_scale
        latent_dim = deter_dim + stoch_dim

        # --- 世界模型元件 (World Model components) ---
        self.encoder = ImageEncoder(obs_channels, embed_dim).to(self.device)
        self.rssm = RSSM(deter_dim, stoch_dim, embed_dim, action_dim).to(self.device)
        self.decoder = ImageDecoder(latent_dim, obs_channels).to(self.device)
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ELU(),
            nn.Linear(400, 1),
        ).to(self.device)

        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_head.parameters()),
            lr=lr_model, eps=1e-5
        )

        # --- 行為學習 (潛在空間中的演員-評論家) ---
        self.actor = ActorNetwork(latent_dim, action_dim).to(self.device)
        self.critic = ValueNetwork(latent_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 儲存完整集數的重播緩衝區（Dreamer 需要序列資料）
        self._episode_buffer: deque = deque(maxlen=500_000)
        self._current_h = None
        self._current_z = None
        self._current_action = None

    def reset_state(self) -> None:
        """在集數開始時重置 RSSM 狀態。"""
        self._current_h, self._current_z = self.rssm.initial_state(1)
        self._current_action = torch.zeros(1, self.action_dim, device=self.device)

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """obs: (C, H, W) uint8 — resize to 64×64 then encode."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device) / 255.0  # (1,C,H,W)
        obs_t = F.interpolate(obs_t, size=(64, 64), mode='bilinear', align_corners=False)
        with torch.no_grad():
            embed = self.encoder(obs_t)
            h, z, _, _ = self.rssm.obs_step(
                self._current_h, self._current_z,
                self._current_action, embed
            )
            latent = torch.cat([h, z], dim=-1)
            action, _ = self.actor(latent)

        self._current_h = h
        self._current_z = z
        self._current_action = action
        return action.cpu().numpy()[0]

    def store(self, obs, action, reward, done):
        """obs: (H, W, C) uint8 — resize to (C, 64, 64) uint8 for compact storage."""
        obs_t = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0) / 255.0  # (1,C,H,W)
        obs_64 = F.interpolate(obs_t, size=(64, 64), mode='bilinear', align_corners=False)
        obs_u8 = (obs_64.squeeze(0).clamp(0, 1) * 255).byte().numpy()  # (C,64,64) uint8
        self._episode_buffer.append({
            "obs": obs_u8,
            "action": np.atleast_1d(np.array(action, dtype=np.float32)),
            "reward": float(reward),
            "done": done,
        })

    def update(self) -> dict:
        """
        雙階段 Dreamer 更新流程。

        階段 1：世界模型訓練（encoder + RSSM posterior + decoder + reward head）
        階段 2：潛在空間想像中的行為學習（actor + critic，lambda-returns）
        """
        SEQ = 16   # 每條序列長度
        B   = 4    # 批次大小（CPU 友好）

        buf = list(self._episode_buffer)
        if len(buf) < SEQ + 1:
            return {}

        # ── 取樣序列批次 ───────────────────────────────────────
        starts = np.random.randint(0, len(buf) - SEQ, size=B)
        obs_list, act_list, rew_list = [], [], []
        for s in starts:
            obs_list.append(np.stack([buf[s + t]["obs"]    for t in range(SEQ)]))  # (T,C,64,64)
            act_list.append(np.stack([buf[s + t]["action"] for t in range(SEQ)]))  # (T,ad)
            rew_list.append([buf[s + t]["reward"]           for t in range(SEQ)])

        obs_np = np.array(obs_list, dtype=np.uint8)    # (B,T,C,64,64)
        act_np = np.array(act_list, dtype=np.float32)  # (B,T,ad)
        rew_np = np.array(rew_list, dtype=np.float32)  # (B,T)

        obs_t    = torch.FloatTensor(obs_np).to(self.device) / 255.0  # (B,T,C,64,64)
        act_t    = torch.FloatTensor(act_np).to(self.device)           # (B,T,ad)
        rew_t    = torch.FloatTensor(rew_np).to(self.device)           # (B,T)
        obs_flat = obs_t.view(B * SEQ, *obs_t.shape[2:])               # (B*T,C,64,64)

        # ── 階段 1：世界模型訓練 ──────────────────────────────
        embed_flat = self.encoder(obs_flat)              # (B*T, embed_dim)
        embed      = embed_flat.view(B, SEQ, -1)         # (B, T, embed_dim)

        h, z = self.rssm.initial_state(B)
        post_ms, post_ss, prior_ms, prior_ss = [], [], [], []
        hs_list, zs_list = [], []

        for t in range(SEQ):
            a = act_t[:, t]  # (B, ad)
            e = embed[:, t]  # (B, embed_dim)
            h, z, pm, ps = self.rssm.obs_step(h, z, a, e)

            # 先驗（Prior）：從同一個 h 計算，不重跑 GRU
            pri = self.rssm.trans_net(h)
            pri_m, pri_ls = pri.chunk(2, -1)
            pri_s = F.softplus(pri_ls) + 0.1

            post_ms.append(pm);   post_ss.append(ps)
            prior_ms.append(pri_m); prior_ss.append(pri_s)
            hs_list.append(h);    zs_list.append(z)

        post_ms_t  = torch.stack(post_ms,  1)   # (B,T,stoch)
        post_ss_t  = torch.stack(post_ss,  1)
        prior_ms_t = torch.stack(prior_ms, 1)
        prior_ss_t = torch.stack(prior_ss, 1)
        hs_t = torch.stack(hs_list, 1)           # (B,T,deter)
        zs_t = torch.stack(zs_list, 1)           # (B,T,stoch)
        lats     = torch.cat([hs_t, zs_t], -1)  # (B,T,latent)
        lats_flat = lats.view(B * SEQ, -1)

        # 影像重建損失
        recon      = self.decoder(lats_flat)
        recon_loss = F.mse_loss(recon, obs_flat.detach())

        # 獎勵預測損失
        rew_pred    = self.reward_head(lats_flat).squeeze(-1)
        reward_loss = F.mse_loss(rew_pred, rew_t.view(-1))

        # KL(後驗 || 先驗)，free-bits = 1.0 防止過度壓縮
        post_dist  = torch.distributions.Normal(post_ms_t, post_ss_t)
        prior_dist = torch.distributions.Normal(prior_ms_t, prior_ss_t)
        kl         = torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1)
        kl_loss    = torch.clamp(kl, min=1.0).mean()

        model_loss = recon_loss + reward_loss + self.kl_scale * kl_loss
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.rssm.parameters()) +
            list(self.decoder.parameters()) + list(self.reward_head.parameters()),
            100.0
        )
        self.model_optimizer.step()

        # ── 階段 2：潛在空間想像中的行為學習 ─────────────────
        idx    = np.random.randint(0, B * SEQ, size=B)
        h_imag = hs_t.view(B * SEQ, -1)[idx].detach()
        z_imag = zs_t.view(B * SEQ, -1)[idx].detach()

        i_rews, i_vals, i_lats = [], [], []
        for _ in range(self.imagine_horizon):
            lat      = torch.cat([h_imag, z_imag], -1)
            act_i, _ = self.actor(lat)                              # 梯度流向 actor
            h_imag, z_imag, _, _ = self.rssm.img_step(h_imag, z_imag, act_i)
            lat_n    = torch.cat([h_imag, z_imag], -1)
            i_rews.append(self.reward_head(lat_n).squeeze(-1))
            i_vals.append(self.critic(lat_n).squeeze(-1))
            i_lats.append(lat_n)

        rews_t = torch.stack(i_rews)   # (H, B)
        vals_t = torch.stack(i_vals)
        H      = self.imagine_horizon

        # Lambda-returns（最後一步以 critic 引導，stop-gradient）
        lam_ret = torch.zeros_like(rews_t)
        last    = vals_t[-1].detach()
        for t in reversed(range(H)):
            if t == H - 1:
                lam_ret[t] = rews_t[t] + self.gamma * last
            else:
                lam_ret[t] = rews_t[t] + self.gamma * (
                    (1 - self.lambda_) * vals_t[t + 1].detach()
                    + self.lambda_     * lam_ret[t + 1].detach()
                )

        # Actor loss：極大化 lambda-return
        actor_loss = -lam_ret.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()

        # Critic loss：預測 lambda-return（使用 detach 避免更新世界模型）
        lats_i   = torch.stack(i_lats)   # (H, B, latent)
        val_pred = self.critic(lats_i.detach().view(H * B, -1)).squeeze(-1).view(H, B)
        critic_loss = F.mse_loss(val_pred, lam_ret.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        self.total_steps += 1
        return {
            "model/recon":   float(recon_loss.item()),
            "model/reward":  float(reward_loss.item()),
            "model/kl":      float(kl_loss.item()),
            "actor_loss":    float(actor_loss.item()),
            "critic_loss":   float(critic_loss.item()),
        }

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "decoder": self.decoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, os.path.join(path, "dreamer.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "dreamer.pt"), map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.rssm.load_state_dict(ckpt["rssm"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


class StateDreamerAgent(BaseAgent):
    """
    State-based Dreamer：使用低維狀態向量（非像素）作為觀測。

    與 DreamerAgent 相同的 RSSM 架構，但以 MLP 取代 CNN encoder/decoder，
    適合 Pendulum、CartPole 等低維連續控制任務，訓練速度顯著提升。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        embed_dim: int = 64,
        deter_dim: int = 128,
        stoch_dim: int = 20,
        lr_model: float = 6e-4,
        lr_actor: float = 8e-5,
        lr_critic: float = 8e-5,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        imagine_horizon: int = 15,
        kl_scale: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.imagine_horizon = imagine_horizon
        self.kl_scale = kl_scale
        self.action_scale = action_scale
        self.state_dim = state_dim
        latent_dim = deter_dim + stoch_dim

        self.encoder = StateEncoder(state_dim, embed_dim).to(device)
        self.rssm = RSSM(deter_dim, stoch_dim, embed_dim, action_dim).to(device)
        self.decoder = StateDecoder(latent_dim, state_dim).to(device)
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ELU(),
            nn.Linear(256, 1),
        ).to(device)

        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.rssm.parameters()) +
            list(self.decoder.parameters()) + list(self.reward_head.parameters()),
            lr=lr_model, eps=1e-5,
        )

        self.actor = ActorNetwork(latent_dim, action_dim).to(device)
        self.critic = ValueNetwork(latent_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self._episode_buffer: deque = deque(maxlen=200_000)
        self._current_h = None
        self._current_z = None
        self._current_action = None

    def reset_state(self) -> None:
        self._current_h, self._current_z = self.rssm.initial_state(1)
        self._current_action = torch.zeros(1, self.action_dim, device=self.device)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embed = self.encoder(state_t)
            h, z, _, _ = self.rssm.obs_step(
                self._current_h, self._current_z, self._current_action, embed
            )
            latent = torch.cat([h, z], dim=-1)
            action, _ = self.actor(latent)
        self._current_h = h
        self._current_z = z
        self._current_action = action
        return (action * self.action_scale).cpu().numpy()[0]

    def store(self, state, action, reward, done):
        self._episode_buffer.append({
            "obs":    np.array(state,  dtype=np.float32),
            "action": np.atleast_1d(np.array(action, dtype=np.float32) / self.action_scale),
            "reward": float(reward),
            "done":   bool(done),
        })

    def update(self) -> dict:
        SEQ = 16
        B   = 8

        buf = list(self._episode_buffer)
        if len(buf) < SEQ + 1:
            return {}

        starts = np.random.randint(0, len(buf) - SEQ, size=B)
        obs_list, act_list, rew_list = [], [], []
        for s in starts:
            obs_list.append(np.stack([buf[s + t]["obs"]    for t in range(SEQ)]))
            act_list.append(np.stack([buf[s + t]["action"] for t in range(SEQ)]))
            rew_list.append([buf[s + t]["reward"]           for t in range(SEQ)])

        obs_np = np.array(obs_list, dtype=np.float32)   # (B,T,state_dim)
        act_np = np.array(act_list, dtype=np.float32)   # (B,T,action_dim)
        rew_np = np.array(rew_list, dtype=np.float32)   # (B,T)

        obs_t    = torch.FloatTensor(obs_np).to(self.device)   # (B,T,state_dim)
        act_t    = torch.FloatTensor(act_np).to(self.device)   # (B,T,ad)
        rew_t    = torch.FloatTensor(rew_np).to(self.device)   # (B,T)
        obs_flat = obs_t.view(B * SEQ, self.state_dim)         # (B*T,state_dim)

        # 階段 1：世界模型
        embed_flat = self.encoder(obs_flat)              # (B*T, embed_dim)
        embed      = embed_flat.view(B, SEQ, -1)

        h, z = self.rssm.initial_state(B)
        post_ms, post_ss, prior_ms, prior_ss = [], [], [], []
        hs_list, zs_list = [], []

        for t in range(SEQ):
            a = act_t[:, t]
            e = embed[:, t]
            h, z, pm, ps = self.rssm.obs_step(h, z, a, e)
            pri = self.rssm.trans_net(h)
            pri_m, pri_ls = pri.chunk(2, -1)
            pri_s = F.softplus(pri_ls) + 0.1
            post_ms.append(pm);   post_ss.append(ps)
            prior_ms.append(pri_m); prior_ss.append(pri_s)
            hs_list.append(h);    zs_list.append(z)

        post_ms_t  = torch.stack(post_ms,  1)
        post_ss_t  = torch.stack(post_ss,  1)
        prior_ms_t = torch.stack(prior_ms, 1)
        prior_ss_t = torch.stack(prior_ss, 1)
        hs_t = torch.stack(hs_list, 1)
        zs_t = torch.stack(zs_list, 1)
        lats      = torch.cat([hs_t, zs_t], -1)
        lats_flat = lats.view(B * SEQ, -1)

        recon      = self.decoder(lats_flat)
        recon_loss = F.mse_loss(recon, obs_flat.detach())

        rew_pred    = self.reward_head(lats_flat).squeeze(-1)
        reward_loss = F.mse_loss(rew_pred, rew_t.view(-1))

        post_dist  = torch.distributions.Normal(post_ms_t, post_ss_t)
        prior_dist = torch.distributions.Normal(prior_ms_t, prior_ss_t)
        kl         = torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1)
        kl_loss    = torch.clamp(kl, min=1.0).mean()

        model_loss = recon_loss + reward_loss + self.kl_scale * kl_loss
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.rssm.parameters()) +
            list(self.decoder.parameters()) + list(self.reward_head.parameters()),
            100.0,
        )
        self.model_optimizer.step()

        # 階段 2：潛在空間想像
        idx    = np.random.randint(0, B * SEQ, size=B)
        h_imag = hs_t.view(B * SEQ, -1)[idx].detach()
        z_imag = zs_t.view(B * SEQ, -1)[idx].detach()

        i_rews, i_vals, i_lats = [], [], []
        for _ in range(self.imagine_horizon):
            lat      = torch.cat([h_imag, z_imag], -1)
            act_i, _ = self.actor(lat)
            h_imag, z_imag, _, _ = self.rssm.img_step(h_imag, z_imag, act_i)
            lat_n = torch.cat([h_imag, z_imag], -1)
            i_rews.append(self.reward_head(lat_n).squeeze(-1))
            i_vals.append(self.critic(lat_n).squeeze(-1))
            i_lats.append(lat_n)

        rews_t = torch.stack(i_rews)
        vals_t = torch.stack(i_vals)
        H      = self.imagine_horizon

        lam_ret = torch.zeros_like(rews_t)
        last    = vals_t[-1].detach()
        for t in reversed(range(H)):
            if t == H - 1:
                lam_ret[t] = rews_t[t] + self.gamma * last
            else:
                lam_ret[t] = rews_t[t] + self.gamma * (
                    (1 - self.lambda_) * vals_t[t + 1].detach()
                    + self.lambda_     * lam_ret[t + 1].detach()
                )

        actor_loss = -lam_ret.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()

        lats_i   = torch.stack(i_lats)
        val_pred = self.critic(lats_i.detach().view(H * B, -1)).squeeze(-1).view(H, B)
        critic_loss = F.mse_loss(val_pred, lam_ret.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        self.total_steps += 1
        return {
            "model/recon":  float(recon_loss.item()),
            "model/reward": float(reward_loss.item()),
            "model/kl":     float(kl_loss.item()),
            "actor_loss":   float(actor_loss.item()),
            "critic_loss":  float(critic_loss.item()),
        }

    def save(self, path: str) -> None:
        self._ensure_dir(path)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "rssm":    self.rssm.state_dict(),
            "decoder": self.decoder.state_dict(),
            "actor":   self.actor.state_dict(),
            "critic":  self.critic.state_dict(),
        }, os.path.join(path, "dreamer_state.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "dreamer_state.pt"), map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.rssm.load_state_dict(ckpt["rssm"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
