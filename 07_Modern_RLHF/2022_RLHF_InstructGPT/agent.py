"""
RLHF (InstructGPT) 流程 — 來自人類回饋的強化學習 (Reinforcement Learning from Human Feedback)。

包含三階段訓練：
    1. 監督式微調 (Supervised Fine-Tuning, SFT)：在演示資料上微調語言模型。
    2. 獎勵模型 (Reward Model, RM)：在人類偏好對比資料上訓練獎勵模型。
    3. PPO RLHF：對比獎勵模型並加入 KL 懲罰項來最佳化 SFT 策略。

參考文獻：
    Ouyang, L., et al. (2022). Training language models to follow instructions
    with human feedback. NeurIPS 2022. arXiv:2203.02155.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# 輕量級 Transformer 元件（僅供示意，非生產規模）
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """單個 Transformer 區塊：包含自注意力機制 (Self-attention) 與前饋網路 (FFN)。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # 帶有因果掩碼 (Causal mask) 的自注意力機制
        h, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(h))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TinyLM(nn.Module):
    """
    用於演示目的的小型語言模型。

    生產環境中：請使用 HuggingFace GPT-2/LLaMA 並搭配 trl 函式庫。

    引數：
        vocab_size:  詞彙表大小。
        d_model:     嵌入層與隱藏層維度。
        n_layers:    Transformer 層數。
        n_heads:     注意力頭數量。
        max_seq_len: 最大序列長度。
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """用於自回歸解碼 (Autoregressive decoding) 的上三角掩碼。"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.masked_fill(mask, float("-inf"))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        引數：
            input_ids: (batch, seq_len) 的 token ID。
        回傳：
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        mask = self._causal_mask(T, input_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.lm_head(x)

    def get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """回傳最後一個 token 的隱藏狀態（供獎勵模型使用）。"""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        mask = self._causal_mask(T, input_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)[:, -1, :]  # 取最後一個 token


class RewardModel(nn.Module):
    """
    建立在語言模型骨幹上的獎勵模型 (Reward Model)。

    接收（提示 + 回應）token 並輸出一個標量獎勵值。
    透過 Bradley-Terry 成對排序損失進行訓練。

    引數：
        backbone: 預訓練語言模型（以 SFT 模型作為起點）。
        d_model:  骨幹模型的隱藏層維度。
    """

    def __init__(self, backbone: TinyLM):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        引數：
            input_ids: (batch, seq_len) 的提示 + 回應 token。
        回傳：
            reward: (batch,) 每個序列的標量獎勵。
        """
        hidden = self.backbone.get_hidden(input_ids)
        return self.reward_head(hidden).squeeze(-1)

    def ranking_loss(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> torch.Tensor:
        """
        Bradley-Terry 成對排序損失 (Pairwise ranking loss)。

        L = -E[ log sigma(r_chosen - r_rejected) ]

        被選中的回應 (Chosen) 應該比被拒絕的回應 (Rejected) 獲得更高的獎勵。
        """
        r_chosen = self.forward(chosen_ids)
        r_rejected = self.forward(rejected_ids)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        return loss


class RLHFAgent:
    """
    RLHF 訓練流程：SFT -> RM -> PPO。

    此類別負責編排三個訓練階段。
    在實際生產環境中，請整合 HuggingFace trl (TRL 函式庫)。

    引數：
        vocab_size:   詞彙表大小。
        d_model:      語言模型隱藏層維度。
        n_layers:     Transformer 層數。
        n_heads:      注意力頭數量。
        max_seq_len:  最大 token 序列長度。
        lr_sft:       SFT 學習率。
        lr_rm:        獎勵模型學習率。
        lr_ppo:       PPO 策略學習率。
        kl_coef:      KL 懲罰項係數（防止策略偏移過大）。
        clip_eps:     PPO 截斷 (Clip) 引數。
        device:       "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 128,
        lr_sft: float = 1e-4,
        lr_rm: float = 1e-4,
        lr_ppo: float = 1e-5,
        kl_coef: float = 0.1,
        clip_eps: float = 0.2,
        device: str = "cpu",
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.device = device

        # 第一階段：SFT 策略
        self.sft_model = TinyLM(vocab_size, d_model, n_layers, n_heads, max_seq_len).to(device)
        self.sft_optimizer = optim.AdamW(self.sft_model.parameters(), lr=lr_sft, weight_decay=0.1)

        # 第二階段：獎勵模型（從 SFT 骨幹初始化）
        self.reward_model = RewardModel(
            TinyLM(vocab_size, d_model, n_layers, n_heads, max_seq_len).to(device)
        ).to(device)
        self.rm_optimizer = optim.AdamW(self.reward_model.parameters(), lr=lr_rm)

        # 第三階段：PPO 策略（從 SFT 模型初始化） + 凍結的參考 SFT 模型
        self.policy = TinyLM(vocab_size, d_model, n_layers, n_heads, max_seq_len).to(device)
        self.ref_policy = TinyLM(vocab_size, d_model, n_layers, n_heads, max_seq_len).to(device)
        self.ppo_optimizer = optim.AdamW(self.policy.parameters(), lr=lr_ppo)

    # --- 第一階段：監督式微調 (Supervised Fine-Tuning) ---

    def sft_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        一步 SFT 梯度更新：在演示資料 (Demonstration data) 上執行交叉熵損失。

        引數：
            input_ids: (batch, seq_len) 的 token ID。
            labels:    (batch, seq_len) 目標 token（-100 代表忽略該提示 token）。
        """
        self.sft_model.train()
        logits = self.sft_model(input_ids)           # (B, T, V)

        # 為「預測下一個 token」進行偏移 (Shift)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        self.sft_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.sft_model.parameters(), max_norm=1.0)
        self.sft_optimizer.step()

        return {"sft_loss": float(loss.item())}

    # --- 第二階段：獎勵模型訓練 (Reward Model Training) ---

    def rm_step(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> dict:
        """
        一步獎勵模型梯度更新，基於偏好對比資料（較佳 > 較差）。

        引數：
            chosen_ids:   (batch, seq_len) 較佳的回應 token。
            rejected_ids: (batch, seq_len) 較差的回應 token。
        """
        self.reward_model.train()
        loss = self.reward_model.ranking_loss(chosen_ids, rejected_ids)

        # 獎勵差距指標 (Reward margin metric)
        with torch.no_grad():
            r_chosen = self.reward_model(chosen_ids)
            r_rejected = self.reward_model(rejected_ids)
            reward_margin = (r_chosen - r_rejected).mean()

        self.rm_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.reward_model.parameters(), max_norm=1.0)
        self.rm_optimizer.step()

        return {
            "rm_loss": float(loss.item()),
            "reward_margin": float(reward_margin.item()),
        }

    # --- 第三階段：PPO-RLHF ---

    @torch.no_grad()
    def compute_rewards(self, input_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """
        為 PPO 計算逐 token 的獎勵值。

        獎勵值 = r_RM - kl_coef * KL(policy || ref_policy)

        每個 token 的 KL 近似值：log pi(a|s) - log pi_ref(a|s)
        """
        # 序列末端的獎勵模型分數 (RM reward)
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        rm_reward = self.reward_model(full_ids)  # (batch,)

        # 逐 token 的 KL 散度 (KL per token)
        policy_logits = self.policy(response_ids)
        ref_logits = self.ref_policy(response_ids)
        policy_lp = F.log_softmax(policy_logits, dim=-1)
        ref_lp = F.log_softmax(ref_logits, dim=-1)
        kl_per_token = (policy_lp.exp() * (policy_lp - ref_lp)).sum(dim=-1)  # (B, T)
        kl_penalty = kl_per_token.sum(dim=-1)  # (B,)

        return rm_reward - self.kl_coef * kl_penalty

    def ppo_step(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict:
        """
        一步 PPO 梯度更新。

        優勢值 Advantage = 獎勵值 reward（此處為簡化版；生產環境請使用 GAE）。
        損失 Loss = -min(r * A, clip(r, 1-eps, 1+eps) * A)

        引數：
            prompt_ids:    (batch, prompt_len) 提示 ID。
            response_ids:  (batch, response_len) 回應 ID。
            old_log_probs: (batch,) 舊策略下的對數機率。
            rewards:       (batch,) 每個回應的標量獎勵。
        """
        self.policy.train()

        # 計算回應的目前對數機率 (Current log-probs)
        logits = self.policy(response_ids)  # (B, T, V)
        log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # 取回應 token 的對數機率平均值作為序列對數機率的代理值
        # TODO: 在生產環境中，應使用逐 token 的對數機率與 GAE 優勢值
        token_log_probs = log_probs_all.gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, T)
        new_log_probs = token_log_probs.mean(dim=-1)  # (B,)

        # PPO 比例 (Ratio)
        ratio = (new_log_probs - old_log_probs).exp()
        advantages = rewards - rewards.mean()  # 簡單的基準線 (Simple baseline)

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()

        self.ppo_optimizer.zero_grad()
        ppo_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.ppo_optimizer.step()

        return {
            "ppo_loss": float(ppo_loss.item()),
            "mean_reward": float(rewards.mean().item()),
            "mean_ratio": float(ratio.mean().item()),
        }

    # --- 從 SFT 初始化第三階段 ---

    def init_ppo_from_sft(self) -> None:
        """將 SFT 模型權重複製到策略網路與參考策略網路。"""
        self.policy.load_state_dict(self.sft_model.state_dict())
        self.ref_policy.load_state_dict(self.sft_model.state_dict())
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        print("PPO 策略與參考模型已從 SFT 模型初始化。")

    def initialize_rm_from_sft(self) -> None:
        """從 SFT 權重初始化獎勵模型骨幹（此為常用做法）。"""
        self.reward_model.backbone.load_state_dict(self.sft_model.state_dict())
        print("獎勵模型骨幹已從 SFT 模型初始化。")

    # --- 持久化 (Persistence) ---

    def save(self, path: str, phase: str = "ppo") -> None:
        os.makedirs(path, exist_ok=True)
        ckpt = {
            "sft_model": self.sft_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "policy": self.policy.state_dict(),
        }
        torch.save(ckpt, os.path.join(path, f"rlhf_{phase}.pt"))

    def load(self, path: str, phase: str = "ppo") -> None:
        ckpt = torch.load(os.path.join(path, f"rlhf_{phase}.pt"), map_location=self.device)
        self.sft_model.load_state_dict(ckpt["sft_model"])
        self.reward_model.load_state_dict(ckpt["reward_model"])
        self.policy.load_state_dict(ckpt["policy"])
