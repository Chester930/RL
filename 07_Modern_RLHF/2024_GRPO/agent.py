"""
GRPO — 群體相對策略最佳化 (Group Relative Policy Optimization)。

相對於 PPO-RLHF 的核心創新：
    - 完全去除了價值/評論家 (Value/Critic) 網路。
    - 透過「群體 (GROUP)」對比來估計優勢值 (Advantages)：
      對於每個提示詞，取樣 G 個回應，計算每個回應的獎勵值，
      然後在該群體內對獎勵進行正規化處理以獲得優勢值。
    - 這避免了訓練獨立價值函式的需求，同時能保持訓練過程穩定。

應用案例：DeepSeek-R1、Qwen 推理模型。

參考文獻：
    Shao, Z., Wang, P., Zhu, Q., et al. (2024).
    DeepSeekMath: Pushing the limits of mathematical reasoning in open language models.
    arXiv:2402.03300.

    以及：DeepSeek-R1 技術報告 (2025). arXiv:2501.12948.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Callable, Optional


class TinyLM(nn.Module):
    """用於展示 GRPO 的極簡自回歸語言模型。"""

    def __init__(self, vocab_size: int = 1000, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, max_seq_len: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.lm_head(self.norm(x))  # (B, T, V)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        用於回應生成的貪婪取樣 (Greedy) / 溫度取樣 (Temperature sampling)。

        引數：
            prompt_ids:    (batch, prompt_len) 提示詞 ID。
            max_new_tokens: 要生成的回應 token 數量。
        回傳：
            generated: (batch, prompt_len + max_new_tokens) 的完整序列。
        """
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(ids)[:, -1, :]  # (B, V)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return ids


def compute_sequence_log_prob(model: TinyLM, input_ids: torch.Tensor,
                               prompt_len: int) -> torch.Tensor:
    """
    計算回應 (Response) token 的對數機率總和。

    引數：
        model:      語言模型。
        input_ids:  (batch, seq_len) 提示 + 回應的 token ID。
        prompt_len: 要跳過的提示 token 數量。
    回傳：
        log_probs: (batch,) 對數機率總和張量。
    """
    logits = model(input_ids)                              # (B, T, V)
    shift_logits = logits[:, :-1, :]                       # (B, T-1, V)
    shift_labels = input_ids[:, 1:]                        # (B, T-1)

    log_p = F.log_softmax(shift_logits, dim=-1)            # (B, T-1, V)
    token_log_p = log_p.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # 僅對回應 token 進行累加求和
    mask = torch.zeros_like(token_log_p)
    mask[:, prompt_len:] = 1.0

    return (token_log_p * mask).sum(dim=-1)  # (B,)


class GRPOAgent:
    """
    GRPO：群體相對策略最佳化 (Group Relative Policy Optimization)。

    核心演演算法流程：
        對於每個提示詞 x：
            1. 從目前策略中取樣 G 個回應 {y_1, ..., y_G}。
            2. 使用獎勵函式 r(x, y_i) 為每個回應打分。
            3. 在群體內將分數正規化以獲得優勢值 (Advantages)：
               A_i = (r_i - mean(r)) / std(r)    （群體相對基準）
            4. 透過類 PPO 的截斷目標函式更新策略，並加入 KL 懲罰項。

    完全不需要價值網路 — 群體平均值本身就是最完美的基準線！

    引數：
        vocab_size:    詞彙表大小。
        d_model:       語言模型隱藏層維度。
        n_layers:      Transformer 層數。
        n_heads:       注意力頭數量。
        max_seq_len:   最大序列長度。
        group_size:    G — 每個提示詞取樣的回應數量。
        beta:          KL 正規化係數（對比參考策略的懲罰項）。
        clip_eps:      PPO 截斷 (Clip) 引數。
        lr:            學習率。
        temperature:   生成回應時的取樣溫度。
        device:        "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 256,
        group_size: int = 8,
        beta: float = 0.04,
        clip_eps: float = 0.2,
        lr: float = 1e-6,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.group_size = group_size
        self.beta = beta
        self.clip_eps = clip_eps
        self.temperature = temperature
        self.device = device

        # 策略模型（可訓練）
        self.model = TinyLM(vocab_size, d_model, n_layers, n_heads, max_seq_len).to(device)
        # 參考模型（凍結的 SFT 或先前的策略檢查點）
        self.ref_model = TinyLM(vocab_size, d_model, n_layers, n_heads, max_seq_len).to(device)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                                     betas=(0.9, 0.95), weight_decay=0.1)

    def init_from_sft(self, sft_state_dict: dict) -> None:
        """從 SFT 模型初始化策略模型與參考模型。"""
        self.model.load_state_dict(sft_state_dict)
        self.ref_model.load_state_dict(sft_state_dict)
        print("GRPO：策略模型與參考模型已從 SFT 權重初始化。")

    # --- 群體取樣與獎勵計算 (Group sampling and reward computation) ---

    @torch.no_grad()
    def sample_group(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        為單個提示詞取樣 G 個回應。

        引數：
            prompt_ids: (1, prompt_len) 或 (prompt_len,) 單個提示詞。
        回傳：
            responses: (G, prompt_len + max_new_tokens) 完整序列張量。
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        # 將提示詞重複 G 次
        prompts = prompt_ids.repeat(self.group_size, 1)  # (G, prompt_len)
        return self.model.generate(prompts, max_new_tokens, self.temperature)

    def compute_group_advantages(self, rewards: torch.Tensor,
                                  eps: float = 1e-8) -> torch.Tensor:
        """
        在群體內正規化獎勵以獲得優勢值 (Advantages)。

        A_i = (r_i - mean(r)) / (std(r) + eps)

        這是 GRPO 的核心創新：無需學習價值函式即可獲得群體相對基準。

        引數：
            rewards: (G,) 群體中每個回應的獎勵。
        回傳：
            advantages: (G,) 正規化後的優勢值。
        """
        mean_r = rewards.mean()
        std_r = rewards.std() + eps
        return (rewards - mean_r) / std_r

    # --- GRPO 損失函式計算 (GRPO loss computation) ---

    def grpo_loss(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        prompt_len: int,
    ) -> tuple:
        """
        包含 KL 懲罰項的 GRPO 截斷策略梯度損失。

        L = -E[ min(r * A, clip(r, 1-eps, 1+eps) * A) ] + beta * KL(pi || pi_ref)

        其中 r = pi_theta / pi_old (重要性權重)
              A = 群體正規化後的優勢值

        引數：
            prompt_ids:    (G, prompt_len)
            response_ids:  (G, prompt_len + response_len) 完整序列
            advantages:    (G,) 群體相對優勢值
            old_log_probs: (G,) 舊策略下的對數機率（更新前）
            prompt_len:    提示詞 token 數量
        回傳：
            loss:    損失標量
            metrics: 指標字典
        """
        G = response_ids.shape[0]

        # 目前策略的對數機率 (Current policy log-probs)
        new_log_probs = compute_sequence_log_prob(self.model, response_ids, prompt_len)

        # 參考策略的對數機率 (Reference policy log-probs) — 用於 KL 懲罰項
        with torch.no_grad():
            ref_log_probs = compute_sequence_log_prob(self.ref_model, response_ids, prompt_len)

        # PPO 比例 (Ratio)
        ratio = (new_log_probs - old_log_probs).exp()

        # 截斷的代理損失 (Clipped surrogate loss)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL 懲罰：beta * (log pi_theta - log pi_ref)
        kl = (new_log_probs - ref_log_probs).mean()
        kl_loss = self.beta * kl

        total_loss = policy_loss + kl_loss

        return total_loss, {
            "policy_loss": float(policy_loss.item()),
            "kl": float(kl.item()),
            "total_loss": float(total_loss.item()),
            "mean_ratio": float(ratio.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
        }

    # --- 完整 GRPO 更新步數 (Full GRPO step) ---

    def update(
        self,
        prompt_ids: torch.Tensor,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_new_tokens: int = 64,
        n_epochs: int = 1,
    ) -> dict:
        """
        針對單個提示詞的完整 GRPO 更新流程。

        步驟：
            1. 從目前策略中取樣 G 個回應。
            2. 使用獎勵函式為回應評分。
            3. 計算群體相對優勢值。
            4. 包含 KL 懲罰項的 PPO 更新（執行 n_epochs 次）。

        引數：
            prompt_ids:   (prompt_len,) 單個提示詞的 token ID。
            reward_fn:    回呼函式(prompt_ids, response_ids) -> 回傳 (G,) 的獎勵張量。
            max_new_tokens: 生成回應的長度。
            n_epochs:     對同一群體資料進行 PPO 最佳化的次數。
        """
        prompt_len = prompt_ids.shape[-1]

        # 步驟 1：取樣 G 個回應
        with torch.no_grad():
            responses = self.sample_group(prompt_ids, max_new_tokens)  # (G, full_len)

        # 步驟 2：為回應評分
        with torch.no_grad():
            prompt_repeated = prompt_ids.unsqueeze(0).repeat(self.group_size, 1)
            rewards = reward_fn(prompt_repeated, responses).float()  # (G,)

        # 步驟 3：群體相對優勢值 (Group-relative advantages)
        advantages = self.compute_group_advantages(rewards)

        # 步驟 4：計算舊對數機率（作為 PPO 比例的參考基準）
        with torch.no_grad():
            old_log_probs = compute_sequence_log_prob(self.model, responses, prompt_len)

        all_metrics = {}
        for epoch in range(n_epochs):
            self.model.train()
            loss, metrics = self.grpo_loss(
                prompt_repeated, responses, advantages, old_log_probs, prompt_len
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            all_metrics = metrics

        all_metrics["mean_reward"] = float(rewards.mean().item())
        all_metrics["reward_std"] = float(rewards.std().item())
        return all_metrics

    # --- 持久化 (Persistence) ---

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "ref_model": self.ref_model.state_dict(),
        }, os.path.join(path, "grpo_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "grpo_checkpoint.pt"), map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ref_model.load_state_dict(ckpt["ref_model"])
