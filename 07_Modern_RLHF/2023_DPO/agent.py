"""
DPO — 直接偏好最佳化 (Direct Preference Optimization)。

省去了獨立的獎勵模型 (Reward Model) 訓練與 PPO 強化學習階段。
取而代之的是，利用獎勵與策略之間的解析解 (Closed-form) 對映關係，
直接從偏好資料中最佳化語言模型策略。

核心洞察：
    在 RLHF 目標下的最優策略具有以下解析解：
        pi*(y|x) ∝ pi_ref(y|x) * exp(r(x,y) / beta)

    將其求逆，獎勵可以表示為：
        r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + Z(x)

    代入 Bradley-Terry 偏好模型後，DPO 損失函式會自動消去
    難以處理的配分函式 (Partition function) Z(x)。

參考文獻：
    Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S.,
    & Finn, C. (2023). Direct Preference Optimization: Your Language Model
    is Secretly a Reward Model. NeurIPS 2023. arXiv:2305.18290.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple


class TinyLM(nn.Module):
    """用於展示 DPO 的極簡語言模型。"""

    def __init__(self, vocab_size: int = 1000, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, max_seq_len: int = 128):
        super().__init__()
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
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.lm_head(self.norm(x))  # 輸出形狀：(B, T, V)


def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor,
                      prompt_len: int) -> torch.Tensor:
    """
    僅計算回應 (Response) token 的對數機率總和。

    引數：
        logits:     (batch, seq_len, vocab_size)
        labels:     (batch, seq_len) token ID
        prompt_len: 要跳過的提示 (Prompt) token 數量（在損失計算中忽略）
    回傳：
        log_probs: (batch,) 回應 token 的對數機率總和
    """
    # 為「預測下一個 token」進行偏移 (Shift)
    shift_logits = logits[:, :-1, :]       # (B, T-1, V)
    shift_labels = labels[:, 1:]            # (B, T-1)

    log_p_all = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
    token_log_p = log_p_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # 掩蓋掉提示 (Prompt) token
    mask = torch.zeros_like(token_log_p)
    mask[:, prompt_len:] = 1.0

    return (token_log_p * mask).sum(dim=-1)  # (B,)


class DPOAgent:
    """
    直接偏好最佳化 (Direct Preference Optimization)。

    直接從偏好對（較佳, 較差）中訓練策略模型，無需獨立的獎勵模型。

    DPO 損失函式：
        L_DPO = -E[ log σ( beta * (log_ratio_chosen - log_ratio_rejected) ) ]

    其中：
        log_ratio = log pi_theta(y|x) - log pi_ref(y|x)

    這在數學上等同於最佳化 RLHF 目標，但不需要獎勵模型 (RM) 或 PPO。

    引數：
        vocab_size:   詞彙表大小。
        d_model:      模型隱藏層維度。
        n_layers:     Transformer 層數。
        n_heads:      注意力頭數量。
        max_seq_len:  最大序列長度。
        beta:         KL 正規化係數（溫度引數）。beta 越高代表越接近參考策略。
        lr:           學習率。
        label_smoothing: 偏好標籤平滑處理 (0 = 硬標籤，0.1 = 輕微平滑)。
        device:       "cpu" 或 "cuda"。
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 128,
        beta: float = 0.1,
        lr: float = 1e-5,
        label_smoothing: float = 0.0,
        device: str = "cpu",
    ):
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.device = device

        # 策略模型（可訓練）
        self.model = TinyLM(vocab_size, d_model, n_layers, n_heads, max_seq_len).to(device)
        # 參考模型（凍結的 SFT 模型）
        self.ref_model = TinyLM(vocab_size, d_model, n_layers, n_heads, max_seq_len).to(device)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0)

    def init_from_sft(self, sft_state_dict: dict) -> None:
        """從 SFT 模型權重初始化策略模型與參考模型。"""
        self.model.load_state_dict(sft_state_dict)
        self.ref_model.load_state_dict(sft_state_dict)
        print("DPO：策略模型與參考模型已從 SFT 權重初始化。")

    def _compute_log_ratio(
        self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor, prompt_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        為較佳 (Chosen) 與較差 (Rejected) 序列計算 log(pi/pi_ref)。

        回傳：
            pi_logprob_chosen:   (batch,)
            pi_logprob_rejected: (batch,)
            ref_logprob_chosen:  (batch,)
            ref_logprob_rejected:(batch,)
        """
        # 策略模型的 Logits
        pi_logits_chosen = self.model(chosen_ids)
        pi_logits_rejected = self.model(rejected_ids)

        # 參考模型 Logits（不計算梯度）
        with torch.no_grad():
            ref_logits_chosen = self.ref_model(chosen_ids)
            ref_logits_rejected = self.ref_model(rejected_ids)

        pi_lp_chosen = compute_log_probs(pi_logits_chosen, chosen_ids, prompt_len)
        pi_lp_rejected = compute_log_probs(pi_logits_rejected, rejected_ids, prompt_len)
        ref_lp_chosen = compute_log_probs(ref_logits_chosen, chosen_ids, prompt_len)
        ref_lp_rejected = compute_log_probs(ref_logits_rejected, rejected_ids, prompt_len)

        return pi_lp_chosen, pi_lp_rejected, ref_lp_chosen, ref_lp_rejected

    def dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        prompt_len: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        為一批偏好對 (Preference pairs) 計算 DPO 損失。

        L_DPO = -E[ log σ( beta * (log_ratio_w - log_ratio_l) ) ]

        其中 log_ratio = log pi_theta - log pi_ref

        引數：
            chosen_ids:   (batch, seq_len) 較佳（人類偏好）的回應 token。
            rejected_ids: (batch, seq_len) 較差（非人類偏好）的回應 token。
            prompt_len:   提示 token 數量（不計入對數機率總和計算）。
        回傳：
            loss:    DPO 損失標量。
            metrics: 包含獎勵值與準確率的字典。
        """
        pi_lp_w, pi_lp_l, ref_lp_w, ref_lp_l = self._compute_log_ratio(
            chosen_ids, rejected_ids, prompt_len
        )

        # 對數比例（即隱式獎勵 / Implicit rewards）
        log_ratio_w = pi_lp_w - ref_lp_w   # (batch,)
        log_ratio_l = pi_lp_l - ref_lp_l   # (batch,)

        # DPO loss
        logits = self.beta * (log_ratio_w - log_ratio_l)

        if self.label_smoothing > 0:
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # 指標 (Metrics)
        with torch.no_grad():
            reward_w = self.beta * log_ratio_w.mean()
            reward_l = self.beta * log_ratio_l.mean()
            reward_margin = reward_w - reward_l
            accuracy = (logits > 0).float().mean()  # 隱式獎勵與人類偏好一致的比例

        return loss, {
            "dpo_loss": float(loss.item()),
            "reward_chosen": float(reward_w.item()),
            "reward_rejected": float(reward_l.item()),
            "reward_margin": float(reward_margin.item()),
            "accuracy": float(accuracy.item()),
        }

    def update(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor,
               prompt_len: int = 0) -> dict:
        """一步 DPO 梯度更新。"""
        self.model.train()
        loss, metrics = self.dpo_loss(chosen_ids, rejected_ids, prompt_len)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return metrics

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "ref_model": self.ref_model.state_dict(),
        }, os.path.join(path, "dpo_checkpoint.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "dpo_checkpoint.pt"), map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ref_model.load_state_dict(ckpt["ref_model"])
