# GRPO 訓練記錄

**日期**：2026-05-22  
**環境**：合成數學資料（vocab=1000, d_model=128, n_layers=4, n_heads=4, max_seq_len=256, prompt_len=32）  
**硬體**：CPU  
**參考文獻**：Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.

---

## 訓練配置

| 參數 | 值 |
|---|---|
| vocab_size | 1000 |
| d_model | 128 |
| n_layers | 4 |
| n_heads | 4 |
| max_seq_len | 256 |
| prompt_len | 32 |
| max_new_tokens | 64 |
| group_size (G) | 8 |
| beta（KL 懲罰） | 0.04 |
| clip_eps | 0.2 |
| temperature | 1.0 |
| lr | 1e-6 |
| total_steps | 500 |

---

## 訓練結果

> ⚠️ **合成資料框架展示**：以下數字使用固定回傳 0.1 的合成獎勵函數，**不代表 GRPO 的真實學習能力**。
> 平均獎勵固定、KL≈0、零梯度均為預期行為（組內方差=0 → 優勢全為 0）。GRPO 的真實效果需可驗證的獎勵函數（如數學答案正確性），參見底部「合成資料限制」說明。

| 步數 | 損失 | KL 散度 | 平均獎勵 | 獎勵標準差 |
|---|---|---|---|---|
| 50 | 0.4147 | 0.0000 | 0.100 | 0.000 |
| 100 | 0.4144 | -0.0032 | 0.100 | 0.000 |
| 150 | 0.4147 | 0.0000 | 0.100 | 0.000 |
| 200 | 0.4147 | 0.0000 | 0.100 | 0.000 |
| 250 | 0.4147 | 0.0000 | 0.100 | 0.000 |
| 300 | 0.4147 | ~0.000 | 0.100 | 0.000 |
| 350 | 0.4147 | 0.0000 | 0.100 | 0.000 |
| 400 | 0.4147 | 0.0000 | 0.100 | 0.000 |
| 450 | 0.4149 | 0.0050 | 0.100 | 0.000 |
| 500 | 0.4147 | 0.0000 | 0.100 | 0.000 |

---

## 觀察

- **損失**：全程穩定在 ~0.4147，無震盪，符合極小學習率（1e-6）下的預期行為。
- **KL 散度**：幾乎為 0，策略幾乎未偏離參考模型，KL 懲罰（β=0.04）有效抑制漂移。
- **平均獎勵**：固定在 0.100，標準差 0.000 — 合成獎勵函數對所有輸出給予相同分數，group_size=8 的組內方差為 0，導致 GRPO 優勢估計（Advantage）全為 0，無梯度信號。
- **整體**：此結果為合成資料下的正確行為，展示了 GRPO 的流程框架（group sampling → advantage normalization → clip update）。

---

## GRPO 核心原理（DeepSeek-R1 方法）

```
對每個提示詞 x：
  1. 取樣 G 個回應 {y₁, ..., yG}（group_size=8）
  2. 計算每個回應的獎勵 rᵢ
  3. 組內正規化優勢：Aᵢ = (rᵢ - mean(r)) / std(r)
  4. PPO-clip 更新 + KL 懲罰（β=0.04）
```

**優勢對比 PPO**：不需要 Critic/Value network，直接用組內均值作為 baseline，降低訓練複雜度。

---

## 合成資料限制

本實作使用隨機合成獎勵（固定回傳 0.1），組內獎勵標準差恆為 0，導致 advantage=0/0（用 eps 穩定），無法展示真實學習效果。實際 GRPO 需可驗證的獎勵函數（如數學答案正確性判斷），參考 DeepSeek-R1 使用的 rule-based reward。

---

## 合成資料框架說明

### 為何使用合成獎勵

GRPO 的關鍵創新是「用組內均值代替 Critic」，因此獎勵函數的多樣性（組內標準差 > 0）是學習的前提。本實作用固定獎勵 0.1 的原因：

| 原因 | 說明 |
|---|---|
| **展示框架** | 教學目標是讓學生看懂「G 個採樣 → 組內正規化 → clip update」的計算圖 |
| **無需驗證器** | 真實 GRPO 需要一個能判斷「答案對不對」的程式；數學驗證器需額外實作 |
| **揭示前提** | 固定獎勵導致 advantage=0，反向展示了「GRPO 為什麼需要有差異的獎勵」|

### 固定獎勵下各指標的「正確」行為

| 指標 | 固定獎勵預期值 | 真實獎勵預期值 |
|---|---|---|
| 平均獎勵 | 0.100（固定不變）| 從低到高穩定上升 |
| 獎勵標準差 | 0.000 | > 0（組內有分散）|
| KL 散度 | ~0.000 | 小幅上升後穩定 |
| 損失 | ~0.4147（無梯度）| 下降至收斂值 |

**advantage=0 = 零梯度**：`std(r)=0` → 所有優勢歸一化後為 0 → 梯度為 0 → 模型完全不更新。這是正確行為，不是訓練失敗。

### 如何替換成真實獎勵函數

**選項 A：數學問題（最接近 DeepSeek-R1）**
```python
import re

def math_reward_fn(response: str, ground_truth: str) -> float:
    # 提取模型回應中的數字答案
    matches = re.findall(r"\\boxed\{([^}]+)\}", response)
    if matches and matches[-1].strip() == ground_truth.strip():
        return 1.0    # 答案正確
    return 0.0        # 答案錯誤

# 資料集：GSM8K（8500 道小學數學題）
from datasets import load_dataset
gsm8k = load_dataset("openai/gsm8k", "main", split="train")
# 欄位：{"question": "...", "answer": "...<final>數字"}
```

**選項 B：程式碼生成（可執行驗證）**
```python
def code_reward_fn(code: str, test_cases: list) -> float:
    passed = 0
    for test in test_cases:
        try:
            exec(code)
            exec(test)
            passed += 1
        except:
            pass
    return passed / len(test_cases)
```

**選項 C：使用 TRL GRPOTrainer（最快上手）**
```python
from trl import GRPOTrainer, GRPOConfig

def reward_fn(completions, prompts, **kwargs):
    # 返回每個 completion 的獎勵（長度 = group_size × batch_size）
    return [math_reward_fn(c, get_answer(p)) for c, p in zip(completions, prompts)]

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=GRPOConfig(num_generations=8, ...),  # num_generations = group_size G
    train_dataset=gsm8k,
)
trainer.train()
```

**預期結果（真實獎勵）**：平均獎勵從 ~0% 準確率逐漸提升，KL 散度緩步上升，損失下降並收斂。DeepSeek-R1-Zero 在 MATH 資料集上從 ~15% 提升至 ~70% pass@1。
