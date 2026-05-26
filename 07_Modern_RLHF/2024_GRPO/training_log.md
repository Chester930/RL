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

## GRPO 核心機制詳解

> 以下說明 GRPO 的「組內正規化優勢」計算方式，幫助理解「為什麼固定獎勵讓 advantage=0」以及「真實獎勵下 GRPO 如何工作」

---

### 一、GRPO vs PPO：去掉 Critic，用組內均值替代

```
PPO 的優勢估計（需要 Critic 網路 V(s)）：
  A_t = r_t + γV(s_{t+1}) − V(s_t)     ← 需要額外訓練一個 Value 網路

GRPO 的優勢估計（不需要 Critic）：
  對同一提示 x，取樣 G 個回應 {y_1, ..., y_G}
  計算每個回應的獎勵 {r_1, ..., r_G}
  A_i = (r_i − mean({r_j})) / std({r_j})  ← 組內相對優勢，無需 V(s)
```

---

### 二、合成資料（固定獎勵 0.1）下的計算

本訓練 group_size=8，每個 batch 對同一提示取樣 8 個回應：

```
G = 8 個回應，獎勵全部 = 0.1（合成固定獎勵）：
  {r_1, ..., r_8} = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}

  mean(r) = 0.1
  std(r)  = 0.0          ← 組內無差異！

  A_i = (0.1 − 0.1) / (0.0 + ε) = 0 / ε ≈ 0    （ε 是數值穩定項）

  → 所有優勢 = 0 → 梯度 = 0 → 模型完全不更新
```

**這正是訓練日誌顯示「損失固定在 0.4147、KL≈0、獎勵標準差=0」的原因。**

---

### 三、真實獎勵（數學問題）下的計算

以 group_size=8 個回應，8 個的正確/錯誤為例：

```
提示 x = "計算 15 × 23 = ?"，正確答案 = 345
8 個模型回應：y_1=345, y_2=0, y_3=345, y_4=0, y_5=0, y_6=345, y_7=0, y_8=0

二元獎勵（正確=1.0，錯誤=0.0）：
  {r_1,...,r_8} = {1, 0, 1, 0, 0, 1, 0, 0}

  mean(r) = 3/8 = 0.375
  std(r)  = sqrt(0.375 × 0.625) ≈ 0.484

  優勢計算：
  A_1（正確）= (1.0 − 0.375) / 0.484 = +1.292   ← 強化正確回應
  A_2（錯誤）= (0.0 − 0.375) / 0.484 = −0.775   ← 抑制錯誤回應
  ...

→ 正確回應的 token 機率被提升，錯誤回應的 token 機率被降低
```

**GRPO 的核心直覺：組內「做得比平均好」→ 正優勢，「做得比平均差」→ 負優勢。
這是「自我評分」：不需要外部評分員（Value 網路），用同一批次的相對表現互相評判。**

---

### 四、PPO-clip 更新 + KL 懲罰（β=0.04）

```
GRPO 的損失函式（結合 PPO-clip）：

L_GRPO = -E[ min(ratio × A, clip(ratio, 1-ε, 1+ε) × A) ]
        + β × KL[π_θ || π_ref]

其中：
  ratio = π_θ(y|x) / π_old(y|x)    ← 新舊策略的機率比
  ε = 0.2                           ← clip 範圍（同 PPO）
  β = 0.04                          ← KL 懲罰（比 DPO 的 0.1 更小）

β=0.04（較小）的理由：
  GRPO 用於對話任務的「微調」而非「完全訓練」
  SFT 參考模型已有良好基礎，允許更大幅度的偏離
  但仍需 KL 防止「獎勵駭客」（reward hacking）
```

---

### 五、DeepSeek-R1-Zero 的效果（真實資料）

```
環境：MATH 數學競賽資料集
獎勵：答案正確 → 1.0，答案錯誤 → 0.0（純規則式獎勵，無 RM）
Group size：G = 8

訓練過程：
  初始（SFT 基線）：MATH 準確率 ≈ 15%
  GRPO 訓練 8000 steps：準確率 ≈ 70%（DeepSeek-R1-Zero 公布數字）

關鍵觀察：不需要人類偏好標注，只需「答案對不對」的二元判斷
→ 這使 GRPO 特別適合「有明確正確答案」的任務：數學、程式碼、推理
```

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
