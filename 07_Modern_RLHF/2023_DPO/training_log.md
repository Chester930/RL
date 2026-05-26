# DPO 訓練記錄

**日期**：2026-05-22  
**環境**：合成資料（vocab=1000, d_model=128, n_layers=4, n_heads=4, seq_len=64, batch=16）  
**硬體**：CPU  
**參考文獻**：Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.

---

## 訓練配置

| 參數 | 值 |
|---|---|
| vocab_size | 1000 |
| d_model | 128 |
| n_layers | 4 |
| n_heads | 4 |
| max_seq_len | 64 |
| batch_size | 16 |
| total_steps | 1000 |
| beta（KL 正規化） | 0.1 |
| lr | 1e-5 |
| label_smoothing | 0.0 |

---

## 訓練結果

> ⚠️ **合成資料框架展示**：以下數字使用隨機合成偏好對，**不代表 DPO 的真實學習能力**。
> 準確率震盪於 50% 為預期行為（無真實偏好信號）。DPO 的真實效果需真實人類標注資料，參見底部「合成資料限制」說明。

| 步數 | 損失 | 獎勵差距 | 準確率 |
|---|---|---|---|
| 100 | 3.3353 | +1.9183 | 56.25% |
| 200 | 6.9600 | +0.6436 | 50.00% |
| 300 | 4.2311 | +4.6332 | 62.50% |
| 400 | 2.0963 | +7.2929 | 68.75% |
| 500 | 6.1761 | -0.7977 | 37.50% |
| 600 | 5.1228 | +1.2303 | 50.00% |
| 700 | 3.1404 | +5.5979 | 62.50% |
| 800 | 5.8942 | -1.2085 | 37.50% |
| 900 | 4.2567 | +3.3226 | 50.00% |
| 1000 | 6.2731 | -0.2388 | 31.25% |

**Checkpoint**：`checkpoints/dpo_step500`、`checkpoints/dpo_step1000`

---

## 觀察

- **損失**：在 2.1–6.9 之間震盪，無穩定收斂趨勢，符合合成隨機資料的預期（chosen/rejected 無真實信號差異）。
- **獎勵差距**：在 -1.2 至 +7.3 之間劇烈波動，整體均值接近 0，顯示模型無法從隨機資料中學習偏好排序。
- **準確率**：在 31.25%–68.75% 之間震盪，平均約 50%，即隨機猜測水準，符合合成資料的無信號情境。

---

## DPO 核心機制詳解

> 以下說明 DPO 損失函式的推導過程，幫助理解「為什麼 DPO 不需要訓練獨立的 Reward Model」

---

### 一、從 RLHF 目標到 DPO 閉式解

**RLHF 的優化目標（KL 約束的 RL）：**

```
max_π  E_{x~D, y~π}[r(x, y)] − β · KL[π(y|x) || π_ref(y|x)]

其中：
  r(x, y)     ← 獨立訓練的 Reward Model（RLHF 的額外成本）
  π_ref       ← SFT 參考模型（凍結不訓練）
  β           ← KL 懲罰強度（本訓練：β=0.1）
```

**DPO 的關鍵洞察：上述問題有閉式最優解：**

```
π*(y|x) = π_ref(y|x) × exp(r(x,y)/β) / Z(x)

其中 Z(x) 是配分函數（歸一化常數）

反推：r(x,y) = β × log(π*(y|x)/π_ref(y|x)) + β × log Z(x)
```

**帶入 Bradley-Terry 偏好模型（chosen y_w 優於 rejected y_l）：**

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l))

注意：log Z(x) 在相減時消去！

→ P(y_w ≻ y_l | x) = σ(β·log(π*(y_w|x)/π_ref(y_w|x)) − β·log(π*(y_l|x)/π_ref(y_l|x)))
```

**DPO 損失函式（最大化偏好可能性）：**

```
L_DPO(θ) = −E_{(x,y_w,y_l)~D}[
    log σ( β × log(π_θ(y_w|x)/π_ref(y_w|x))
         − β × log(π_θ(y_l|x)/π_ref(y_l|x)) )
]

簡化記法（以 logits 差值 Δ 表示）：
  Δ = β × (log π_θ(y_w|x) − log π_ref(y_w|x))
    − β × (log π_θ(y_l|x) − log π_ref(y_l|x))

  L_DPO = −log σ(Δ)   ← 交叉熵損失，Δ 越大損失越小
```

**關鍵：r(x,y) 從未被顯式計算——DPO 直接從語言模型的 log 機率差值推算隱式獎勵。**

---

### 二、beta=0.1 的意義（KL 懲罰強度）

```
β 控制「學多遠離 SFT 參考模型」的速度：

β 大（如 0.5）：KL 懲罰重 → 策略緊貼 π_ref → 偏好學習慢但穩定
β 小（如 0.01）：KL 懲罰輕 → 策略可大幅偏離 π_ref → 偏好學習快但可能失控

本訓練 β=0.1（DPO 論文預設值）：適中的 KL 約束
```

**隱式獎勵差值計算（一對偏好樣本）：**

```
假設訓練後某步的 log π_θ 與 log π_ref：
  y_w（chosen）：log π_θ = -2.3，log π_ref = -2.8  → chosen 被強化 +0.5
  y_l（rejected）：log π_θ = -3.1，log π_ref = -2.6  → rejected 被弱化 -0.5

Δ = β × ((−2.3) − (−2.8)) − β × ((−3.1) − (−2.6))
  = 0.1 × 0.5 − 0.1 × (−0.5)
  = 0.05 + 0.05 = 0.10

loss = −log σ(0.10) = −log(0.525) ≈ 0.644

若 Δ→+∞（模型完全區分偏好）：loss → −log(1) = 0
若 Δ=0（無法區分）：loss = −log(0.5) ≈ 0.693（ln2）
```

**本訓練損失 2.1–6.9（遠高於 0.693）的原因：合成資料 y_w/y_l 隨機生成，無真實偏好信號，Δ 圍繞 0 震盪 → 損失在 ln2 附近隨機波動。**

---

### 三、訓練指標解讀

| 指標 | 合成資料行為 | 真實資料期望行為 |
|---|---|---|
| 損失 | 2.1–6.9（高，不收斂）| 穩定下降至 ~0.5–1.5 |
| 獎勵差距（chosen−rejected）| 圍繞 0 波動 | 持續上升至 >1.0 |
| 準確率 | ~50%（隨機猜測）| 收斂至 65–80% |

**獎勵差距（reward margin）是 DPO 最重要的監控指標：**

```
reward_w = β × (log π_θ(y_w|x) − log π_ref(y_w|x))   ← chosen 的隱式獎勵
reward_l = β × (log π_θ(y_l|x) − log π_ref(y_l|x))   ← rejected 的隱式獎勵
margin   = reward_w − reward_l                          ← 偏好差距

真實訓練中 margin 持續上升 → 模型越來越能區分好壞回答
合成訓練中 margin ≈ 0 → 沒有信號可學
```

---

## DPO vs RLHF 教學對比

| 面向 | RLHF (PPO) | DPO |
|---|---|---|
| 是否需要獨立 RM | ✅ 需要 | ❌ 不需要 |
| 訓練複雜度 | 三階段（SFT→RM→PPO）| 單階段（直接優化） |
| 穩定性 | PPO 超參敏感 | 相對穩定 |
| 核心公式 | RM 分數 + KL 懲罰 | closed-form Bradley-Terry |

**合成資料限制**：本實作使用隨機資料，無法展示 DPO 的真實收斂效果。生產環境需真實人類偏好對標注，可參考 HuggingFace TRL 的 `DPOTrainer`。

---

## 合成資料框架說明

### 為何使用合成資料

DPO 相較 RLHF 更易實作（無需 RM + PPO），但仍需真實偏好對資料。本實作用合成資料的原因：

| 原因 | 說明 |
|---|---|
| **教學定位** | 展示 DPO 損失函式（Bradley-Terry + KL）的計算流程，而非重現論文結果 |
| **無需外部依賴** | 真實偏好資料集需 HuggingFace 帳號或網路存取；合成資料可離線執行 |
| **對比 RLHF** | 重點是讓學生比較「DPO 無需訓練 RM」vs「RLHF 三階段」的架構差異 |

### 合成資料下各指標的「正確」行為

| 指標 | 合成資料預期值 | 真實資料預期值 |
|---|---|---|
| 損失 | 2–7 之間震盪，不收斂 | 穩定下降至 ~0.5–1.5 |
| 獎勵差距（chosen−rejected）| 圍繞 0 震盪 | 穩定上升至正值（>1.0）|
| 準確率 | ~50%（隨機猜測）| 收斂至 65–80% |

**準確率 50% ≠ 訓練失敗**：合成的 chosen/rejected 對是隨機生成的，無真實「哪個更好」的信號，模型正確地學到「無從分辨」。

### 如何替換成真實資料

**步驟一：載入偏好資料集**
```python
from datasets import load_dataset

# 選項 A：Anthropic HH-RLHF（對話，~700k 筆）
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
# 欄位：{"chosen": "Human: ...\nAssistant: ...", "rejected": "..."}

# 選項 B：Stanford SHP（StackExchange 問答偏好，~385k 筆）
dataset = load_dataset("stanfordnlp/SHP", split="train")
# 欄位：{"history": "...", "human_ref_A": "...", "human_ref_B": "...", "labels": 0/1}
```

**步驟二：使用 TRL DPOTrainer**
```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(beta=0.1, max_length=512, ...)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,       # SFT 參考模型（凍結）
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

**預期結果（真實資料）**：損失穩定下降，準確率收斂至 65–75%，chosen 獎勵持續高於 rejected 獎勵。

### DPO vs RLHF 在真實資料下的核心差距

- **DPO 優勢**：單階段訓練，無需維護 RM 和 PPO 的超參
- **DPO 限制**：需要靜態偏好對資料集；無法像 RLHF 那樣動態生成新的偏好回饋
- **實務選擇**：資料充足 → DPO；需要持續收集人類回饋 → RLHF
