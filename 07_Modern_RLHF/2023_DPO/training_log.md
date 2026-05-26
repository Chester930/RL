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
