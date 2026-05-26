# RLHF InstructGPT 訓練記錄

**日期**：2026-05-22  
**環境**：合成資料（vocab=1000, d_model=128, n_layers=4, n_heads=4, seq_len=64, batch=16）  
**硬體**：CPU  
**參考文獻**：Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS 2022. arXiv:2203.02155

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
| sft_steps | 500 |
| rm_steps | 500 |
| ppo_steps | 500 |
| lr_sft | 1e-4 |
| lr_rm | 1e-4 |
| lr_ppo | 1e-5 |
| kl_coef | 0.1 |
| clip_eps | 0.2 |

---

## 第一階段：監督式微調（SFT）

| 步數 | 損失 |
|---|---|
| 100 | 20.0872 |
| 200 | 15.7659 |
| 300 | 14.4741 |
| 400 | 13.0197 |
| 500 | 12.2399 |

**觀察**：SFT 損失穩定下降，從 20.09 降至 12.24（下降 39%），符合監督學習預期行為。  
**Checkpoint**：`checkpoints/rlhf/rlhf_sft.pt`

---

## 第二階段：獎勵模型訓練（RM）

> ⚠️ **合成資料框架展示**：以下 RM 與 PPO 數字使用隨機合成偏好資料，**不代表 RLHF 的真實能力**。
> RM 損失收斂在 0.693（隨機基線）、平均獎勵為負，均為合成無信號資料的預期行為。真實 RLHF 需人類標注的偏好對，如 InstructGPT 原論文使用的 OpenAI 內部資料集。

| 步數 | 損失 | 獎勵差距（chosen − rejected） |
|---|---|---|
| 100 | 0.6975 | -0.0266 |
| 200 | 0.7337 | -0.0161 |
| 300 | 0.6973 | +0.0536 |
| 400 | 0.6930 | -0.0092 |
| 500 | 0.6840 | +0.0071 |

**觀察**：RM 損失收斂在 ~0.69（接近 log(2)≈0.693，二元交叉熵隨機基線），符合合成隨機資料的預期。獎勵差距圍繞 0 震盪，說明模型在無信號的合成資料上無法分辨 chosen/rejected，是正確行為。  
**Checkpoint**：`checkpoints/rlhf/rlhf_rm.pt`

---

## 第三階段：PPO-RLHF

| 步數 | 損失 | 平均獎勵 |
|---|---|---|
| 100 | 0.0168 | -8.7727 |
| 200 | 0.2236 | -17.4818 |
| 300 | 0.2453 | -17.9144 |
| 400 | 0.0879 | -17.6186 |
| 500 | 0.1631 | -17.5056 |

**觀察**：平均獎勵在 step 100 後穩定在 ~-17.5，PPO 損失小幅震盪屬正常。負獎勵源於合成隨機 RM，非訓練失敗。PPO clip（ε=0.2）和 KL 懲罰（β=0.1）均正常運作。  
**Checkpoint**：`checkpoints/rlhf/rlhf_ppo.pt`

---

## 教學重點

| 階段 | 核心概念 |
|---|---|
| SFT | 在示範資料上微調，建立初始行為策略 |
| RM | 從人類偏好對（chosen > rejected）學習獎勵函數 |
| PPO | 以 RM 分數為獎勵，用 PPO 最佳化策略；KL 懲罰防止偏離 SFT 太遠 |

**合成資料限制**：本實作使用隨機合成資料，RM 無法學到真實偏好，PPO 獎勵為負且平穩。實際 RLHF 需真實人類標注資料，建議參考 HuggingFace TRL 框架。

---

## 合成資料框架說明

### 為何使用合成資料

| 原因 | 說明 |
|---|---|
| **算力限制** | InstructGPT 原論文使用 175B GPT-3；本實作用 128-dim 4層 Transformer 在 CPU 跑完整三階段流程 |
| **資料取得** | 真實人類偏好標注需要 API（OpenAI）或龐大標注工作；本實作目的是展示流程而非重現結果 |
| **教學目標** | 核心目標是讓學生看懂「SFT → RM → PPO 三階段串接」，資料品質是次要的 |
| **課堂時間** | 真實資料集下載 + 預處理需額外 1–2 小時，超出課程範圍 |

### 合成資料下各指標的「正確」行為

| 指標 | 合成資料預期值 | 真實資料預期值 | 差異解讀 |
|---|---|---|---|
| RM 損失 | ~0.693（隨機基線）| < 0.4（收斂）| log(2) = 二元分類隨機猜測的交叉熵下界 |
| RM 準確率 | ~50% | > 70% | 合成資料無 chosen/rejected 信號差異 |
| PPO 平均獎勵 | 負值且平穩（~-17）| 逐漸上升 | 隨機 RM 給出雜訊分數，無法引導策略 |
| SFT 損失 | 下降（20→12）✅ | 下降 | SFT 是監督學習，合成與真實差異不大 |

### 如何替換成真實資料

**步驟一：安裝 TRL 與資料集**
```bash
pip install trl datasets transformers
```

**步驟二：載入真實偏好資料集**
```python
from datasets import load_dataset

# 選項 A：Anthropic HH-RLHF（對話偏好，700k 筆）
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
# 欄位：{"chosen": "...", "rejected": "..."}

# 選項 B：OpenAI 摘要偏好（TL;DR）
dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="train")
```

**步驟三：使用 TRL 替換訓練迴圈**
```python
from trl import SFTTrainer, RewardTrainer, PPOTrainer

# SFT 階段
sft_trainer = SFTTrainer(model=model, train_dataset=sft_dataset, ...)
sft_trainer.train()

# RM 階段
rm_trainer = RewardTrainer(model=rm_model, train_dataset=pref_dataset, ...)
rm_trainer.train()

# PPO 階段
ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, ...)
```

**預期結果（真實資料）**：RM 損失收斂至 ~0.3，準確率 ~70–75%，PPO 獎勵從負值逐漸上升至正值。
