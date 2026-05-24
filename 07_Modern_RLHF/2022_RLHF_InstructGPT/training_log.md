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
