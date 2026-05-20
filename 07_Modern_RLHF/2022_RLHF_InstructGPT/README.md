# RLHF — 來自人類回饋的強化學習 (Reinforcement Learning from Human Feedback, InstructGPT)

## 論文

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C.,
Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L.,
Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022).  
*Training language models to follow instructions with human feedback*.  
NeurIPS 2022. arXiv:2203.02155.

---

## 三階段流程 (Three-Phase Pipeline)

```
第一階段：監督式微調 (Supervised Fine-Tuning, SFT)
    資料集：人類編寫的演示資料 (Demonstrations)
    方法：  標準語言模型微調（交叉熵損失）
    輸出：  SFT 模型 π_SFT

第二階段：訓練獎勵模型 (Reward Model, RM)
    資料集：人類偏好對比資料 (Chosen > Rejected)
    方法：  Bradley-Terry 排序損失
             L = -E[ log σ(r_chosen - r_rejected) ]
    輸出：  獎勵模型 r_θ

第三階段：PPO-RLHF 強化學習最佳化
    目標：  極大化 r_θ(prompt, response) - kl_coef * KL(π || π_SFT)
    方法：  對比參考模型 (Reference model) 並加入 KL 懲罰項的 PPO 演演算法
    輸出：  InstructGPT 策略 (Policy)
```

---

## KL 懲罰項 (KL Penalty)

```
獎勵值 = r_RM(x, y) - beta * KL(pi_RL(y|x) || pi_SFT(y|x))

原因：防止策略 (Policy) 過度擬合獎勵模型（稱為獎勵作弊 / Reward Hacking）。
      確保生成的內容不會偏離參考模型 (SFT) 的分佈太遠。
beta 值通常在 0.01 到 0.1 之間（InstructGPT 使用自適應調整）。
```

---

## 獎勵模型損失 (Reward Model Loss)

```
L_RM = -E_{(x,y_w,y_l)~D_pref} [ log σ(r_θ(x, y_w) - r_θ(x, y_l)) ]

y_w = 人類偏好中「較佳 (Chosen)」的回應
y_l = 人類偏好中「較差 (Rejected)」的回應
```

---

## 生產環境實作 (Production Implementation)

在實際大規模訓練大型語言模型 (LLM) 時，建議使用以下工具：
```bash
pip install trl transformers accelerate
# 使用 trl 提供的：PPOTrainer, RewardTrainer, SFTTrainer
```

本程式碼架構僅用於展示 RLHF 的核心演演算法邏輯。實際生產規模需要考慮分散式訓練、批次生成最佳化以及正確的分詞 (Tokenization) 流程。
