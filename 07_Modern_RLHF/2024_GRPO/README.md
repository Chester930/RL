# GRPO — 群體相對策略最佳化 (Group Relative Policy Optimization)

## 論文

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, H.,
Li, Y. K., Wu, Y., & Guo, D. (2024).  
*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*.  
arXiv:2402.03300.

相關應用：  
DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*. arXiv:2501.12948.

---

## 核心思想 (Key Idea)

傳統的 PPO-RLHF 需要一個價值/評論家 (Value/Critic) 網路，其引數量通常與策略網路相同（造成極大的記憶體負擔）。
GRPO 的核心貢獻在於去除了價值網路，改用**群體相對獎勵 (Group-relative rewards)** 作為優勢值的基準 (Baseline)：

```
對於每個提示詞 x，取樣一組回應 (Group of G responses):
    y_1, ..., y_G ~ pi_theta(y|x)

使用獎勵函式為每個回應評分:
    r_1, ..., r_G = reward_fn(x, y_i)

計算群體相對優勢值 (Group-relative advantage):
    A_i = (r_i - mean(r)) / std(r)

更新損失函式: L = -E[ min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) ] + beta * KL(pi || pi_ref)
```

完全不需要價值函式 — 群體自身的平均獎勵就是最完美的基準線！

---

## 與 PPO-RLHF 對比 (vs PPO-RLHF)

| 特性 | PPO-RLHF | GRPO |
|----------|----------|------|
| **價值網路 (Value net)** | 需要（與策略網路同大小） | **不需要** |
| **記憶體消耗** | 約 4 倍模型大小 (Policy+Ref+RM+V) | **約 2 倍模型大小** (Policy+Ref) |
| **優勢值估計** | 透過學習到的 V 進行 GAE | **群體內相對正規化** |
| **KL 散度控制** | 通常採自適應 (Adaptive) | 固定 beta 或自適應 |
| **最佳用途** | 通用 RLHF (如對話助手) | **數學推理、程式碼、可驗證任務** |

---

## 獎勵函式設計 (Reward Function Design)

針對數學推理任務（如 DeepSeek-R1），GRPO 採用了純規則導向的獎勵：
```
格式獎勵 (Format reward): 若回應符合 <think>...</think><answer>A</answer> 格式則 +0.1
準確度獎勵 (Accuracy reward): 若最終答案正確（可透過真值驗證）則 +1.0
```

完全不需要獨立的獎勵模型 (RM) — 這避免了獎勵作弊 (Reward hacking) 並降低了訓練不穩定性。

---

## 群體大小 (Group Size)

```
典型的 G 取值在 4 到 16 之間：
- 較大的 G: 優勢值估計更精準穩定，但單次更新的計算量與視訊記憶體需求較大。
- 較小的 G: 計算速度快，但估計值的噪聲較大。
- DeepSeek-R1 在數學推理任務中設定 G = 8。
```

---

## 主要結果 (Key Results)

GRPO 結合「可驗證數學獎勵」與「思維鏈 (Chain-of-Thought)」提示，讓 **DeepSeek-R1** 在**完全不使用監督式推理資料 (SFT data)** 的情況下，透過純強化學習在 MATH、AIME 與 Codeforces 等基準測試中達到與 OpenAI o1 相當的世界頂尖水準。
