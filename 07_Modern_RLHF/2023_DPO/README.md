# DPO — 直接偏好最佳化 (Direct Preference Optimization)

## 論文

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023).  
*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.  
NeurIPS 2023. arXiv:2305.18290.

---

## 核心思想 (Key Idea)

傳統的 RLHF 流程包含三個步驟：SFT -> RM (獎勵模型訓練) -> PPO (強化學習最佳化)。
DPO 的核心貢獻在於將獎勵模型訓練與 PPO 最佳化**合併為單一的監督式損失 (Supervised loss)**，直接在偏好資料上進行訓練，無需顯式的獎勵模型或複雜的強化學習過程。

```
最優 RLHF 策略具有以下解析解 (Closed form)：
    pi*(y|x) ∝ pi_ref(y|x) * exp(r(x,y) / beta)

重新排列可得： r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + Z(x)

將此獎勵項代入 Bradley-Terry 偏好模型：
    p(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
                    = σ(beta * [log(pi(y_w|x)/pi_ref(y_w|x))
                               - log(pi(y_l|x)/pi_ref(y_l|x))])

DPO 損失函式： L = -E[ log σ(beta * (log_ratio_w - log_ratio_l)) ]
```

由於 Z(x) 在計算中會相互抵消，因此**不需要額外訓練獎勵模型**，也不需要計算難以處理的配分函式 (Partition function)。

---

## 與 RLHF 對比 (Comparison with RLHF)

| 特性 | RLHF (基於 PPO) | DPO (直接最佳化) |
|----------|-----------|-----|
| **獎勵模型** | 顯式 (Explicit) 訓練 | 隱式 (Implicit) 存在 |
| **線上取樣** | 需要（計算複雜且不穩定） | **不需要**（離線監督訓練） |
| **實作複雜度** | 高 | **低** |
| **記憶體消耗** | 約 4 倍模型大小 | **約 2 倍模型大小** (策略 + 參考) |
| **穩定性** | 對 KL 係數與超引數極度敏感 | **通常非常穩定** |
| **效能表現** | 目前的最頂尖技術 (SOTA) | **極具競爭力且更易落地** |

---

## Beta 引數 (Beta Parameter)

| Beta 取值 | 效果與特性 | 穩定性 |
| :---: | :--- | :---: |
| **β → 0** | 忽略參考模型，純粹極大化人類偏好（容易過擬合） | 最差 |
| **β ≈ 0.1** | **典型取值**：在偏好滿足與穩定性之間取得最佳平衡 | 優 |
| **β → ∞** | 強制保持接近參考模型，效果等同於行為複製 (BC) | 最高 |


---

## 生產環境使用 (Production Usage)

```bash
pip install trl transformers
# 在實際大規模 LLM 微調中，請使用 trl 的 DPOTrainer
from trl import DPOTrainer, DPOConfig
```
