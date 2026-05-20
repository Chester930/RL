# 10 Safe RL | 安全強化學習

在最大化獎勵的同時，確保 Agent 不會違反安全約束條件（不踩紅線）。

---

## 核心概念

```
普通 RL：最大化 E[Σ r_t]
Safe RL：最大化 E[Σ r_t]，同時滿足 E[Σ c_t] ≤ d

  r_t：獎勵（越高越好）
  c_t：代價 / 違規次數（成本，必須控制在 d 以內）
  d  ：安全預算（最多可以違規幾次）
```

這類問題正式稱為 **Constrained MDP（CMDP）**。

---

## 演演算法列表

| 目錄 | 演演算法 | 中文名 | 論文 |
|------|--------|--------|------|
| `2017_CPO` | Constrained Policy Optimization | 約束策略最佳化 | Achiam et al. 2017 |
| `2019_PPO_Lagrangian` | Lagrangian PPO | 拉格朗日安全PPO | Ray et al. 2019 |

---

## 學習路徑

**前置知識**：Policy Gradient（03）、PPO（03/04）

**建議順序**：CPO → PPO_Lagrangian

---

## 為什麼重要

| 應用場景 | 如果沒有 Safe RL |
|:---|:---|
| 自動駕駛 | 撞車也算「探索」 |
| 醫療機器人 | 危險操作也可能換來高分 |
| 工廠機械手臂 | 破壞裝置只要完成任務就行 |
| 金融交易 | 孤注一擲博高報酬 |

Safe RL 是 RL 從實驗室走向現實產品的關鍵一步。
