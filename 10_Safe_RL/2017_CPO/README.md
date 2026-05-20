# CPO | 約束策略最佳化

> Constrained Policy Optimization  
> Achiam, Held, Tamar, Abbeel — ICML 2017

---

## 核心思想

在 TRPO 的基礎上加入約束條件，保證每次策略更新**不超過安全預算 d**：

```
最大化：E[Σ r_t]          ← 一般獎勵（越高越好）
約束：  E[Σ c_t] ≤ d      ← 安全代價（不能超過 d）
        D_KL(π_old ‖ π) ≤ δ ← 每步策略更新幅度限制（繼承自 TRPO）

使用二階近似（類 TRPO）求解帶約束的最佳化問題
```

**保證**：在一定條件下，每次策略更新都會滿足約束，而不只是訓練完才滿足。

---

## 環境 / 實驗設定

- 環境：Safety Gym（Point、Car、Doggo）
- 代價函式：靠近危險區域 / 碰撞障礙物 → c_t = 1

---

## 檔案結構

```
2017_CPO/
├── agent.py        # CPO Agent（constrained policy update）
├── train.py        # 訓練指令碼
└── training_log.md # 訓練日誌
```

---

## 論文

- Paper: https://arxiv.org/abs/1705.10528
