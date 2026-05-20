# PPO-Lagrangian | 拉格朗日安全 PPO

> Benchmarking Safe Exploration in Deep Reinforcement Learning  
> Ray, Achiam, Amodei — Safety Gym 2019

---

## 核心思想

用**拉格朗日乘數法**把安全約束轉換成軟約束，嵌入 PPO 的目標函式：

```
原始問題（硬約束）：
  max E[Σ r_t]  s.t.  E[Σ c_t] ≤ d

轉換（拉格朗日鬆弛）：
  L(π, λ) = E[Σ r_t] - λ × (E[Σ c_t] - d)

  λ（拉格朗日乘數）自動調整：
    違規太多 → λ ↑ → 安全代價懲罰加重 → 策略更保守
    違規太少 → λ ↓ → 懲罰減弱 → 策略更積極
```

**優點**：比 CPO 實作簡單，可以直接套在 PPO 上，訓練穩定。  
**缺點**：不像 CPO 有嚴格的每步約束保證。

---

## 環境 / 實驗設定

- 環境：Safety Gym（同 CPO）
- 與 CPO 使用相同基準，方便對比

---

## 檔案結構

```
2019_PPO_Lagrangian/
├── agent.py        # PPO + 自動調整 λ
├── train.py        # 訓練指令碼
└── training_log.md # 訓練日誌
```

---

## 論文

- Paper: https://arxiv.org/abs/1910.12156
