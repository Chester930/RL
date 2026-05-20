# HIRO | 離策略修正層次RL

> Data-Efficient Hierarchical Reinforcement Learning  
> Nachum et al. — NeurIPS 2018

---

## 核心思想

結合 Off-Policy 訓練與層次架構，解決兩層訓練資料不一致的問題：

```
High-level（每 c 步）：選擇子目標 g_t（在觀測空間）
Low-level （每步）  ：選擇原始動作，追求達到 s + g_t

問題：High-level 存舊 transition 時，low-level 策略已更新
解法：Off-Policy Correction
  → 重新取樣多個候選 g，挑選讓舊 low-level 行為機率最大的 g'
  → 用修正後的 g' 取代原始 g 來訓練 High-level
```

**優點**：比 FeUdal 更 sample-efficient，兩層都可以 off-policy 訓練。

---

## 環境 / 實驗設定

- 環境：MuJoCo（Ant Maze、Ant Push、Ant Fall）
- 特點：連續動作空間、超稀疏獎勵（只有到終點才給分）

---

## 檔案結構

```
2018_HIRO/
├── agent.py        # High-level + Low-level Agent（TD3 based）
├── train.py        # 訓練指令碼（off-policy correction）
└── training_log.md # 訓練日誌
```

---

## 論文

- Paper: https://arxiv.org/abs/1805.08296
