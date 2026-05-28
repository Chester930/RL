# 01 — 傳統 RL 用於機器人（2015–2018）

> **時代定位**：第一個讓連續動作控制真正可行的時代  
> **核心突破**：從像素/感測器直接學習關節控制，無需手工設計控制器

---

## 為什麼這個時代重要

在 DDPG 之前，機器人控制主要依賴：
- 手工設計的 PID 控制器（需要大量工程經驗）
- 軌跡最佳化（需要精確的動態模型）
- 傳統 RL 只能處理離散動作（無法直接控制關節角度）

**DDPG 的出現讓「深度 RL + 連續動作控制」成為可能。**

---

## 本章節演算法

### 2015_DDPG — Deep Deterministic Policy Gradient
```
論文：Lillicrap et al. (2015), arXiv:1509.02971
機構：Google DeepMind

核心創新：
  - 確定性策略梯度（DPG）+ 深度神經網路
  - Experience Replay Buffer（借鑒 DQN）
  - Target Network（穩定訓練）
  - Exploration Noise（Ornstein-Uhlenbeck 或 Gaussian）

架構：
  Actor: s → a（確定性，輸出連續動作）
  Critic: (s,a) → Q（評估動作價值）

機器人應用：
  - 關節角度控制（Reacher, Ant, HalfCheetah）
  - 末端執行器軌跡控制
  - 本專案結果：Pendulum best -86.1
```

### 2018_TD3 — Twin Delayed Deep Deterministic
```
論文：Fujimoto et al. (2018), arXiv:1802.09477
機構：McGill University

解決 DDPG 的三個問題：
  1. 過估計偏差 → 雙 Critic 取最小值
  2. 策略更新過激 → 延遲 Actor 更新（每 2 步 Critic 更新 1 步 Actor）
  3. Q 估計高方差 → 目標動作加高斯噪音（平滑化）

本專案結果：Pendulum best -117.6
```

### 2018_SAC — Soft Actor-Critic
```
論文：Haarnoja et al. (2018), arXiv:1801.01290
機構：UC Berkeley

核心創新：最大熵強化學習
  目標：max_π E[R + α·H(π)]
    R：累積獎勵（傳統目標）
    H(π)：策略熵（鼓勵探索）
    α：自動調整溫度參數

優點：
  - 樣本效率最高（Pendulum 100K 步收斂）
  - 不需精細調參（α 自動調整）
  - 隨機策略天然避免過擬合

本專案結果：Pendulum -171.8（100K steps）
```

### 2017_HER — Hindsight Experience Replay
```
論文：Andrychowicz et al. (2017), arXiv:1707.01495
機構：OpenAI

解決問題：稀疏獎勵導致 RL 無法學習操作任務
  傳統做法：抓取物體成功才有 +1，失敗全是 0
             探索空間太大，幾乎不可能碰巧成功
  HER 洞察：「即使沒抓到 A，但碰到了 B，B 就當成目標重算」

偽代碼：
  for each episode trajectory (s₀,g,a₀,r₀,...,sT):
    store original (sₜ, g, aₜ, rₜ, sₜ₊₁)
    sample k achieved goals g' from {s₀,...,sT}
    store hindsight (sₜ, g', aₜ, r'ₜ, sₜ₊₁)  ← 事後重標記！

本專案結果：FetchReach-v4，Epoch 160 首達 100%，400-500 ep 穩定 100%
```

---

## 本章節與後續章節的關係

```
本章（Classic RL）→ 03 Goal-Cond（HER 延伸）
                 → 08 RL Finetune（RLT 的 Actor-Critic 頭來自這裡）
```

---

## 參考論文
- DDPG: [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- TD3: [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
- SAC: [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- HER: [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
