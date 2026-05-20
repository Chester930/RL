# TD3 — 雙延遲深度確定性策略梯度 (Twin Delayed DDPG)

## 論文

Fujimoto, S., van Hoof, H., & Meger, D. (2018).  
*Addressing Function Approximation Error in Actor-Critic Methods*. ICML 2018. arXiv:1802.09477.

---

## 核心思想 (Key Idea)

TD3 識別並解決了 DDPG 在實際應用中遇到的三個主要失敗模式：

### 1. 過估計偏差 (Overestimation Bias)

與 DQN 類似，DDPG 中的最大化操作會導致對 Q 值的持續高估。TD3 透過使用**雙評論家 (Twin Critics)** 並在計算目標值時取其最小值來緩解此問題：

```
y = r + gamma * min(Q1_target(s', pi_target(s')), Q2_target(s', pi_target(s')))
```

### 2. 不穩定的演員更新 (Unstable Actor Updates)

演員網路容易過度擬合評論家中不精確的評估。TD3 引入了**延遲策略更新 (Delayed Policy Updates)** — 只有在評論家更新了 $d=2$ 次之後，才執行一次演員網路與目標網路的更新，這能讓評論家網路在引導演員之前變得更加穩定。

### 3. Q 函式誤差 (Q-function Errors)

Q 函式中的尖峰可能會導致模型開發出錯誤的動作價值。TD3 使用**目標策略平滑 (Target Policy Smoothing)** — 在計算目標動作時加入經過剪裁的隨機雜訊，這能使目標動作在小範圍內變動，從而讓 Q 函式學會更平滑的動作價值估計：

```
a' = clip(pi_target(s') + clip(N(0, sigma), -c, c), -a_max, a_max)
```

---

## 三項改進總結 (Three Improvements Summary)

| 改進措施 | 解決的問題 | 實作方式 |
|-------------|------------------|----------------|
| **雙評論家** | Q 值高估問題 | 計算目標值時取 min(Q1, Q2) |
| **延遲更新** | 演員訓練不穩定 | 演員與目標網路每隔 d=2 步更新一次 |
| **策略平滑** | Q 函式劇烈波動 | 在目標動作中加入經剪裁的雜訊 |

---

## 更新時序 (Update Schedule)

```
每一步： 更新兩個評論家 (Critics)
每兩步： 更新演員 (Actor) 以及兩套目標網路 (Target Networks，採軟更新)
```

---

## DDPG vs TD3 比較

| 特性 | DDPG | TD3 |
|----------|------|-----|
| 評論家數量 | 1 | 2 (雙網路) |
| 演員更新頻率 | 每一步 | 每兩步 (延遲更新) |
| 目標動作 | pi_target(s') | pi_target(s') + 剪裁雜訊 |
| Q 值偏差 | 高 (容易過估) | 較低且穩定 |
| 訓練穩定性 | 較低 | 顯著提高 |
