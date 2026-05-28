# 03 — 目標條件學習（2017–2021）

> **時代定位**：讓一個策略泛化到多種目標，解決稀疏獎勵難題  
> **核心突破**：HER 事後重標記 + Universal Value Function

---

## 核心概念：目標條件策略

```
普通策略：π(a | s) → 只會做一件事
目標條件策略：π(a | s, g) → 會針對不同目標做不同事

例子：
  π(a | 「手臂當前狀態」, 「把紅球移到 (0.3, 0.2, 0.1)」)
  → 不同目標 g → 不同動作序列
  → 一個模型學會所有位置的 Reach/Push/Pick
```

---

## 本章節演算法

### 2017_HER_GoalRL — HER + 目標條件完整框架
```
超越單純 HER：
  - Goal-conditioned Q-function: Q(s, a, g)
  - Universal Value Functions (UVFAs)
  - Multi-goal 訓練：同時學習所有目標
  - Curriculum Learning：從易到難的目標排程

HER 三種重標記策略：
  future：從同軌跡未來的狀態取目標（效果最好）
  episode：從同集任意狀態取目標
  random：從所有集隨機取目標

實際效果（本專案 FetchReach-v4）：
  無 HER：幾乎 0% 成功率（稀疏獎勵 RL 探索不到）
  有 HER：Epoch 160 首達 100%，之後穩定 100%
```

### 2021_GCSL — Goal-Conditioned Supervised Learning
```
論文：Ghosh et al. (2021), arXiv:2011.10024

更激進的觀點：不用 RL，純監督學習就能做目標條件策略！

原理：
  1. 讓智能體隨機行動收集軌跡
  2. 對每個軌跡，把最終到達的狀態當成「目標」
  3. 對每個步驟，監督學習：「在狀態 s 時為了達到 g 應該執行動作 a」
  4. 迭代：策略改善 → 到達更好的狀態 → 更好的監督信號

優點：訓練比 RL + HER 更穩定
限制：需要任務具有一定結構（起點到終點路徑合理）
```

---

## 目標條件學習的局限與突破

```
局限：
  - 目標必須是可量化的狀態（「移到 (x,y,z)」）
  - 不能直接處理「拿起杯子」這類語言目標
  - 多物體、長序列任務仍然困難

下一步：
  語言目標 → SayCan, RT-2
  「拿起杯子」→ 語言模型理解 → 分解為可量化子目標
```

---

## 參考論文
- HER: [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
- UVFA: Schaul et al. (2015) [arXiv:1506.08941](https://arxiv.org/abs/1506.08941)
- GCSL: [arXiv:2011.10024](https://arxiv.org/abs/2011.10024)
