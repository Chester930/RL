# REINFORCE | 基礎策略梯度演演算法

## 論文

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4), 229–256.

---

## 核心思想 (Key Idea)

REINFORCE 是最簡單的策略梯度 (Policy Gradient) 演演算法。它透過沿著期望回報的梯度方向直接最佳化策略引數：

```
theta <- theta + alpha * G_t * grad_theta log pi(a_t | s_t; theta)
```

「強化那些導致高回報的動作；抑制那些導致低迴報的動作。」

---

## 策略梯度定理 (Policy Gradient Theorem)

```
grad_theta J(theta) = E_{pi} [ G_t * grad_theta log pi(a_t | s_t; theta) ]
```

對數導數技巧 (Log derivative trick) 讓我們能透過樣本估計此梯度：
```
對數機率技巧: grad E[f(x)] = E[f(x) * grad log p(x)]
```

---

## 演演算法 (Algorithm)

```
針對每個集數 (Episode)：
    收集軌跡 (Trajectory)：(s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T)
    針對每個時步 t = 0 到 T-1：
        G_t = sum_{k=t}^{T} gamma^{k-t} * r_{k+1}
        theta += alpha * G_t * grad log pi(a_t | s_t; theta)
```

---

## 方差縮減 (Variance Reduction)

REINFORCE 具有**高方差**的缺點 — 即使在相同狀態採取相同動作，也可能因為後續隨機事件導致回報截然不同。

常用的方差縮減技術：

**1. 減去基準值 (Baseline subtraction)**（無偏）：
```
theta += alpha * (G_t - b(s_t)) * grad log pi(a_t | s_t)
```

**2. 回報歸一化 (Return normalization)**（批次級別）：
```
G_t = (G_t - mean(G)) / (std(G) + eps)
```

**3. 演員-評論家 (Actor-Critic)**（使用可學習的 V(s_t) 作為基準 — 見後續演演算法）

---

## 侷限性 (Limitations)

- 僅在**完整集數**結束後才更新（無法進行線上學習）。
- **高方差**導致訓練緩慢且雜訊多。
- 樣本效率低下（每條軌跡僅使用一次）。
- 學習率難以調整（對步長非常敏感）。

這些問題催生了 A2C、PPO、TRPO 等後續演演算法。

---

## 關鍵公式 (Key Equations)

```
回報 (Return)：  G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k+1}
損失函式 (Loss)： L = -sum_t G_t * log pi(a_t | s_t)     [負號是因為我們要進行最大化]
更新規則：        theta <- theta - grad L
```

---

## 方法脈絡 (Lineage)

**建立於**
- 蒙特卡羅 MC (`01_Tabular_Basics/1980s_MC`) — 使用完整集數回報 G_t 作為更新訊號，相同的倒序累積計算；從表格式 Q(s,a) 延伸至引數化策略 π(a|s;θ)
- 動態規劃 DP (`01_Tabular_Basics/1950s_DP`) — V(s) 基準值 (baseline) 的概念：用狀態價值估計降低梯度方差

**直接延伸出**

| 演算法 | 資料夾 | 繼承點 |
|:---|:---|:---|
| A2C | `03_Policy_Gradient/2016_A2C` | REINFORCE + 學習式基準值：用神經網路 V(s) 取代固定基準，形成 Actor-Critic 架構 |
| A3C | `02_Value_Based_Deep/2016_A3C` | 非同步並行 REINFORCE + Critic；多個 Worker 同時探索，梯度匯總到全域網路 |
| PPO | `03_Policy_Gradient/2017_PPO` | REINFORCE + clip ratio 信賴域 + GAE 優勢估計，解決更新步長過大導致策略崩潰 |

**橫向對比**

| 演算法 | 資料夾 | 主要差異 |
|:---|:---|:---|
| 蒙特卡羅 MC | `01_Tabular_Basics/1980s_MC` | 表格式 Q(s,a) 更新 vs 引數化策略梯度；MC 直接更新查詢表，REINFORCE 透過梯度最佳化神經網路 |
| Q-Learning | `01_Tabular_Basics/1989_QLearning` | 基於價值（隱含策略 argmax Q）vs 基於策略（直接輸出動作分佈）；Q-Learning 學 Q*，REINFORCE 學 π* |
