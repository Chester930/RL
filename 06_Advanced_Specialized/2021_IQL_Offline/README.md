# IQL — 隱式 Q 學習 (Implicit Q-Learning) 與離線強化學習 (Offline RL)

## 論文

Kostrikov, I., Nair, A., & Levine, S. (2021).  
*Offline Reinforcement Learning with Implicit Q-Learning*.  
ICLR 2022. arXiv:2110.06169.

---

## 核心思想 (Key Idea)

在**完全不查詢資料集以外動作 (OOD actions)**的情況下訓練 Q 函式。

傳統離線 RL 演演算法（如 CQL）會透過懲罰項來處理 OOD 動作，而 IQL 則直接從根本上避免了查詢 OOD 動作。它透過使用 V(s') 代替 max_a' Q(s', a') 來計算貝爾曼目標值，從而徹底解決了分佈偏移 (Distributional shift) 的問題。

```
V-更新： argmin_V  E_D[ L_tau(Q(s,a) - V(s)) ]   <- 期望分位數回歸 (Expectile regression)
Q-更新： argmin_Q  E_D[ (r + gamma*V(s') - Q(s,a))^2 ]  <- 安全的貝爾曼備份
A-更新： argmax_pi E_D[ exp(beta * (Q(s,a) - V(s))) * log pi(a|s) ]  <- 優勢加權回歸 (AWR)
```

---

## 期望分位數回歸 (Expectile Regression)

```
L_tau(u) = |tau - I(u<0)| * u^2

- tau = 0.5 -> 普通均方誤差 (MSE)，估計均值。
- tau = 0.7 -> 上期望分位數 (Upper expectile)，論文預設值。
- tau -> 1.0 -> 估計最大值 (隱式最大 Q 學習)。
```

這使得 V(s) 能夠逼近資料集策略下可能達到的最大 Q 值，但不需要像 SAC 那樣對動作空間進行取樣或最佳化。

---

## 優勢加權回歸 (Advantage-Weighted Regression, AWR)

```
演員損失 (Actor loss) = -E[ w(s,a) * log pi(a|s) ]
w(s,a) = clip(exp(beta * (Q(s,a) - V(s))), max=100)

高優勢 (High advantage) -> 高權重 -> 更強烈地學習 (克隆) 該 (s,a) 動作對。
```

---

## 與 CQL 對比 (vs CQL)

| 特性 | CQL | IQL |
|----------|-----|-----|
| **OOD 查詢** | 是（透過重要性取樣估計） | **從不查詢** |
| **訓練穩定性** | 對 cql_alpha 引數較敏感 | **非常穩定** |
| **策略提取** | 類似 SAC 的線上最佳化 | 離線 AWR (加權行為複製) |
| **D4RL 擅長場景** | medium, medium-replay | **medium-expert, antmaze** |
