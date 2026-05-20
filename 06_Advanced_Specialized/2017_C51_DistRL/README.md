# C51 — 分類式 DQN 與分散式強化學習 (Distributional RL)

## 論文

Bellemare, M. G., Dabney, W., & Munos, R. (2017).  
*A Distributional Perspective on Reinforcement Learning*.  
ICML 2017. arXiv:1707.06887.

---

## 核心思想 (Key Idea)

學習**完整的回報分佈 (Return distribution)** $Z(s,a)$，而不僅僅是學習其期望值 $E[Z(s,a)] = Q(s,a)$。

```
Z(s, a) ~ 在 N 個原子 (Atoms) {z_1, ..., z_N} 上的分類分佈 (Categorical distribution)
          其中 z_i = v_min + i * (v_max - v_min) / (N-1)

分散式貝爾曼方程式 (Distributional Bellman):
    T Z(s, a) = R + gamma * Z(s', pi(s'))
```

---

## 貝爾曼投影 (Bellman Projection)

```
針對每個原子 z_j:
    T_z_j = clip(r + gamma*(1-done)*z_j, v_min, v_max)
    b_j   = (T_z_j - v_min) / delta_z   # 小數索引 (Fractional index)
    m[floor(b_j)] += p_j * (ceil(b_j) - b_j)
    m[ceil(b_j)]  += p_j * (b_j - floor(b_j))

損失函式 = -sum_j m_j * log p_j(s, a)   # 交叉熵 (Cross-entropy)
```

---

## 超引數 (Hyperparameters)

| 引數 | 數值 |
|-----------|-------|
| N (原子數量) | 51 |
| v_min | -10 |
| v_max | 10 |
| delta_z | 20/50 = 0.4 |
| 學習率 (lr) | 6.25e-5 |

---

## 為什麼有效 (Why it works)

- 回報分佈能捕捉**「多模態 (Multi-modality)」**特性（例如：分辨高風險高回報路徑與穩健路徑）。
- 相較於傳統的純量 TD 誤差，提供了更豐富且穩定的學習訊號。
- 它是後續許多強大演演算法（如 Rainbow、QR-DQN、IQN）的理論核心。
