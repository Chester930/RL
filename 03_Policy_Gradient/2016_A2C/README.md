# A2C — 優勢演員-評論家 (Advantage Actor-Critic)

## 參考文獻

Mnih, V., et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML 2016.  
A2C (同步版本的 A3C) 由於 OpenAI Baselines (2017) 的推廣而廣受歡迎。

---

## 核心思想 (Key Idea)

A2C 是 A3C 的**同步且具確定性 (Deterministic)** 的版本。與 A3C 的非同步工作者模式不同，A2C 會等待所有平行環境完成 n 個步數後再統一進行更新。這使得它：
- **對 GPU 更友善**：支援批次更新 (Batched updates)。
- **更具可複現性**：消除了非同步更新帶來的隨機性。
- **實驗表現優秀**：在實證上與 A3C 相當甚至更好。

---

## 廣義優勢估計 (Generalized Advantage Estimation, GAE)

GAE 在 TD(0) 優勢（低方差、高偏差）與蒙特卡羅優勢（高方差、低偏差）之間進行權衡：

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)     (TD 誤差)

A_t^GAE(lambda) = sum_{k=0}^{inf} (gamma * lambda)^k * delta_{t+k}
```

- `lambda = 0`: A_t = delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)  (即 TD 優勢，低方差)
- `lambda = 1`: A_t = G_t - V(s_t)  (即 MC 優勢，低偏差)

---

## 損失函式 (Loss Function)

```
演員損失 (Actor loss)：   L_actor  = -E[log pi(a_t|s_t) * A_t]
評論家損失 (Critic loss)：  L_critic = E[(V(s_t) - G_t)^2]
熵獎勵 (Entropy)：        L_ent    = -E[H(pi(s_t))]

總損失：  L = L_actor + c_v * L_critic + c_e * L_ent
```

---

## A2C vs A3C 比較

| 特性 | A3C | A2C |
|----------|-----|-----|
| 工作者模式 | 非同步 (Async)，無須協調 | 同步 (Sync)，等待所有工作者 |
| 可複現性 | 具不確定性 (Non-deterministic) | 具確定性 (Deterministic) |
| GPU 效率 | 較低 | 較高 |
| 樣本效率 | 相似 | 相似 |
| 實務首選 | 較少見 | 更常見 |

---

## 關鍵公式 (Key Equations)

```
GAE:     A_t = sum_{k>=0} (gamma*lambda)^k * [r_{t+k} + gamma*V(s_{t+k+1}) - V(s_{t+k})]
回報 (Returns)： G_t = A_t + V(s_t)
```
