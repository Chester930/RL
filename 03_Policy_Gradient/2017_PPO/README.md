# PPO — 近端策略最佳化 (Proximal Policy Optimization)

## 論文

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).  
*Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

---

## 核心思想 (Key Idea)

PPO 透過一個極為簡潔的**剪裁機制 (Clipping mechanism)** 來近似 TRPO 的信任域。它透過限制機率比率 (Probability ratio) 來防止過大的策略更新，以極低的實作複雜度達到了與 TRPO 相近的穩定性。

```
L_CLIP(theta) = E_t [ min(r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t) ]

其中 r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
```

使用 `min` 函式是為了確保目標函式是一個下界 (Lower bound) — 這反映了我們對大幅度策略變動持悲觀且謹慎的態度，確保更新是穩健的。

---

## 為什麼有效 (Why PPO Works)

```
比率 < 1-eps: 優勢 > 0 → 動作表現良好，但比率被「剪裁」 (不要推動得太過頭)
比率 > 1+eps: 優勢 < 0 → 動作表現差勁，但比率被「剪裁」 (不要過度抑制)
其餘情況：     執行標準的策略梯度更新
```

剪裁機制防止了策略在單次更新中變動過大，從而提供了一個隱性且高效的信任域 (Implicit Trust Region)。

---

## 演演算法 (Algorithm)

```
針對每次更新：
    1. 使用當前策略 pi_old 收集 T 個步數的資料 (Rollout)
    2. 計算 GAE 優勢函式 A_t
    3. 針對每個訓練週期 (Epoch) K：
       針對每個小批次 (Mini-batch)：
           計算機率比率 r_t = pi_theta / pi_old
           L_CLIP = mean(min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t))
           L_VF = mean((V(s_t) - G_t)^2)
           L_ENT = mean(H(pi(s_t)))
           Loss = -L_CLIP + c_v * L_VF - c_e * L_ENT
           執行梯度更新步驟
```

---

## PPO vs TRPO 比較

| 特性 | PPO | TRPO |
|----------|-----|------|
| 約束方式 | 軟剪裁 (Soft clipping) | 強 KL 約束 (Hard KL) |
| 實作難度 | 簡單 (約 100 行程式碼) | 複雜 (需 CG + 線性搜尋) |
| 取樣更新比 | 每次取樣更新多次 (K=10) | 每次取樣僅更新一次 |
| 效能表現 | 略好 | 優秀 |
| 流行原因 | 簡單且極為可靠 | 理論基礎紮實但難以實作 |

---

## 關鍵超引數 (Key Hyperparameters)

| 引數 | 常用數值 | 影響效果 |
|-----------|--------------|--------|
| clip_eps | 0.1 - 0.2 | 策略允許變動的幅度限制 |
| n_steps | 2048 | 更新前的資料取樣長度 |
| n_epochs | 10 | 每次取樣後的梯度更新次數 |
| gae_lambda | 0.95 | 優勢函式的權衡平滑引數 |
| lr | 3e-4 | Adam 最佳化器的學習率 |

---

## 關鍵公式 (Key Equations)

```
機率比率:   r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
CLIP 損失:  E_t[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
GAE 估計:   A_t = sum_k (gamma*lambda)^k * delta_{t+k}
總損失函式:  L = L_CLIP - c_v * L_VF + c_e * L_ENT
```

PPO 目前是實務中最廣泛使用的策略梯度演演算法，廣泛應用於機器人學、遊戲 AI 以及大語言模型的強化學習 (RLHF)。

---

## 訓練結果 (Training Results)

### CartPole-v1（150K steps）

| 步數 | Eval 平均回報 |
|------|-------------|
| 20K  | **500.0 ± 0.0** |
| 150K | 500.0 ± 0.0 |

**20K steps 就收斂到滿分**，Clipping 使 KL 散度穩定維持在 0.003~0.009。

### LunarLander-v3（300K steps）

> gymnasium 新版棄用 LunarLander-v2，本實作使用 v3（介面相同）。

| 步數 | Eval 平均回報 |
|------|-------------|
| 40K  | -177.4 ± 26.0 |
| 122K | 161.8 ± 93.0 |
| 163K | 199.6 ± 77.0（達到過關線） |
| 204K | 240.4 ± 39.8 |
| 286K | **283.9 ± 20.5** |

同一套 PPO agent，無需任何修改，最終達到 **283.9 分**（過關標準 >200）。
策略標準差從 77 降至 20.5，展現了 PPO 的穩定收斂特性。

---

## PPO vs A3C 核心差異

| 面向 | PPO | A3C |
|------|-----|-----|
| 更新方式 | On-policy + Clipping（可重複利用資料 K 次） | On-policy（每條軌跡只用一次） |
| 穩定性 | 高（Clip 限制策略跳變） | 低（無保護，Critic 易發散） |
| CartPole 結果 | 500.0（20K steps 收斂） | 33.9（Critic 發散） |
| 超引數敏感度 | 低 | 高 |

**Clip fraction** 指每個 mini-batch 中比率 r 被剪裁的比例。理想值約 0.1~0.2：
- 過高（>0.3）：策略更新太激進，clip 在頻繁截斷 → 考慮降低 lr 或增大 n_steps
- 過低（<0.05）：策略幾乎沒在更新 → 可以嘗試增大 clip_eps

---

## GAE 白話說明

GAE (Generalized Advantage Estimation) 解決了優勢估計的 bias-variance 權衡：

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)   ← TD 誤差（低 variance，高 bias）
A_t = delta_t + (gamma*lambda) * delta_{t+1} + ...  ← 多步加權平均
```

`lambda=0`：純 TD（低 variance，但 Critic 誤差直接汙染優勢）  
`lambda=1`：純 Monte Carlo（低 bias，但高 variance）  
`lambda=0.95`：黃金平衡點，實務上最常用。
