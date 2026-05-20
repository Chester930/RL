# TRPO — 信任域策略最佳化 (Trust Region Policy Optimization)

## 論文

Schulman, J., Levine, S., Abbeel, P., Jordan, M. I., & Moritz, P. (2015).  
*Trust Region Policy Optimization*. ICML 2015. arXiv:1502.05477.

---

## 核心思想 (Key Idea)

標準的策略梯度演演算法可能會因為步長過大而導致效能崩潰。TRPO 透過**約束新舊策略之間的 KL 散度 (KL divergence)** 來解決此問題，從而保證了效能的單調提升 (Monotonic Improvement)：

```
最大化：   L(theta) = E_t [ pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) * A_t ]
限制條件： KL(pi_theta_old || pi_theta) <= delta
```

這種信任域 (Trust Region) 約束確保了我們在更新策略時不會過於激進，從而維持訓練的穩定性。

---

## 演演算法 (Algorithm)

```
1. 在舊策略 pi_old 下進行資料取樣 (Rollouts)
2. 計算優勢函式 A_t (通常使用 GAE)
3. 計算策略梯度 g = grad L(theta_old)
4. 求解 Hx = g 以獲得自然梯度方向 x
   (H = Fisher 訊息矩陣 = E[grad log pi * grad log pi^T])
5. 計算滿足 KL 約束的最大步長：
   s = sqrt(2*delta / x^T H x) * x
6. 使用回溯線性搜尋 (Backtracking line search) 尋找滿足 KL 的最大實際步長
7. 使用梯度下降更新評論家 (Critic) 網路
```

---

## 共軛梯度法 (Conjugate Gradient)

由於顯式計算 $H^{-1}$ 的成本極高（引數向量可能高達數百萬維），TRPO 使用**共軛梯度法 (CG)** 來疊代求解 $Hx = g$。這僅需要計算 **Fisher-向量積 (Fisher-vector products)**，該項可透過自動微分技術求得，而不需要顯式儲存矩陣 $H$：

```
FVP(v) = d/d_theta [grad KL^T v]   (無須顯式計算 H)
```

---

## TRPO vs PPO 比較

| 特性 | TRPO | PPO |
|----------|------|-----|
| 約束方式 | 強 KL 約束 (Hard constraint) | 軟剪裁 (Soft clipping) |
| 實作難度 | 複雜 (需 CG + 線性搜尋) | 簡單 |
| 理論保證 | 單調提升保證 | 近似保證 |
| 效能表現 | 優秀 | 優秀且通常略好 |
| 執行時間 | 較慢 (計算密集) | 快 |

TRPO 是 PPO 的重要前身。PPO 透過更簡潔的目標函式設計，達到了與 TRPO 相近甚至更好的效果。

---

## 關鍵公式 (Key Equations)

```
代理目標函式 (Surrogate objective)： L(theta) = E_t [r_t(theta) * A_t]
                                 其中 r_t = pi_theta(a|s) / pi_old(a|s)

KL 約束條件： KL(pi_old || pi_new) <= delta

自然梯度方向： d = H^{-1} g     (透過 CG 求解)
步長大小：     s = sqrt(2*delta / d^T H d) * d
```
