# World Models — 世界模型

## 論文

Ha, D., & Schmidhuber, J. (2018).  
*World Models*. arXiv:1803.10122. NeurIPS 2018.

---

## 核心思想 (Key Idea)

代理人可以學習世界在空間上 (V) 與時間上 (M) 的壓縮表示，然後完全在自己的**想像（夢境環境）**中訓練一個微小的控制器 (C)。

```
V (視覺/Vision)     ： 原始畫素觀測 -> 潛在向量 z (使用 VAE)
M (記憶/Memory)     ： (z_t, a_t, h_t) -> h_{t+1} (使用 MDN-RNN)
C (控制器/Controller)： [z_t, h_t] -> 動作 (線性層，由 CMA-ES 演化)
```

---

## 三個核心元件 (Three Components)

| 元件 | 網路架構 | 功能與目的 |
|-----------|---------|---------|
| **V** | VAE (卷積編碼器 + 反摺積解碼器) | 將每一幀畫素影像壓縮至低維向量 $z \in \mathbb{R}^{32}$ |
| **M** | MDN-RNN (LSTM + 混合密度網路) | 預測未來的分佈 $p(z_{t+1} \| z_t, a_t, h_t)$ |
| **C** | 線性層 (Linear layer) | 將狀態表示 $(z, h)$ 對映至動作；由 CMA-ES 進行演化最佳化 |

---

## 訓練流程 (Training Pipeline)

```
階段 1： 收集隨機取樣資料 (Rollouts)
        訓練 VAE：ELBO = 重建損失 + KL 散度

階段 2： 將影像幀編碼為 z 序列
        訓練 MDN-RNN：最小化混合高斯分佈下 $z_{t+1}$ 的負對數似然 (NLL)

階段 3： CMA-ES 演化控制器 C 的權重
        適應度 (Fitness) = 平均集數獎勵 (可於真實環境或「夢境環境」中測試)
```

---

## MDN-RNN 輸出內容 (MDN-RNN Output)

在每一個時間步，MDN 頭部會針對潛在空間輸出：
```
pi:    (n_mix,)        混合成分的權重 (Mixture weights)
mu:    (n_mix, z_dim)  高斯分佈的均值 (Gaussian means)
sigma: (n_mix, z_dim)  高斯分佈的標準差 (Gaussian std devs)
done:  純量 (scalar)   集數結束 (Episode termination) 的機率預測
```

---

## CMA-ES — 協方差矩陣自適應進化策略

- 控制器的引數量極少，僅有 `(z_dim + h_dim) * action_dim + action_dim` 個引數（在預設設定下約 918 個）。
- CMA-ES 將適應度函式（獎勵）視為黑盒 — **不需要計算梯度**，非常適合處理不可導或離散的最佳化目標。
- 安裝方式：`pip install cma`
