# 優先經驗回放 (Prioritized Experience Replay, PER)

## 論文

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015).  
*Prioritized Experience Replay*. ICLR 2016. arXiv:1511.05952.

---

## 核心思想 (Key Idea)

標準的 DQN 從經驗回放池中「均勻地」(Uniformly) 取樣。但某些轉移資料比其他的更具資訊量 — 那些具有高 TD 誤差 (TD Error) 的樣本在單次取樣中能讓網路學到更多。

PER 根據 TD 誤差的大小，按比例取樣轉移資料：

```
P(i) = p_i^alpha / sum_k p_k^alpha

其中 p_i = |delta_i| + epsilon    (比例式變體)
      p_i = 1 / rank(i)             (排名式變體)
```

---

## 重要性取樣校正 (Importance Sampling Correction)

非均勻取樣會引入偏差。PER 使用重要性取樣權重 (IS weights) 來修正此問題：

```
w_i = (1 / (N * P(i)))^beta

權重會進行歸一化：w_i /= max_j w_j
```

`beta` 會從訓練初期的 `beta_start` (0.4) 逐漸增加到 1.0，隨著訓練趨於穩定，逐步強化偏差修正。

---

## SumTree 資料結構 (SumTree Data Structure)

SumTree 讓取樣與優先權更新的時間複雜度都維持在 O(log N)：

```
          42          <- 總優先權 (根節點)
         /   \
       29     13
      /  \   /  \
    13   16  5   8   <- 葉節點優先權 (個別樣本)
```

取樣方式：在 [0, 總和] 區間抽取一個均勻分佈的數 u，然後從根節點向下遍歷樹。

---

## 比例式 vs 排名式 (Proportional vs Rank-Based)

| 變體 | 優先權計算 | 穩定性 | 備註 |
|---------|---------|---------|-------|
| **比例式 (Proportional)** | `|TD| + eps` | 對離群值較敏感 | 本實作採用之方式 |
| **排名式 (Rank-based)** | `1/rank` | 較穩定 | 厚尾分佈 (Heavy-tail) |

---

## 關鍵超引數 (Key Hyperparameters)

| 引數 | 數值 | 效果 |
|-----------|-------|--------|
| alpha | 0.6 | 優先權指數 (0=均勻取樣, 1=完全 PER) |
| beta_start | 0.4 | IS 修正係數 (逐漸增加至 1.0) |
| epsilon | 1e-6 | 確保所有樣本的優先權皆不為零 |
