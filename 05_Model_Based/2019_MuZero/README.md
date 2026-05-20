# MuZero — 透過學習模型與規劃掌控複雜環境

## 論文

Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., Lillicrap, T., & Silver, D. (2019).  
*Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. Nature 588, 604–609. arXiv:1911.08265.

---

## 核心思想 (Key Idea)

**AlphaZero 需要預先知道遊戲規則。** 而 MuZero 則是從零開始學習一切 — 包括環境模型 — 它使用了三個學習到的核心函式：

```
h： 觀測 -> 隱藏狀態          (表示模型 / Representation)
g： (隱藏狀態, 動作) -> (下一個狀態, 獎勵)  (動態模型 / Dynamics)
f： 隱藏狀態 -> (策略, 價值)      (預測模型 / Prediction)
```

蒙特卡羅樹搜尋 (MCTS) 完全在潛在空間中利用 g 與 f 進行規劃，完全無需接觸真實環境。

---

## 三個核心網路 (Three Networks)

| 網路 | 輸入 | 輸出 | 目的 |
|---------|-------|--------|---------|
| **表示模型 h** | 觀測 $o_t$ | 狀態 $s_t$ | 編碼觀測影像為隱藏狀態 |
| **動態模型 g** | ($s_t, a_t$) | ($s_{t+1}, r_t$) | 預測下一個狀態與即時獎勵 |
| **預測模型 f** | $s_t$ | ($\pi_t, v_t$) | 為 MCTS 提供策略分佈與價值評估 |

---

## 潛在空間中的 MCTS (MCTS in Latent Space)

```
根節點： s_0 = h(obs)
針對每一次模擬：
    1. 選擇 (Selection)： 根據樹中的 UCB 分數向下走
    2. 擴充套件 (Expansion)： 對每個可能動作執行 s_{t+1}, r_t = g(s_t, a_t)
    3. 評估 (Evaluation)： pi, v = f(s_{t+1})
    4. 回傳 (Backup)： 將價值沿路徑向上更新至樹根

動作選擇： 根據根節點的造訪次數 (Visit counts) 分佈進行取樣
```

---

## 訓練 (Training)

從儲存的對局中，針對 $K$ 個展開步數進行訓練：
```
s_0 = h(obs_t)
針對 k = 1..K:
    pi_k, v_k = f(s_{k-1})
    r_k, s_k = g(s_{k-1}, a_{t+k-1})

損失函式：
    策略損失： cross_entropy(pi_k, MCTS_造訪次數_{t+k})
    價值損失： cross_entropy(v_k, 引導目標_{t+k})
    獎勵損失： cross_entropy(r_k, r_{t+k})
```

---

## MuZero vs AlphaZero 比較

| 特性 | AlphaZero | MuZero |
|----------|-----------|--------|
| 環境模型 | 使用已知規則 | 自主學習 (g, h, f) |
| 適用環境 | 僅限棋盤遊戲 (Board games) | Atari 影像 + 棋盤遊戲 |
| 規劃方式 | 對真實狀態進行樹搜尋 | 對潛在狀態進行樹搜尋 |
| 效能表現 | 在圍棋、西洋棋達超越人類水平 | 在 Atari 與棋盤遊戲皆達超越人類水平 |
