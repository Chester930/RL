# RL² | 用迴圈網路實現元學習

> Learning to Reinforcement Learn  
> Wang et al. — 2016

---

## 核心思想

把 Agent 設計成 **RNN（迴圈神經網路）**，讓它用隱藏狀態記住過去任務的經驗。  
不需要顯式的 inner/outer loop——RNN 的隱藏狀態自然扮演「記憶」的角色。

```
輸入：(觀測, 前一步獎勵, 前一步動作, 是否結束)
↓
RNN 隱藏狀態 h_t（跨集數不重置，跨任務才重置）
↓
輸出：動作分佈 π(a|h_t)
```

**關鍵**：在同一個任務的多集訓練中，h_t 會學會「這個任務的規律」。

---

## 環境 / 實驗設定

- 環境：Multi-armed bandit、迷宮導航、視覺辨認任務
- 訓練：每次輸入多集（episode）軌跡，RNN 跨集保留狀態

---

## 檔案結構

```
2016_RL2/
├── agent.py        # RNN-based Meta Agent
├── train.py        # 訓練指令碼（多工取樣）
└── training_log.md # 訓練日誌
```

---

## 論文

- Paper: https://arxiv.org/abs/1611.05763
