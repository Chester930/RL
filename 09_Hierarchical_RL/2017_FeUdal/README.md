# FeUdal Networks | 封建網路層次架構

> FeUdal Networks for Hierarchical Reinforcement Learning  
> Vezhnevets et al. — ICML 2017

---

## 核心思想

仿照「封建制度」的兩層結構：

```
Manager（領主）：
  - 在低維嵌入空間設定「子目標向量」g_t
  - 每 c 步才更新一次（c=10 左右）
  - 用 cosine similarity 計算內在獎勵給 Worker

Worker（農奴）：
  - 每步執行動作
  - 內在目標：讓當前狀態朝著 g_t 方向移動
```

**關鍵設計**：Manager 在抽象空間設目標，Worker 在原始動作空間執行，兩者共享底層特徵提取器（Percept）。

---

## 環境 / 實驗設定

- 環境：MiniGrid、Montezuma's Revenge（Atari 難關）
- 特點：適合稀疏獎勵 + 長序列任務

---

## 檔案結構

```
2017_FeUdal/
├── agent.py        # Manager + Worker 兩層網路
├── train.py        # 訓練指令碼
└── training_log.md # 訓練日誌
```

---

## 論文

- Paper: https://arxiv.org/abs/1703.01161
