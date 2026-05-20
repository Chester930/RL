# PEARL | 機率嵌入元強化學習

> Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables  
> Rakelly et al. — ICML 2019

---

## 核心思想

用一個**機率編碼器**把任務壓縮成低維向量 z（latent context），  
Agent 根據 z 調整策略，不需要像 MAML 一樣反覆計算高階梯度。

```
Encoder：把過去幾筆 (s, a, r, s') 壓縮成 z ~ q(z | context)
Policy ：π(a | s, z)
Q-func ：Q(s, a, z)

推斷期：只需少量互動更新 z，不更新整個網路
訓練期：off-policy，可重複利用舊資料
```

**優點**：比 MAML 高效得多（不需要二階梯度），支援 off-policy 訓練。

---

## 環境 / 實驗設定

- 環境：MuJoCo（HalfCheetah、Ant、Humanoid 的速度/方向變體）
- 任務數：100~200 個訓練任務，20 個測試任務

---

## 檔案結構

```
2019_PEARL/
├── agent.py        # PEARL Agent（encoder + SAC）
├── train.py        # 訓練指令碼（meta-train + meta-test）
└── training_log.md # 訓練日誌
```

---

## 論文

- Paper: https://arxiv.org/abs/1903.08254
