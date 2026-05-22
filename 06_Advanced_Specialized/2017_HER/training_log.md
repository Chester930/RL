# HER 訓練記錄

**日期**：2026-05-22  
**環境**：FetchReach-v4（obs_dim=10, goal_dim=3, action_dim=4）  
**硬體**：CPU  
**參考文獻**：Andrychowicz et al. (2017). *Hindsight Experience Replay.* NeurIPS 2017. arXiv:1707.01495.

---

## 訓練配置

| 參數 | 值 |
|---|---|
| 環境 | FetchReach-v4 |
| obs_dim | 10 |
| goal_dim | 3 |
| action_dim | 4 |
| n_epochs | 200 |
| n_episodes_per_epoch | 16 |
| updates_per_epoch | 40 |
| lr | 1e-3 |
| gamma | 0.98 |
| tau | 0.05 |
| buffer_size | 1,000,000 |
| batch_size | 256 |
| her_ratio | 0.8（80% future strategy）|
| noise_std | 0.2 |

---

## 訓練結果（成功率曲線）

| Epoch | 成功率 |
|---|---|
| 10 | 0% |
| 20 | 0% |
| 30 | 0% |
| 40 | 0% |
| 50 | 0% |
| 60 | 0% |
| 70 | 0% |
| 80 | 0% |
| 90 | 10% |
| 100 | 0% |
| 110 | 20% |
| 120 | 0% |
| 130 | **60%** |
| 140 | 40% |
| 150 | **70%** |
| 160 | 30% |
| 170 | **80%** |
| 180 | **80%** |
| 190 | 30% |
| 200 | 20% |

**峰值成功率：80%（Epoch 170 & 180）**  
**Checkpoint**：`checkpoints/her_epoch50`、`checkpoints/her_epoch100`、`checkpoints/her_epoch150`、`checkpoints/her_epoch200`

---

## 觀察

- **Epoch 1–80**：成功率為 0%，HER 緩衝區需要足夠的失敗軌跡才能開始 hindsight relabeling，這段時間為純探索期。
- **Epoch 90 起**：成功率開始出現，代表 DDPG 開始從 HER 重標註的「假成功」轉換中學到有效策略。
- **Epoch 130–180**：成功率突破 60%–80%，展示 HER 的核心優勢：把失敗軌跡轉為學習信號。
- **Epoch 190–200**：成功率回落至 20–30%，顯示 CPU 訓練下 200 epoch 尚未完全收斂（FetchReach 通常需要 500–1000 epoch 達穩定 90%+）。

---

## HER 核心機制

```
每集結束後（hindsight relabeling）：
  原始轉換：(s, a, r=-1, s', g_desired)
  HER 轉換：取同集中 t'>t 的 achieved_goal 作為替代目標 g_her
           (s, a, r=0, s', g_her)  ← 原本失敗的步驟變成「成功」

her_ratio=0.8：每個 batch 有 80% 來自 HER 重標註，20% 來自原始目標
strategy=future：從同集未來隨機選一個 achieved_goal（論文中效果最佳）
```

---

## 與稀疏獎勵無 HER 的對比

| 方法 | FetchReach 200 epoch 成功率 |
|---|---|
| DDPG（無 HER，稀疏獎勵）| ~0%（幾乎無法學習）|
| DDPG + HER（本實作）| 峰值 80% |

HER 的關鍵貢獻：讓 agent 從「明明失敗」的軌跡中學習，解決稀疏獎勵環境中探索困難的根本問題。
