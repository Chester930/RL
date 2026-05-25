# MuZero 訓練日誌

## CartPole-v1（2026-05-17）

| 引數 | 數值 |
|------|------|
| n_episodes | 500 |
| hidden_dim | 64 |
| lr | 1e-3 |
| weight_decay | 1e-4 |
| num_simulations | 25 |
| unroll_steps | 5 |
| td_steps | 10 |
| gamma | 0.997 |
| support_size | 601（[-300, 300]，步長 1） |
| B（批次） | 16 遊戲 / 更新 |

### 訓練過程

| 集數 | 回報 |
|------|------|
| 50  | 9.0  |
| 100 | 10.0 |
| 150 | 10.0 |
| 200 | 9.0  |
| 250 | 9.0  |
| 300 | 9.0  |
| 350 | 9.0  |
| 400 | 9.0  |
| 450 | 8.0  |
| 500 | 9.0（最終） |

隨機策略基線：~8–12 步（Pole 快速傾倒）。500 集後回報仍在基線範圍，策略未收斂。

### Bug 修正記錄

**MCTS 擴充套件 TODO 補完（`agent.py` MCTS.run()）：**
- 修正前：擴充套件時未呼叫 dynamics_net，直接以 root value 作為 backup_value
- 修正後：
  1. 追蹤 `last_action`（選擇階段最後一步動作）
  2. 用 `dynamics_net(parent.hidden_state, a_onehot)` 計算 `next_state` 與 reward
  3. 用 `prediction_net(next_state)` 取得子節點的 policy/value
  4. 設定 `node.hidden_state = next_state`，展開所有子動作

**update() TODO 補完（`agent.py` MuZeroAgent.update()）：**
- 從 replay_buffer 取樣 16 個遊戲
- 每個遊戲隨機起始位置，展開 `unroll_steps=5` 步
- 損失函式：
  - Policy loss：交叉熵（one-hot 動作 vs predicted policy）
  - Value loss：交叉熵（n-step bootstrap target vs predicted value，支撐集表示）
  - Reward loss：交叉熵（actual reward vs predicted reward，支撐集表示）
- 新增 `_scalar_to_support()`：two-hot encoding，將純量值對應至 601-atom 支撐集

**新增 `import torch.nn.functional as F`**

### 結論

- 500 集（~5K steps）的 MuZero 在 CPU 上屬於**概念驗證**訓練
- 原始 MuZero 需要數百萬次模擬（Atari: 200M frames，棋盤遊戲: 更多）才能收斂
- CartPole 隨機基線 ~8–12 步，我們的結果落在相同範圍 → MCTS 在未訓練網路上等同隨機決策
- 架構正確（三網路：h/g/f + MCTS UCB selection + expansion + backup + unroll training），梯度流動正常
- 與 Dreamer 的差異：MuZero 的 MCTS 每步需 N×dynamics_net 前向傳播，CPU 每集耗時更長
- 如需完整效果：需 GPU + 更多 simulations（≥50）+ 更長訓練（≥10K episodes）+ 完整 MCTS policy targets（儲存造訪次數分佈而非 one-hot 動作）

---

## CartPole-v1 重跑（2026-05-24，任務 E）

### 修改內容

| 引數 | 舊值 | 新值 | 原因 |
|------|------|------|------|
| num_simulations | 25 | 50 | 更多模擬提升 MCTS 搜尋品質 |
| n_episodes | 500 | 3000 | 給予充足訓練集數 |
| eval_freq | — | 每 100 集 | 加入定期評估與 best checkpoint 儲存 |

### 訓練過程（eval 每 100 集）

| 集數 | Eval 平均 | 備註 |
|------|-----------|------|
| 100 | 9.1 ± 0.5 | 隨機基線 |
| 500 | 8.8 ± 1.0 | 無改善 |
| 1000 | 9.4 ± 0.8 | 無改善 |
| 1500 | 9.5 ± 0.8 | 無改善 |
| 2000 | 9.2 ± 0.7 | 無改善 |
| 2600 | **10.2 ± 0.6** | 最佳（統計雜訊，非真實學習）|
| 3000 | 9.2 ± 0.6 | 最終，與初始相同 |

### 根本原因分析

**為何 num_simulations=50、3000 集仍無法改善？**

1. **短集數死循環**：CartPole 隨機策略每集 ~9 步。MCTS 在 9 步的短樹上做 50 次模擬，所有動作的 UCB 分數接近相同（Q 值均為小數值），結果等同隨機選動作。

2. **Policy target 是隨機動作**：update() 用 `game[idx]["action"]` 作為策略目標，但這個動作本身是 MCTS 在隨機網路上選出的隨機動作。形成死循環：隨機策略 → 短集數 → 隨機動作目標 → 策略學隨機動作。

3. **Representation 未學習**：Representation network 將 4D 狀態映射到 64D 隱藏空間，但無監督信號強迫它學習有意義的表示。隨機初始化 → 所有狀態的隱藏表示互相糾纏 → dynamics/prediction 網路無法區分好壞狀態。

4. **與真實 MuZero 的差距**：論文實作儲存 MCTS 造訪次數分佈（N(s,a)/N(s)）作為策略目標，而非動作 one-hot。這允許 MCTS 的相對評估（哪個動作被造訪更多次）傳遞到策略網路。本骨架實作缺少此機制。

### 結論

本次重跑確認：**MuZero 的結果停在隨機基線（~9）是結構性問題，非超參數問題。**

- 增加 num_simulations（25→50）和 n_episodes（500→3000）對結果無顯著影響
- 若要讓 MuZero 真正學習，需要：MCTS 造訪次數分佈作為 policy target、temperature scheduling（早期探索 → 晚期貪婪）、以及至少 10K 集以上的訓練
- **教學定位調整**：MuZero 作為「骨架架構展示」，重點放在三網路設計（h/g/f）和 MCTS 流程，而非訓練結果。參考 [muzero-general](https://github.com/werner-duvaud/muzero-general) 看完整實作。
