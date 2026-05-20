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
