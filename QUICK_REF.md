# 講師速查表（課堂備用）

> 課堂中快速定位程式碼、回答聽眾問題用。Ctrl+F 搜尋演算法名稱。

---

## BC（Behavioral Cloning）
**路徑**：`00_Imitation/2004_BC/train.py`

| 超參數 | 預設值 | 行號 |
|--------|--------|------|
| `n_epochs` | 100 | 191 |
| `batch_size` | 256 | 192 |
| `lr` | 1e-3 | 193 |

**結果重點**：
- 0°–90° 表現與 SAC 相當 ✅
- 120°+ 開始崩潰（Compounding Error）❌
- 判斷標準：`-123`（0°）vs `−245`（120°）差距超過 2 倍即為失敗

**常見問題**：
- Q：「lr 調小會不會更好？」→ 問題不在 lr，在於訓練資料不涵蓋偏離狀態
- Q：「batch_size 怎麼選？」→ 256 是標準起點；數據量小可降到 64

---

## Q-Learning
**路徑**：`01_Tabular_Basics/1989_QLearning/train.py`

| 超參數 | 預設值 | 行號 |
|--------|--------|------|
| `alpha`（學習率）| 0.1 | 369 |
| `gamma`（折扣）| 0.99 | 370 |
| `epsilon_start` | 1.0 | 371 |
| `epsilon_end` | 0.01 | 372 |

**結果重點**：
- 環境：FrozenLake-v1（4×4 格子）
- 成功標準：success rate > 70%
- 最終結果列印於行 341，log 寫入 `training_log.md`（行 359）

**常見問題**：
- Q：「alpha 為什麼 0.1，不是 0.01？」→ FrozenLake 狀態空間小，可以用較大 lr 收斂更快
- Q：「gamma 設 0.9 差很多嗎？」→ 短途任務差不多；長途（機器手臂）差異很大

---

## DQN（Deep Q-Network）
**路徑**：`02_Value_Based_Deep/2013_DQN/train.py`

| 超參數 | 預設值 | 行號 |
|--------|--------|------|
| `lr` | 1e-3 | 23（CONFIG dict）|
| `gamma` | 0.99 | 24 |
| `batch_size` | 64 | 26 |
| `target_update` | 每 N 步 | CONFIG 20–39 |

**結果重點**：
- 環境：CartPole-v1
- 150K steps → eval = **500**（滿分）🎉
- Q 值高估：mean Q ≈ 312（理論值約 50）→ 高估是正常現象，不是 bug
- eval 分數列印行 382

**常見問題**：
- Q：「Q 值高估代表有問題嗎？」→ DQN 已知弱點，DDQN 修正此問題；不影響 CartPole 效果
- Q：「batch_size 64 太小嗎？」→ CartPole 狀態簡單，64 夠用；複雜任務建議 256

---

## DDPG（Deep Deterministic Policy Gradient）
**路徑**：`04_Actor_Critic_Continuous/2015_DDPG/train.py`

| 超參數 | 預設值 | 行號 |
|--------|--------|------|
| `lr_actor` | 1e-4 | 82 |
| `lr_critic` | 1e-3 | 83 |
| `gamma` | 0.99 | 84 |
| `tau`（soft update）| 0.005 | 85 |

**結果重點**：
- 環境：Pendulum-v1（單關節扭矩控制）
- 最佳 eval = **−101.6**（接近最優 −100）🎉
- 標準差極大（±126）→ 不穩定是預期現象
- eval 分數列印行 70

**常見問題**：
- Q：「lr_actor 為什麼比 lr_critic 小 10 倍？」→ Actor 更新太快會讓 Critic 來不及追，縮小步長讓訓練穩定
- Q：「tau 是什麼？」→ Target Network 的軟更新比例，每步 θ' ← (1−τ)θ' + τθ

---

## TD3（Twin Delayed Deep Deterministic）
**路徑**：`04_Actor_Critic_Continuous/2018_TD3/train.py`

| 超參數 | 預設值 | 行號 |
|--------|--------|------|
| `lr_actor` / `lr_critic` | 3e-4 | 75 |
| `gamma` | 0.99 | 76 |
| `buffer_size` | 200,000 | 77 |
| `batch_size` | 256 | 77 |
| Actor 更新頻率 | 每 2 步 | `if step % 2 == 0` |

**結果重點**：
- 環境：Pendulum-v1
- 最佳 eval = **−119.8**，70K 步收斂（比 DDPG 快）
- 比 DDPG 數字略差，但標準差更小、更穩定 → 這是正常的
- eval 分數列印行 63

**常見問題**：
- Q：「為什麼 TD3 比 DDPG 分數差但說更好？」→ Pendulum 太簡單，DDPG 偶爾運氣好能達最優；複雜任務 TD3 才勝出；穩定性才是關鍵
- Q：「兩個 Critic 哪個是主的？」→ 都一樣地位，只是取 min；訓練時分開更新

---

## SAC（Soft Actor-Critic）
**路徑**：`04_Actor_Critic_Continuous/2018_SAC/train.py`

| 超參數 | 預設值 | 行號 |
|--------|--------|------|
| `lr` | 3e-4 | 73 |
| `gamma` | 0.99 | 74 |
| `tau` | 0.005 | 75 |
| `alpha`（溫度）| 自動調整 | — |

**結果重點**：
- Pendulum：eval = **−171.8**（比 DDPG/TD3 差 → 正常，Pendulum 太簡單）
- **LunarLanderContinuous-v3**：eval = **262.4**（50K 步突破 200）🎉
- Alpha 自動從 0.32 降至 0.02
- eval + alpha 同時列印行 61

**常見問題**：
- Q：「為什麼 SAC 在 Pendulum 反而最差？」→ Entropy 鼓勵探索，但 Pendulum 只有一種最優動作，多樣性反而是負擔
- Q：「alpha 能手動固定嗎？」→ 可以，但失去自動調整的優勢；建議維持自動

---

## PPO（Proximal Policy Optimization）
**路徑**：`03_Policy_Gradient/2017_PPO/train.py`

| 超參數 | 預設值 | 行號 |
|--------|--------|------|
| `lr` | 3e-4 | 96 |
| `gamma` | 0.99 | 97 |
| `n_steps` | 2048 | 98 |
| `clip_range`（ε）| 0.2 | ≈ 100 |
| `n_epochs` | ≈ 10 | ≈ 101 |

**結果重點**：
- CartPole：20K 步 → eval = **500**（滿分）🎉
- LunarLander-v3：eval ≈ **284**（穩定收斂）
- 對比 REINFORCE：5000 集 eval 仍為 9.5（完全沒學到）
- eval + approx_kl 列印行 74–84

**常見問題**：
- Q：「clip_range 0.2 可以調嗎？」→ 0.1 更保守（收斂慢但更穩）；0.3 更激進（可能不穩）；0.2 是論文推薦值
- Q：「n_steps 和 batch_size 差在哪？」→ n_steps 是每次收集多少步再更新；收集完才分 batch 訓練
- Q：「approx_kl 是什麼？」→ 新舊策略的 KL 散度估計；如果突然飆高（>0.05）代表更新太激進

---

## 快速比較（Pendulum-v1）

| 演算法 | 最佳 Eval | 收斂步數 | 方差 |
|--------|-----------|----------|------|
| DDPG | −101.6 | 100K | 高 ❌ |
| TD3 | −119.8 | 70K | 中 ✅ |
| SAC | −171.8 | 100K | 低（但過探索）|

> Pendulum 最優約為 −100；數字越接近 0 越好

## 快速比較（CartPole-v1）

| 演算法 | 最終 Eval | 步數 |
|--------|-----------|------|
| REINFORCE | 9.5（失敗）| 50K |
| DQN | 500（滿分）| 150K |
| PPO | 500（滿分）| 20K |
