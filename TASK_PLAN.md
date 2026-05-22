# RL 專案任務計畫書

最後更新：2026-05-21（新增延伸任務 A/B/C）

---

## 任務一：Value-Based Deep 三演算法重跑 ✅ 已完成

**狀態：✅ 已完成（2026-05-18~19）**

### 實際結果

| 演算法 | 最佳 eval | 步數 | checkpoint |
|---|---|---|---|
| DDQN | **500.0** | 100K | `best_checkpoints/double_dqn.pt` |
| PER | **490.1** | 180K | `best_checkpoints/per_dqn.pt` |
| DuelingDQN | **500.0** | 200K / 220K / 290K | `checkpoints/dueling_dqn.pt` |

### 修復內容

```python
"lr":            5e-4,   # 原 1e-3
"target_update": 200,    # 原 500
"per_alpha":     0.4,    # PER 原 0.6（第二次重跑修正，峰值從 99 → 490）
```

### 注意事項

- 三個演算法均在 100K~200K 步達到峰值後崩潰（CartPole 長訓練的已知不穩定性）
- **課程展示請使用 `best_checkpoints/` 而非 `checkpoints/`**
- DuelingDQN 未重跑，原始結果已足夠（多次達 500）

---

## 任務二：Behavioral Cloning 實作 ✅ 已完成

**狀態：✅ 已完成（2026-05-18）**

### 實際結果

| 指標 | 數值 |
|---|---|
| BC 最終 eval | -144.6 ± 67.9（100 epochs）|
| Expert（SAC）eval | -170.3 ± 85.6 |
| demos.npz | 50 集 × 200 步 = 10,000 transitions |

### 分佈偏移測試結果

| 角度 | SAC Expert | BC Policy | 狀態 |
|---|---|---|---|
| 0° – 90° | -123 附近 | -123 附近 | ✅ 分佈內 |
| 120° | -122.2 | **-245.0** | ❌ 崩潰 |
| 180° | -353.5 | **-436.5** | ❌ 崩潰 |

### 已完成清單

- [x] `network.py` / `agent.py` / `sac_expert.py` / `collect_demos.py` / `train.py`
- [x] `README.md`
- [x] `training_log.md`（含實際訓練數字與分佈偏移表格）
- [x] 圖 7（`plots/07_bc_vs_sac.png`）已填入真實數字

---

## 任務三：修正課程大綱錯誤 ✅ 已完成

**狀態：✅ 已完成（2026-05-18）**

COURSE_OUTLINE.md 中 SAC 與 TD3 的 Pendulum 數字已修正，
並補充「SAC 的優勢在複雜任務才顯現」的教學說明。

---

## 任務四：訓練曲線視覺化 ✅ 已完成

**狀態：✅ 已完成（2026-05-19）**

### 產出圖表（共 8 張）

| 檔案 | 內容 | 教學用途 |
|---|---|---|
| `00_qtable_explosion.png` | Q-table 大小隨關節數指數爆炸 | 引出 DQN 的必要性 |
| `01_reinforce_failure.png` | REINFORCE eval 停在 9-10 | 對比 PPO |
| `02_dqn_learning.png` | DQN eval 曲線 + Q 值高估 | Replay + Target Net 效果 |
| `03_ddpg_instability.png` | DDPG 振盪曲線 | 引出 TD3 動機 |
| `04_td3_vs_ddpg.png` | TD3 vs DDPG 雙線對比 | Twin Critic 穩定性 |
| `05_sac_alpha.png` | SAC α 自動下降 + eval | 自動溫度調節展示 |
| `06_reinforce_vs_ppo.png` | REINFORCE vs PPO 對比 | Clip 效果一目瞭然 |
| `07_bc_vs_sac.png` | BC vs SAC 分佈偏移（真實數字）| Distribution Shift 視覺衝擊 |

---

## 任務五：Marp 投影片 ✅ 已完成（新增）

**狀態：✅ 已完成（2026-05-19）**

| 檔案 | 說明 |
|---|---|
| `course_slides.md` | Marp 原始檔（38 張投影片）|
| `course_slides.pdf` | 課堂用 PDF（1.1MB）|

重新產出指令：
```bash
cd C:\Users\666\Desktop\RL
npx @marp-team/marp-cli course_slides.md --pdf --output course_slides.pdf --allow-local-files
```

---

## 課程前確認清單 ✅ 全部完成

- [x] 任務一：DDQN/PER/DuelingDQN 有可用結果（best_checkpoints）
- [x] 任務二：BC 訓練完成，training_log.md 有實際數字
- [x] 任務三：COURSE_OUTLINE.md 數字已修正
- [x] 任務四：8 張 PNG 已產出（含 Q-table 爆炸 + BC vs SAC 真實數字）
- [x] 任務五：course_slides.pdf 投影片就緒
- [x] 課堂時間預演（目標 100 分鐘，不超時）

---

## 延伸任務 A：07_Modern_RLHF 三支程式執行 ✅ 已完成

**狀態：✅ 已完成（2026-05-22）**

### 實際結果

| 演算法 | 關鍵指標 | checkpoint |
|---|---|---|
| RLHF InstructGPT | SFT 損失 20.09→12.24，RM ~0.693，PPO 平均獎勵 ~-17.5 | `checkpoints/rlhf/rlhf_{sft,rm,ppo}.pt` |
| DPO | 損失 2.1–6.9 震盪，準確率 ~50%（合成隨機基線） | `checkpoints/dpo_step{500,1000}` |
| GRPO | 損失穩定 ~0.4147，KL≈0，獎勵固定 0.1（合成無信號） | `checkpoints/grpo_step{250,500}` |

### 完成標準

- [x] 三支均執行完畢無錯誤
- [x] 為每支寫 `training_log.md`
- [ ] 演算法總覽表更新（本文末）

---

## 延伸任務 B：HER 實作（Hindsight Experience Replay） ✅ 已完成

**狀態：✅ 已完成（2026-05-22）**

### 背景

HER（Andrychowicz et al., 2017）解決**稀疏獎勵**問題：
把「失敗的軌跡」重新標記目標（achieved_goal → new goal），
讓 DDPG 在機器手臂環境中學會 FetchReach。

課程延伸價值：從 ICM 的「主動探索」過渡到「重新利用失敗經驗」。

### 環境

```python
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
env = gym.make("FetchReach-v3")  # obs 含 achieved_goal / desired_goal
```

### 實作要點

- `agent.py`：DDPG + HER replay buffer（goal concatenation 輸入）
- `train.py`：每集結束後 hindsight relabeling（k=4 個替代目標）
- 評估指標：success_rate（距目標 ≤ 5cm 視為成功）

### 實際結果

| 指標 | 數值 |
|---|---|
| 環境 | FetchReach-v4（obs=10, goal=3, act=4）|
| 峰值成功率 | **80%**（Epoch 170 & 180）|
| 收斂狀況 | Epoch 90 起開始學習，130+ 突破 60% |

### 完成標準

- [x] `06_Advanced_Specialized/2017_HER/agent.py`
- [x] `06_Advanced_Specialized/2017_HER/train.py`
- [x] `06_Advanced_Specialized/2017_HER/training_log.md`（success_rate 曲線）
- [ ] 演算法總覽表更新

---

## 延伸任務 C：MADDPG 實作（多智能體） ✅ 部分完成

**狀態：⚠️ 部分完成（2026-05-22，訓練至 ep 8800/10000 後因時間不足停止）**

### 實際結果

| 指標 | 數值 |
|---|---|
| 環境 | SimpleCoopEnv（2 agents，合作最小化狀態範數）|
| 訓練回合 | 8,800 / 10,000 |
| 峰值獎勵 | **-3.77**（ep 3600）|
| 最終均值 | ~-9.0（震盪，未完全收斂）|

### 完成標準

- [x] `06_Advanced_Specialized/2017_MADDPG_MARL/agent.py`
- [x] `06_Advanced_Specialized/2017_MADDPG_MARL/train.py`
- [x] `06_Advanced_Specialized/2017_MADDPG_MARL/training_log.md`
- [x] 演算法總覽表更新
- [ ] **重跑**（50k 回合，詳見任務 D）

---

## 任務時間分配（2-3 小時）

```
[00:00 - 00:30]  任務 A：跑 RLHF / DPO / GRPO（代碼就緒，直接執行）
[00:30 - 01:45]  任務 B：實作 HER（重點任務，gymnasium-robotics 已裝）
[01:45 - 02:45]  任務 C：安裝 pettingzoo[mpe] + 實作 MADDPG
[02:45 - 03:00]  更新演算法總覽表 + git commit
```

**優先順序：A > B > C**（C 時間不夠可跳過）

---

## 下次任務（待執行）

### 任務 D：重跑建議清單 ⬜

根據本次訓練結果，以下演算法建議重跑：

| 優先 | 演算法 | 原因 | 建議修改 |
|---|---|---|---|
| 🔴 高 | MADDPG | 8800 ep 未收斂，震盪大 | episodes: 10k → **50k**；考慮換 `simple_spread_v3` |
| 🔴 高 | HER | ep 190–200 成功率回落（30%/20%），未穩定 | epochs: 200 → **500**；加 eval 滾動平均 |
| 🟡 中 | Dreamer | 最終 -836 僅示範用，world model 未真正學習 | 延長至 500 ep；修正 RSSM latent dim |
| 🟡 中 | MuZero | 回報 ~9 為隨機基線，MCTS simulation 極少 | num_simulations: 提高；CartPole 跑更多集 |
| 🟢 低 | A2C | CartPole 單環境不穩定（357.5） | 改用 4 平行環境（VecEnv）|

### 執行指令（任務 D）

```powershell
# MADDPG 50k 回合（修改 train.py total_episodes=50000）
cd C:\Users\666\Desktop\RL\06_Advanced_Specialized\2017_MADDPG_MARL
C:\Users\666\Desktop\RL\venv\Scripts\python.exe train.py

# HER 500 epochs
cd C:\Users\666\Desktop\RL\06_Advanced_Specialized\2017_HER
# 先修改 train.py n_epochs=500，再執行
C:\Users\666\Desktop\RL\venv\Scripts\python.exe train.py
```

**預估時間**：MADDPG ~4.5 小時 / HER ~45 分鐘（可先跑 HER）

---

## 已完成演算法總覽

| 章節 | 演算法 | 結果 | 狀態 |
|---|---|---|---|
| 00 | BC | eval=-144.6，分佈偏移測試完整 | ✅ |
| 01 | Q-Learning | FrozenLake 表格收斂 | ✅ |
| 02 | DQN | CartPole eval=500 | ✅ |
| 02 | DDQN | 最佳 eval=500（100K） | ✅ |
| 02 | PER | 最佳 eval=490（180K） | ✅ |
| 02 | DuelingDQN | 最佳 eval=500（多次） | ✅ |
| 03 | REINFORCE | eval=9.5（高方差，符合教學目的）| ✅ |
| 03 | PPO | CartPole 500 + LunarLander 283.9 | ✅ |
| 03 | A2C | CartPole 峰值 357.5 | ✅ |
| 03 | TRPO | CartPole 峰值 433.0 | ✅ |
| 04 | DDPG | Pendulum -101.6 | ✅ |
| 04 | TD3 | Pendulum -119.8 | ✅ |
| 04 | SAC | Pendulum -171.8 + LunarLanderCont. 262.4 | ✅ |
| 05 | Dreamer | Pendulum -836（示範用）| ✅ |
| 05 | MuZero | CartPole ~9（MCTS 展示用）| ✅ |
| 06 | ICM | MountainCar -145.2（成功登頂）| ✅ |
| 06 | C51 | CartPole 峰值 372.0 | ✅ |
| 06 | HER | FetchReach-v4 峰值成功率 80%（Epoch 170）| ✅ |
| 06 | MADDPG | 峰值 -3.77（ep 3600），均值 -9（8800 ep，未完全收斂）| ⚠️ 需重跑 |
| 07 | RLHF/InstructGPT | SFT 損失 12.24，PPO 平均獎勵 -17.5 | ✅ |
| 07 | DPO | 準確率 ~50%（合成基線） | ✅ |
| 07 | GRPO | 損失 ~0.4147，KL≈0 | ✅ |
