# RL 專案任務計畫書

最後更新：2026-05-19（全部核心任務完成）

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

## 已完成演算法總覽（課程就緒）

| 章節 | 演算法 | 結果 | 可用？|
|---|---|---|---|
| 00 | BC | eval=-144.6，分佈偏移測試完整 | ✅ |
| 01 | Q-Learning | FrozenLake 表格收斂 | ✅ |
| 02 | DQN | CartPole eval=500 | ✅ |
| 02 | DDQN | 最佳 eval=500（100K），best checkpoint | ✅ |
| 02 | PER | 最佳 eval=490（180K），best checkpoint | ✅ |
| 02 | DuelingDQN | 最佳 eval=500（多次），checkpoint | ✅ |
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

---

## 課程前確認清單

- [x] 任務一：DDQN/PER/DuelingDQN 有可用結果（best_checkpoints）
- [x] 任務二：BC 訓練完成，training_log.md 有實際數字
- [x] 任務三：COURSE_OUTLINE.md 數字已修正
- [x] 任務四：8 張 PNG 已產出（含 Q-table 爆炸 + BC vs SAC 真實數字）
- [x] 任務五：course_slides.pdf 投影片就緒
- [ ] 課堂時間預演（目標 100 分鐘，不超時）

---

## 任務五（原）：其他章節補完（低優先，非課程必需）

### 可直接跑（無需安裝依賴）
| 演算法 | 目錄 | 預計時間 |
|---|---|---|
| Dreamer | `05_Model_Based/2019_Dreamer/` | 已完成 ✅ |
| MuZero | `05_Model_Based/2019_MuZero/` | 已完成 ✅ |

### 需安裝依賴（課後延伸，非必需）
| 演算法 | 需要安裝 | 預計時間 |
|---|---|---|
| MBPO | `pip install "gymnasium[mujoco]"` | ~30 分鐘 |
| HER | `pip install "gymnasium[mujoco]" gymnasium-robotics` | ~20 分鐘 |
| CQL / IQL | MuJoCo + D4RL 資料集 | ~60 分鐘 |
| MADDPG / MAPPO | `pip install pettingzoo[mpe]` | ~30 分鐘 |
