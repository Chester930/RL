# RL 專案 Claude Code 說明

## Python 環境

**永遠使用專案內的 venv，不要使用系統 Python。**

```
venv 路徑：C:\Users\666\Desktop\RL\venv\Scripts\python.exe
pip  路徑：C:\Users\666\Desktop\RL\venv\Scripts\pip.exe
```

執行任何 Python 指令時，用完整路徑或先啟動 venv：

```powershell
# 啟動 venv（每個新終端執行一次）
C:\Users\666\Desktop\RL\venv\Scripts\Activate.ps1

# 或直接用完整路徑執行（不需啟動）
C:\Users\666\Desktop\RL\venv\Scripts\python.exe train.py
```

## 訓練指令規則

- 所有 `python train.py` 都必須改為 venv 的 python
- 所有 `pip install` 都必須改為 venv 的 pip
- 工作目錄：`C:\Users\666\Desktop\RL`

## 已安裝套件（venv）

| 套件 | 版本 |
|---|---|
| torch | 2.11.0+cpu |
| gymnasium | 1.3.0 |
| gymnasium-robotics | 1.4.2 |
| mujoco | 3.8.1 |
| pettingzoo | 1.26.1 |
| matplotlib | 3.10.9 |
| tensorboard | 2.20.0 |
| Box2D | 2.3.10 |

CUDA 不可用（CPU 訓練）。

## 專案結構

- `00_Imitation/2004_BC/` — Behavioral Cloning ✅
- `02_Value_Based_Deep/` — DQN / DDQN / PER / DuelingDQN / A3C / Rainbow ✅
- `03_Policy_Gradient/` — REINFORCE / PPO / A2C / TRPO ✅
- `04_Actor_Critic_Continuous/` — DDPG / TD3 / SAC ✅
- `05_Model_Based/` — DynaQ / WorldModels / Dreamer / MuZero / MBPO ✅
- `06_Advanced_Specialized/` — C51 / HER / ICM / CQL / IQL / MADDPG / MAPPO ✅
- `07_Modern_RLHF/` — RLHF / DPO / GRPO ✅
- `08_Meta_RL/` — RL²（GRU + N-Armed Bandit，後半段命中率 0.418 vs 隨機 0.200）✅；MAML / PEARL 僅 README
- `09_Hierarchical_RL/` — Options（FourRooms + SMDP Q-learning，成功率 84%）✅；FeUdal / HIRO 僅 README
- `10_Safe_RL/` — PPO-Lagrangian ✅ / CPO ✅（SafePendulum，待訓練）
- `plots/` — 訓練曲線產生指令碼
- `COURSE_OUTLINE.md` — 機器人 RL 課程大綱（100 分鐘，7 種演算法）
- `TASK_PLAN.md` — 任務進度追蹤（詳細）

## 全專案程式碼慣例（2026-05-26 起）

所有 `train.py` 均已套用以下標準：
- **隨機種子**：`config["seed"] = 42`，train() 開頭設定 `random/np/torch` 種子
- **最佳 checkpoint**：eval 超過歷史最佳時自動存 `checkpoints/best/`
- **NaN 偵測**：`agent.update()` 後檢查 loss，NaN 時立即 raise RuntimeError
- **崩潰警告**：eval 低於歷史最佳 30% 時印 `[WARNING] eval 崩潰`

## 任務進度（2026-05-27）

### 已完成（全部）

- [x] 任務一：DDQN/PER/DuelingDQN 重跑 ✅（best_checkpoints 存在）
- [x] 任務二：BC 訓練 ✅（eval=-144.6，分佈偏移測試完整）
- [x] 任務三：COURSE_OUTLINE.md 數字修正 ✅
- [x] 任務四：訓練曲線圖 ✅（8 張 PNG）
- [x] 任務五：Marp 投影片 ✅（course_slides.pdf）
- [x] 延伸 A：RLHF / DPO / GRPO ✅
- [x] 延伸 B：HER ✅（FetchReach-v4，400–500 ep 全部 100%）
- [x] 延伸 C：MADDPG ✅（50k ep，峰值 -2.04）
- [x] 任務 E：品質不佳演算法重跑 ✅（TRPO / Dreamer / MBPO）
- [x] 任務 F P1：C51 支撐集修正（0/500）✅ 最終 eval 500.0
- [x] 任務 F P2：seed / best-ckpt / training_log 特有指標 ✅
- [x] 任務 F P3 部分：RLHF 合成資料說明 + NaN 偵測 ✅
- [x] F-9：RL²（GRU + N-Armed Bandit）訓練完成 ✅（後半段命中率 0.418 vs 隨機 0.200）
- [x] F-10：Options（FourRooms SMDP Q-learning）訓練完成 ✅（成功率 84%，Flat Q 100%）
- [x] F-11：PPO-Lagrangian + CPO（SafePendulum）程式碼完成 ✅（smoke test 通過，待長訓練）

### 待完成（長訓練 + training_log）

- [ ] R-4：MuZero 重跑（程式已修正，待執行 ~4 小時）
- [ ] PPO-Lagrangian 訓練 + training_log.md（~4–5 小時）
- [ ] CPO 訓練 + training_log.md（~4–5 小時）
