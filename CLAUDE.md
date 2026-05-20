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

- `00_Imitation/2004_BC/` — Behavioral Cloning（程式碼完成，待執行）
- `02_Value_Based_Deep/` — DQN 系列（DDQN/PER/DuelingDQN 待重跑）
- `03_Policy_Gradient/` — REINFORCE / PPO / A2C / TRPO
- `04_Actor_Critic_Continuous/` — DDPG / TD3 / SAC
- `05_Model_Based/` — DynaQ / Dreamer / MuZero / MBPO
- `06_Advanced_Specialized/` — C51 / HER / ICM / CQL / IQL / MADDPG / MAPPO
- `plots/` — 訓練曲線產生指令碼
- `COURSE_OUTLINE.md` — 機器人 RL 課程大綱（100 分鐘，7 種演算法）
- `TASK_PLAN.md` — 任務進度追蹤

## 任務進度（2026-05-18）

- [x] 任務三：COURSE_OUTLINE.md 數字修正
- [ ] 任務一：DDQN/PER/DuelingDQN 重跑（需先清 checkpoints）
- [ ] 任務二：BC 訓練（`python 00_Imitation/2004_BC/train.py`）
- [ ] 任務四：產訓練曲線圖（`python plots/generate_plots.py`）
