# RL 專案任務計畫書

最後更新：2026-05-26（教學素材完善；全部主線任務完成）

---

## 專案完成狀態（截至 2026-05-26）

| 章節 | 完成狀態 |
|---|---|
| 00 Imitation | ✅ BC（eval=-144.6，分佈偏移測試） |
| 01 Tabular | ✅ Q-Learning / SARSA / MC / TD(λ) / N-step / DP / DynaQ |
| 02 Value Deep | ✅ DQN / DDQN / PER / DuelingDQN / A3C / Rainbow / C51 |
| 03 Policy Gradient | ✅ REINFORCE / A2C / TRPO / PPO |
| 04 Actor-Critic | ✅ DDPG / TD3 / SAC |
| 05 Model-Based | ✅ DynaQ / WorldModels / Dreamer / MuZero / MBPO |
| 06 Advanced | ✅ ICM / HER / CQL / IQL / MADDPG / MAPPO |
| 07 Modern RLHF | ✅ RLHF/InstructGPT / DPO / GRPO |
| 教學素材 | ✅ TEACHING_GUIDE.md 建立；6 支 training_log 深化（PPO/DDPG/REINFORCE/DPO/GRPO/MADDPG）|

---

## 待辦任務

> 以下均為「延後，不影響課程教學」的可選任務，依重要性排序。

---

### 🟡 選項一：Safe RL 實作（建議優先）

**優先**：中（概念課程相關度高，且改動量小）  
**預估時間**：6–10 小時（含除錯）  
**演算法**：CPO（2017）、PPO-Lagrangian（2019）

```
CPO：約束策略最佳化，同時最大化累積獎勵並滿足安全約束
PPO-Lagrangian：PPO + 拉格朗日乘數法管理約束違反
```

**建議環境**：
- 自訂 Pendulum + 代價函式（角速度 > 閾值視為不安全）
- 或 `Safety-Gymnasium`（需額外安裝 `pip install safety-gymnasium`）

**目錄**：`10_Safe_RL/`

**完成標準**：
- [ ] `agent.py`（PPO-Lagrangian 或 CPO 其中之一即可）
- [ ] `train.py`（記錄 cost_return 與 reward_return 雙指標）
- [ ] `training_log.md`（展示「安全約束 vs 最大化獎勵」的 trade-off 曲線）

---

### 🟠 選項二：Meta RL 實作

**優先**：中低（概念進階，與現有課程連接較遠）  
**預估時間**：12–24 小時（每個演算法 4–8 小時）  
**演算法**：RL²（2016）、MAML（2017）、PEARL（2019）

**建議環境**：
- RL²：Multi-armed bandit 或 GridWorld（RNN policy 學跨任務適應）
- MAML：HalfCheetah-Dir（正向/反向跑，僅需幾步梯度就能遷移）
- PEARL：MetaWorld 或自訂 goal-conditioned 任務

**目錄**：`08_Meta_RL/`

**完成標準**（可只做 RL² 其一）：
- [ ] `agent.py`（RL² 最容易起步，在現有 PPO 上加 RNN 即可）
- [ ] `train.py`（multi-task 採樣迴圈）
- [ ] `training_log.md`（展示「seen tasks vs unseen tasks 的遷移效果」）

---

### 🟢 選項三：Hierarchical RL 實作

**優先**：低（概念最複雜，實作門檻最高）  
**預估時間**：16–24 小時（每個演算法 4–8 小時）  
**演算法**：Options（1999）、FeUdal（2017）、HIRO（2018）

**建議環境**：
- Options：GridWorld（離散，有明顯子目標）
- FeUdal / HIRO：AntMaze 或自訂多房間迷宮

**目錄**：`09_Hierarchical_RL/`

**完成標準**（可只做 Options 其一）：
- [ ] `agent.py`（高層策略選 option，低層策略執行）
- [ ] `train.py`
- [ ] `training_log.md`（展示 option 的切換時機）

---

## 執行手冊（通用指令模板）

任何演算法訓練時的標準流程：

```powershell
# 1. 進入目錄
cd C:\Users\666\Desktop\RL\<章節>\<演算法>

# 2. 清除舊結果（選擇性）
Remove-Item -Recurse -Force checkpoints, runs -ErrorAction SilentlyContinue

# 3. 啟動訓練
$proc = Start-Process `
    -FilePath "C:\Users\666\Desktop\RL\venv\Scripts\python.exe" `
    -ArgumentList "-u", "train.py" `
    -WorkingDirectory (Get-Location).Path `
    -RedirectStandardOutput "train_output.txt" `
    -RedirectStandardError  "train_err.txt" `
    -NoNewWindow -PassThru
Write-Host "PID: $($proc.Id)"

# 4. 監看即時輸出
Get-Content train_output.txt -Wait -Tail 5
```

**venv Python 路徑**：`C:\Users\666\Desktop\RL\venv\Scripts\python.exe`
