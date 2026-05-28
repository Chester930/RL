# RL 專案任務計畫書

最後更新：2026-05-28（上班時間短任務全部完成：S-1/S-2/S-3/S-4；今晚執行 MuZero v2）

---

## 專案完成狀態

| 章節 | 演算法 | 關鍵結果 | 狀態 |
|---|---|---|---|
| 00 Imitation | BC | eval=-144.6，分佈偏移測試 ✅ | ✅ |
| 01 Tabular | Q-Learning / SARSA / MC / TD(λ) / N-step / DP | — | ✅ |
| 02 Value Deep | DQN / DDQN / PER / DuelingDQN / A3C / Rainbow / C51 | DuelingDQN best 500 ✅ | ✅ |
| 03 Policy Gradient | REINFORCE / A2C / TRPO / PPO | A2C 500.0 ✅ / TRPO 500.0 ✅ | ✅ |
| 04 Actor-Critic | DDPG / TD3 / SAC | DDPG -86.1 ✅ / TD3 -117.6 ✅ / SAC -171.8 ✅ | ✅ |
| 05 Model-Based | DynaQ / WorldModels / Dreamer / MuZero / MBPO | MuZero **🌙 v2 今晚** | 🔧 |
| 06 Advanced | ICM / HER / CQL / IQL / MADDPG / MAPPO | HER 100% ✅ / CQL 2370 ✅ | ✅ |
| 07 Modern RLHF | RLHF / DPO / GRPO | 合成資料展示框架 ✅ | ✅ |
| 08 Meta RL | RL²（MAML/PEARL README） | best 命中率 0.546（2.73× 隨機）✅ | ✅ |
| 09 Hierarchical RL | Options（FeUdal/HIRO README） | 成功率 84% ✅ | ✅ |
| 10 Safe RL | PPO-Lagrangian / CPO | PPO-Lag λ→5.68 ✅ / CPO cost→6.5 ✅ | ✅ |
| 教學素材 | TEACHING_GUIDE.md + training_log | 全章節完整 ✅ | ✅ |

---

## 🌙 今晚待辦

### MuZero v2（🔴 唯一必做）

**目標**：CartPole eval ≥ 100（現況 ~9 = 隨機基線）  
**時間**：~6 小時  
**修正**：policy target 改 MCTS 造訪次數分佈 + num_simulations 100 + 5000 ep（已完成）

```powershell
$proc = Start-Process `
    -FilePath "C:\Users\666\Desktop\RL\venv\Scripts\python.exe" `
    -ArgumentList "-u", "train.py" `
    -WorkingDirectory "C:\Users\666\Desktop\RL\05_Model_Based\2019_MuZero" `
    -RedirectStandardOutput "C:\Users\666\Desktop\RL\05_Model_Based\2019_MuZero\train_output.txt" `
    -RedirectStandardError  "C:\Users\666\Desktop\RL\05_Model_Based\2019_MuZero\train_err.txt" `
    -NoNewWindow -PassThru
Write-Host "MuZero PID: $($proc.Id)"
```

---

## 🕐 上班時間可做（短任務，全部 < 30 分鐘）

> 不需長時間訓練，程式碼調整或快速跑完。

### S-1：Checkpoint 路徑標準化 ✅ 完成（2026-05-28）

部分演算法 checkpoint 在非標準路徑，不影響功能但不一致：

| 演算法 | 現有路徑 | 標準路徑 |
|---|---|---|
| TRPO | `best_checkpoints/trpo.pt` | `checkpoints/best/trpo.pt` |
| SAC | `checkpoints_pendulum/sac.pt` | `checkpoints/best/sac.pt` |
| A2C | `checkpoints/a2c_vecenv_step300000/` | `checkpoints/best/a2c.pt` |

```powershell
# TRPO
$t = "C:\Users\666\Desktop\RL\03_Policy_Gradient\2015_TRPO"
New-Item -ItemType Directory -Force "$t\checkpoints\best"
Copy-Item "$t\best_checkpoints\trpo.pt" "$t\checkpoints\best\trpo.pt"

# SAC
$s = "C:\Users\666\Desktop\RL\04_Actor_Critic_Continuous\2018_SAC"
New-Item -ItemType Directory -Force "$s\checkpoints\best"
Copy-Item "$s\checkpoints_pendulum\sac.pt" "$s\checkpoints\best\sac.pt"

# A2C
$a = "C:\Users\666\Desktop\RL\03_Policy_Gradient\2016_A2C"
New-Item -ItemType Directory -Force "$a\checkpoints\best"
Copy-Item "$a\checkpoints\a2c_vecenv_step300000\a2c.pt" "$a\checkpoints\best\a2c.pt"
```

---

### S-2：A2C train_output.txt 重新生成 ✅ 完成（2026-05-28）

重跑結果：280K 步首達 **500.0 ± 0.0**，300K 步最終 **500.0 ± 0.0**。  
`checkpoints/best/a2c.pt` 已儲存，training_log 已更新確認可重現。

---

### S-3：DynaQ 生成 train_output.txt ✅ 完成（2026-05-28）

K 值比較實驗結果：K=10 最佳（~17% 成功率），K=0/5/50 均為 0.000。  
training_log 已追加 Goldilocks 比較表。
```

---

### S-4：RLAIF 補 README（~10 分鐘）

`07_Modern_RLHF/2022_RLAIF/` 目前完全空白，其他 RLHF 演算法都有 README：

> 內容重點：說明 RLAIF 是 RLHF 的延伸（用 AI 標注代替人工），需要大型教師模型（GPT-4 等級），在 CPU + 合成資料環境下無法有意義地實作。

---

## 🔧 選做訓練（Tier 2–3，需夜跑）

| 編號 | 演算法 | 現況 | 目標 | 成本 | 機率 |
|---|---|---|---|---|---|
| Q-1 | **MADDPG** | 峰值 -2.04，最終 -9.29 | 穩定 < -3 | ~3h | 🟡 70% |
| Q-2 | **MAPPO** | -89，無收斂跡象 | < -50 | ~2h | 🟡 70% |
| Q-3 | **Dreamer** | -868 | 突破 -600 | ~4h | 🟠 50% |

---

## ⛔ 不再重跑（已結案）

| 演算法 | 原因 |
|---|---|
| MBPO | CPU 瓶頸 -1271，training_log 已說明 |
| WorldModels | CMA-ES 需 GPU |
| CPO / PPO-Lag | cost > 5，CPU + 緊約束；training_log 已說明 |

---

## ⚪ 設計上就是差（教學意圖，不重跑）

| 演算法 | 結果 | 設計意圖 |
|---|---|---|
| A3C | 33–54 | 展示缺少真正並行的代價 |
| REINFORCE | 9.5 | 展示高方差、無基準線的失敗 |
| IQL | 314–530 | 隨機離線資料集，需 D4RL |
| RLHF / DPO / GRPO | 合成資料 | 展示框架結構，需真實標注 |
| RLAIF | 未實作 | 需 GPT-4 等級教師模型（S-4 補 README）|

---

## 執行手冊

```powershell
$proc = Start-Process `
    -FilePath "C:\Users\666\Desktop\RL\venv\Scripts\python.exe" `
    -ArgumentList "-u", "train.py" `
    -WorkingDirectory "C:\Users\666\Desktop\RL\<章節>\<演算法>" `
    -RedirectStandardOutput "C:\Users\666\Desktop\RL\<章節>\<演算法>\train_output.txt" `
    -RedirectStandardError  "C:\Users\666\Desktop\RL\<章節>\<演算法>\train_err.txt" `
    -NoNewWindow -PassThru
Write-Host "PID: $($proc.Id)"
```
