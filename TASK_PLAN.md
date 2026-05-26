# RL 專案任務計畫書

最後更新：2026-05-26（新增重訓建議清單）

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

## 重新訓練建議清單

> 以下演算法的現有 training log 有品質問題，重跑後教學展示效果更好。
> 按「教學影響 × 重跑成本」排序，優先做影響大、成本低的項目。

---

### R-1：DuelingDQN — 快照方向錯誤 ✅

**影響**：🔴 高（課堂展示 V/A/Q 分解時，「右傾選 ←」會讓學生困惑）  
**重跑成本**：🟢 低（~1.5 小時，只需跑到 150K 步就停）  
**優先**：**最高**

**問題**：現有 checkpoint 是 300K 步最終模型（eval=190，250K 崩潰後部分恢復）。
最佳模型在 200K 步（eval=500），但 milestone checkpoint 互相覆蓋，best/ 未儲存。
最終 V/A/Q 快照顯示「桿往右傾」仍選 ←，Q 差距僅 0.53，方向判斷失效。

**修改方式**：不需改程式碼，直接重跑即可（train.py 已有 best/ 儲存機制）。

```powershell
cd C:\Users\666\Desktop\RL\02_Value_Based_Deep\2016_DuelingDQN
Remove-Item -Recurse -Force checkpoints, runs -ErrorAction SilentlyContinue
C:\Users\666\Desktop\RL\venv\Scripts\python.exe train.py
# 不需跑完 300K，eval 穩定在 400+ 後可手動中止（約 150K 步）
```

**完成標準**：`checkpoints/best/dueling_dqn.pt` 存在，V/A/Q 快照中「桿往右傾 → 選 →」。

---

### R-2：TRPO — 峰值偏低 ✅

**影響**：🟡 中（CartPole 最高 500，TRPO 只達 245.7，學生會問為什麼比 PPO 差這麼多）  
**重跑成本**：🟡 中（~2 小時，2000 集）  
**優先**：中

**問題**：1000 集重跑後峰值 245.7（ep 350），從未達到 400 以上。
CartPole 是簡單環境，TRPO 理論上可達 500，問題在線搜尋過於保守（全部拒絕時策略凍結）。

**建議修改**（train.py）：
```python
# 延長訓練至 2000 集，並調整線搜尋的最大步數
"n_episodes": 2000,       # 原 1000
"max_kl":     0.01,       # 維持不變
"ls_max_iter": 15,        # 原 10，允許更多線搜尋嘗試
```

**完成標準**：2000 集內峰值達 ≥ 400，best checkpoint 存在。

---

### R-3：MBPO — Pendulum 結果偏差 ⬜

**影響**：🟡 中（-1323.2 與最優（~-100 至 -200）相差 7–13 倍，說服力弱）  
**重跑成本**：🔴 高（~4 小時，需調參 + 200K 步）  
**優先**：中低

**問題**：三次重跑最佳 eval 為 -1323.2（step 80K，104K 中止），後續 Q 不穩定。
Model-based 方法的複合不穩定性：rollout 誤差 + real_ratio + alpha 耦合。

**建議修改**：
```python
"rollout_length_schedule": [(0, 1), (20000, 3), (40000, 5)],  # 慢速增長
"real_ratio":     0.5,    # 原 0.5（維持）
"sac_alpha":      0.2,    # 固定 alpha，移除自動調整（alpha 爆炸是主要崩潰原因）
"total_steps":    200_000, # 原 150K
"grad_clip":      1.0,    # 梯度裁剪（防止 Q 爆炸）
```

**完成標準**：200K 步內峰值達 ≥ -500（比現有 -1323.2 改善 60% 以上）。

---

### R-4：MuZero — 需程式碼修正才能重跑 ⬜

**影響**：🟡 中（目前結果 ~9 = 隨機基線，若能達到 100+ 則大幅提升展示價值）  
**重跑成本**：🔴 極高（需先修程式碼 + 重跑 ~4 小時）  
**優先**：低

**問題**：Policy target 使用 one-hot（當前隨機動作），應使用 MCTS 造訪次數分佈 π_mcts：
```
正確：policy_target = N(s,a) / Σ N(s,a')   ← MCTS 每個動作的造訪比例
錯誤：policy_target = one_hot(a)            ← 當前實作
```

**需要的程式碼修正**（`agent.py` MCTS.run()）：
- 收集每個動作的造訪次數 N(a)
- 訓練時以 N(a)/ΣN 作為 policy_target
- 同時增加 num_simulations（建議 50→100）和 n_episodes（500→3000）

**完成標準**：3000 集內 CartPole eval 達 ≥ 100。

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
