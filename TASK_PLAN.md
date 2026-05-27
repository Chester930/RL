# RL 專案任務計畫書

最後更新：2026-05-27（全面品質審查完成；任務佇列重新排序）

---

## ★ 任務執行佇列（混合排序：重要性 × 難度 × 時間）

> 排序邏輯：教學影響高 > 容易執行 > 時間短。  
> 標籤：🕐今日下班前 ｜ 🌙今夜排隊 ｜ 🔄進行中 ｜ ✅已完成

### 🕐 今日下班前（< 30 分鐘，純文件作業）

| # | 任務 | 說明 | 預估時間 |
|---|---|---|---|
| N-1 | **MBPO training_log 補說明** ✅ | 加「CPU 環境限制」段落，防止學生誤解演算法效果 | 10 分鐘 |
| N-2 | **WorldModels training_log 補說明** ✅ | 同上，說明 GPU vs CPU 的計算差距 | 10 分鐘 |
| N-3 | **RL² training_log 更新** 🔄 | P-1 完成後填入最終最佳命中率（update 1200 已達 0.546！）| 5 分鐘 |
| N-4 | **MADDPG 超參調整** | 分析不穩定原因，修改 train.py 超參，夜跑前準備好 | 15 分鐘 |
| N-5 | **MAPPO 超參調整** | 找出無收斂原因，調整 lr / n_steps，夜跑前準備好 | 15 分鐘 |

### 🌙 今夜排隊（按優先序）

| 優先 | 任務 | 教學影響 | 難度 | 時間 | 備註 |
|---|---|:---:|:---:|---|---|
| 1 | **D-3 MuZero 重跑** | 🔴最高 | 低（程式已修正）| ~4 hr | 結果毫無意義，非跑不可 |
| 2 | **D-1 DDPG 重跑** | 🟡中 | 極低（直接跑）| ~2 hr | 恢復 checkpoint，補齊 demo 能力 |
| 3 | **D-2 TD3 重跑** | 🟡中 | 極低（直接跑）| ~2 hr | 同上 |
| 4 | **M-1 PPO-Lagrangian** | 🟡中 | 低（已 smoke test）| ~4–5 hr | 完成新演算法，填 training_log |
| 5 | **M-2 CPO** | 🟡中 | 低（已 smoke test）| ~5 hr | 完成新演算法，填 training_log |
| 6 | **Q-1 MADDPG 重跑** | 🟡中 | 中（調 lr / target_update）| ~3 hr | 峰值曾達 -2.04，穩定性可改善 |
| 7 | **Q-2 MAPPO 重跑** | 🟡中 | 中（調 lr / n_steps）| ~2 hr | 無收斂趨勢，需調參 |
| 8 | **Q-3 Dreamer 重跑** | 🟠低 | 高（CPU 限制，效果不確定）| ~4 hr | 有趨勢但改善幅度難預測 |

---

## 專案完成狀態（截至 2026-05-27）

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
| 08 Meta RL | ✅ RL²（訓練完成，後半段命中率 0.418）；MAML / PEARL 僅 README |
| 09 Hierarchical RL | ✅ Options（訓練完成，成功率 84%）；FeUdal / HIRO 僅 README |
| 10 Safe RL | ✅ PPO-Lagrangian + CPO（程式完成，待訓練） |
| 教學素材 | ✅ TEACHING_GUIDE.md；6 支 training_log 深化 |

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

**已解決**：2000 集重跑後峰值 500.0（ep 950 首達），後半段穩定收斂。原因是訓練集數不足，延長後線搜尋有足夠回合找到穩定方向。

**建議修改**（train.py）：
```python
# 延長訓練至 2000 集，並調整線搜尋的最大步數
"n_episodes": 2000,       # 原 1000
"max_kl":     0.01,       # 維持不變
"ls_max_iter": 15,        # 原 10，允許更多線搜尋嘗試
```

**完成標準**：2000 集內峰值達 ≥ 400，best checkpoint 存在。

---

### R-3：MBPO — Pendulum 結果偏差 ⚠️ 已盡力，結果存檔

**影響**：🟡 中（-1271.5 與最優（~-100 至 -200）相差約 6-12 倍，說服力弱）  
**重跑成本**：🔴 高（200K 步，~8 小時）  
**優先**：中低

**第四次訓練結果（2026-05-26～05-27，完整 200K 步）**：

| Step | Eval 回報 |
|---|---|
| 130K（resume 起點） | **-1271.5**（全程峰值） |
| 140K | -1340.2 |
| 150K | -1412.1 |
| 160K | -1511.2 |
| 170K | -1545.4 |
| 180K | -1493.2 |
| 190K | -1451.5 |
| 200K（終點） | -1412.0 |

**結論**：未達完成標準（≥ -500）。  
- 峰值 **-1271.5**，比目標 -500 差約 2.5 倍
- Resume 後 real_buffer 重填導致後半段持續下滑，未能超越峰值
- 四次訓練均無法突破 -1271，確認為 CPU 環境下 MBPO + Pendulum 的收斂瓶頸
- **最終存檔值：-1271.5（step 130K best checkpoint）**

**完成標準**：200K 步內峰值達 ≥ -500（比現有 -1323.2 改善 60% 以上）。  
**→ 未達標，結果已存檔，不再重跑。**

---

### R-4：MuZero — 需程式碼修正才能重跑 🔧 程式已修正，待執行

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

> **2026-05-26 更新**：程式碼已修正（agent.py policy target 改用 MCTS 造訪次數分佈，
> train.py 儲存 mcts_probs 至 game_history，num_simulations 50→100）。
> pause/resume 功能已加入（每 500 集存點）。待找時間跑 ~4 小時。

---

### ✅ Safe RL 實作完成（2026-05-27）

**演算法**：PPO-Lagrangian（2019）+ CPO（2017）  
**環境**：SafePendulum（|θ̇| > 2.0 → cost=1，每集限 25 步違規）

| 項目 | 狀態 |
|---|---|
| `agent.py` | ✅ 完成（含 pause/resume） |
| `train.py` | ✅ 完成（記錄 reward + cost 雙指標） |
| `training_log.md` | ⏳ 待訓練後填寫（~4–5 小時） |

---

### ✅ Meta RL 訓練完成（2026-05-27）

**演算法**：RL²（GRU + N-Armed Bandit）

| 項目 | 狀態 |
|---|---|
| `network.py` / `agent.py` | ✅ 完成（含 pause/resume） |
| `train.py` | ✅ 完成（multi-task 採樣迴圈） |
| `training_log.md` | ✅ 填寫完成（最佳後半段命中率 0.418 vs 隨機 0.200） |

---

### ✅ Hierarchical RL 訓練完成（2026-05-27）

**演算法**：Options（FourRooms GridWorld + SMDP Q-learning）

| 項目 | 狀態 |
|---|---|
| `env.py` / `agent.py` | ✅ 完成（含 pause/resume） |
| `train.py` | ✅ 完成（Options vs Flat Q-learning 對比） |
| `training_log.md` | ✅ 填寫完成（Options 84%，Flat Q 100%） |

---

### D-1：DDPG — checkpoint 遺失 🔧 待處理

**問題**：training_log 有真實結果（-101.6 @ 100K），但 `checkpoints/` 目錄不存在，模型無法 load。  
**影響**：🟡中（無法跑 inference demo，但 log 可供教學）  
**修復**：直接重跑 ~2 小時，`agent.save()` 已實作，best checkpoint 會自動儲存。

---

### D-2：TD3 — checkpoint 遺失 🔧 待處理

**問題**：training_log 有真實結果（-119.8 @ 70K），但 `checkpoints/` 目錄不存在。  
**影響**：🟡中（同 D-1）  
**修復**：直接重跑 ~2 小時。

---

### D-3：MuZero — 結果等於隨機基線 🔧 程式已修正，待重跑（即 R-4）

**問題**：eval ~9 = 隨機基線，策略完全未收斂。  
**影響**：🔴最高（結果毫無教學價值）  
**修復**：程式已修正（policy target 改用 MCTS 造訪次數分佈），待重跑 ~4 小時。

---

### D-4：MBPO — 結果偏差（CPU 環境限制）📝 加說明代替重跑

**問題**：-1271（目標 -500），差 2.5 倍。  
**影響**：🔴高（若不說明，學生會誤解演算法本身效果差）  
**決策**：四次重跑均卡在 -1271，確認為 CPU 環境瓶頸，不再重跑。  
**修復**：在 training_log.md 加「CPU 環境限制說明」段落，釐清是硬體瓶頸非演算法問題。

---

### D-5：RL² — 命中率未達目標 🔄 P-1 重跑中

**問題**：後半段命中率 0.418（目標 ≥ 0.60）。  
**影響**：🟡中（有學到，2.1× 隨機基線，但教學展示說服力弱）  
**修復**：P-1 正在進行（update 1000→2000，ent_coef 0.05→0.02）。

---

### P-1：RL² 改善重跑 🔄 執行中

**目標**：後半段命中率從 0.418 提升至 ≥ 0.50

**改動**：
- `n_updates`: 1000 → 2000（延長訓練）
- `ent_coef`: 0.05 → 0.02（降低熵鼓勵後期剝削）
- `n_tasks_per_update`: 20 → 30（穩定梯度估計）

從 update 1000 checkpoint resume，額外訓練 1000 updates（~30 分鐘）。

---

### P-2：Options / RL² 訓練曲線圖 ✅

生成 `training_curve.png`：
- `09_Hierarchical_RL/1999_Options/training_curve.png`：Options vs Flat Q 獎勵 + 成功率
- `08_Meta_RL/2016_RL2/training_curve.png`：命中率 vs 更新次數（含隨機基線）

對應腳本：各目錄下的 `plot_training.py`

---

### P-3：COURSE_OUTLINE.md 補進階主題 ✅

新增「進階主題（第二堂課素材）」章節，涵蓋：
- Meta RL（RL²）：GRU 跨任務記憶、探索→剝削自動切換
- Hierarchical RL（Options）：時間抽象、走廊 option 語意
- Safe RL（PPO-Lag / CPO）：Lagrangian 機制、兩種方法對比

---

---

## 訓練品質問題全覽（依改善機率排序）

> 所有訓練結果不理想的演算法統一列管。
> **改善機率** = 重跑後有機率顯著改善的主觀估計。

---

### 🟢 Tier 1：幾乎確定更好（>90%）

| 編號 | 演算法 | 現況 | 問題根源 | 成本 | 狀態 |
|---|---|---|---|---|---|
| D-3 | **MuZero** | eval ~9（隨機基線）| policy target 用 one-hot（bug），已修正為 MCTS 造訪次數分佈 | ~4 hr | 🔧 待重跑 |
| D-1 | **DDPG** | 無 checkpoint | 訓練有結果（-101.6），只是沒存到檔案 | ~2 hr | 🔧 待重跑 |
| D-2 | **TD3** | 無 checkpoint | 同 DDPG（-119.8 已達到），只差存檔 | ~2 hr | 🔧 待重跑 |

---

### 🟡 Tier 2：很可能更好（70–90%）

| 編號 | 演算法 | 現況 | 問題根源 | 成本 | 狀態 |
|---|---|---|---|---|---|
| D-5 | **RL²** | 命中率 0.418（目標 ≥0.60）| ent_coef 太高、訓練集數不足，update 400 出現崩潰 | ~30 min | 🔄 P-1 重跑中 |
| Q-1 | **MADDPG** | 峰值 -2.04，最終 -9.29（差 4.5×）| 訓練高度不穩定，Critic 更新頻率 / lr 可調整；曾達到好結果代表演算法本身可行 | ~3 hr | 🔧 待評估 |
| Q-2 | **MAPPO** | -89（27% 分位），訓練結束無收斂跡象 | 可能 lr 過高或 clip_eps 需調整；MAPPO 在合作任務理應收斂 | ~2 hr | 🔧 待評估 |

---

### 🟠 Tier 3：有機會改善（40–70%）

| 編號 | 演算法 | 現況 | 問題根源 | 成本 | 狀態 |
|---|---|---|---|---|---|
| Q-3 | **Dreamer** | -868（目標 ~-200）| CPU 限制，僅 500 集；有明顯學習趨勢（-1600 → -868）；再跑 500 集可能突破 -600 | ~4 hr | 🔧 待評估 |

---

### 🔴 Tier 4：可能略有改善但根本瓶頸難突破（<40%）

| 編號 | 演算法 | 現況 | 問題根源 | 成本 | 狀態 |
|---|---|---|---|---|---|
| D-4 | **MBPO** | -1271（目標 -500）| 4 次重跑均卡在 -1271，確認為 CPU 環境下 model error 累積瓶頸 | ~8 hr | 📝 加說明替代重跑 |
| Q-4 | **WorldModels** | 42.5（目標 ~900）| CMA-ES 需 ~3 個月 CPU 時間；GPU 叢集 2 天可達；此環境根本無法在 CPU 改善 | 不切實際 | 📝 加說明替代重跑 |

---

### ⚪ Tier 5：設計上就是差（教學用途，不應重跑）

| 編號 | 演算法 | 現況 | 設計意圖 |
|---|---|---|---|
| Q-5 | **A3C** | 33–54（CartPole）| 刻意展示缺少真正多執行緒並行的代價，作為 PPO/A2C 的反例 |
| Q-6 | **REINFORCE** | 9.5（CartPole）| 刻意展示高方差、無基準線的失敗，引出 A2C/PPO |
| Q-7 | **IQL** | 314–530（HalfCheetah 隨機資料集）| 隨機離線資料集展示，需 D4RL 才能展示真實效果 |
| Q-8 | **RLHF** | RM 損失停在隨機基線 | 合成資料無信號，展示框架結構；需人類標注才能真正訓練 |
| Q-9 | **DPO** | 損失下降但無法量化品質 | 合成偏好資料，展示演算法流程；需真實偏好對 |
| Q-10 | **GRPO** | advantage 全為 0 | 固定獎勵導致組內方差=0，展示正確行為；需可驗證獎勵函數 |

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
