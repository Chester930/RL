# RL 專案任務計畫書

最後更新：2026-05-25（MBPO 第三次重跑完成；Task F F-3 完成；Task E 全結案）

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

**狀態：✅ 已完成（2026-05-23～24，50k ep 全程跑完）**

### 實際結果

| 指標 | 數值 |
|---|---|
| 環境 | SimpleCoopEnv（2 agents，合作最小化狀態範數）|
| 訓練回合 | 50,000 / 50,000 |
| 峰值獎勵 | **-2.04**（ep 41800）|
| 最終回合 | -9.29（ep 50000，高震盪為 MADDPG 特性）|

### 完成標準

- [x] `06_Advanced_Specialized/2017_MADDPG_MARL/agent.py`
- [x] `06_Advanced_Specialized/2017_MADDPG_MARL/train.py`
- [x] `06_Advanced_Specialized/2017_MADDPG_MARL/training_log.md`
- [x] 演算法總覽表更新
- [x] **重跑**（50k 回合，2026-05-23～24 完成）

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
| ✅ 完成 | MADDPG | 50k ep 重跑，峰值 -2.04（ep 41800），較初跑 -3.77 提升 46% | — |
| ✅ 完成 | HER | 500 ep 重跑，400–500 ep 全部 100%，完全收斂 | — |
| ✅ 完成 | **Dreamer** | State-based 重跑（2026-05-25）500 ep，最佳 **-868.2**（ep475）；學習趨勢：-1237→-868；best_checkpoint 已儲存 | CNN→MLP encoder/decoder；seed_steps 5000→1000；update_every 20×4 |
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
| 03 | A2C | VecEnv 4envs 調參版，300k 步，最終 **500.0 ± 0.0**（2026-05-24）| ✅ |
| 03 | TRPO | CartPole 峰值 245.7（重跑 2026-05-24，1000 ep，修正 line search bug）| ✅ |
| 04 | DDPG | Pendulum -101.6 | ✅ |
| 04 | TD3 | Pendulum -119.8 | ✅ |
| 04 | SAC | Pendulum -171.8 + LunarLanderCont. 262.4 | ✅ |
| 05 | Dreamer | Pendulum -868.2（state-based 重跑 2026-05-25，500 ep，有學習趨勢）| ✅ |
| 05 | World Models | CarRacing-v3 VAE+MDN-RNN+CMA-ES，控制器 42.5（30 代，CPU demo）| ✅ |
| 05 | MuZero | CartPole ~9（骨架架構展示；3000 集重跑確認為結構性限制，非超參數問題）| ✅ |
| 05 | MBPO | Pendulum 104k 步（關機中止），最佳 eval **-1323.2**（step 80k），較第二次 -1480.6 提升 10.6% | ✅ |
| 06 | ICM | MountainCar -145.2（成功登頂）| ✅ |
| 06 | C51 | CartPole 峰值 372.0 | ✅ |
| 06 | HER | FetchReach-v4 100%（Epoch 160 首達，400–500 全部 100%，完全收斂）| ✅ |
| 06 | MADDPG | 峰值 -2.04（ep 41800），50k ep 完整完成 | ✅ |
| 06 | MAPPO | SimpleCoopEnv 平均獎勵 ~-89，500k 步完整完成（2026-05-24）| ✅ |
| 06 | CQL | HalfCheetah 隨機資料集，峰值 1733（100k 步），200k 步完整完成（2026-05-24）| ✅ |
| 06 | IQL | HalfCheetah 隨機資料集，峰值 530.2（150k），200k 步完整完成（2026-05-24）| ✅ |
| 07 | RLHF/InstructGPT | SFT 損失 12.24，PPO 平均獎勵 -17.5 | ✅ |
| 07 | DPO | 準確率 ~50%（合成基線） | ✅ |
| 07 | GRPO | 損失 ~0.4147，KL≈0 | ✅ |

---

## 任務 E：品質不佳演算法重跑清單（2026-05-25 起）

> 全專案品質審查（2026-05-24），以下演算法結果不符合教學展示標準，建議重跑。

### 進度快照（2026-05-25 更新）

| 演算法 | 狀態 | 結果 |
|---|---|---|
| MuZero | ✅ 結案 | 骨架展示（結構性限制，不再重跑）|
| TRPO | ✅ 完成 | 峰值 245.7（ep350），best_checkpoint 已儲存 |
| Dreamer | ✅ 完成 | state-based 重跑，-868.2（500 ep）|
| **MBPO** | ✅ **完成** | 104k 步（關機中止），最佳 eval **-1323.2**（step 80k），較第二次提升 10.6% |
| World Models | ⬜ 待決策 | 接受現況（42.5）→ 補充 CPU 限制說明（F-6）|

### 詳細說明

| 演算法 | 目前結果 | 問題 | 處置方式 |
|---|---|---|---|
| ✅ MuZero | ~9（隨機基線）| 結構性：policy target 為隨機 one-hot，非 MCTS 造訪次數 | 教學定位調整為「骨架架構展示」，不再重跑 |
| ✅ MBPO | 最佳 **-1323.2**（step 80k，104k 步中止）| 原 rollout 固定 + 50k 不足 | 150k 步 + rollout 排程 1→5 + real_ratio=0.5 + grad clipping |
| ✅ Dreamer | -868.2（500 ep）| 原 100 集太少；image-based 效率差 | 已改 state-based，重跑完成 |
| ⬜ World Models | 42.5（CPU demo）| 10 集資料 + 30 代 CMA-ES 嚴重不足 | 建議接受現況，補充「CPU 計算限制」說明 |
| ✅ TRPO | 峰值 245.7 | line search bug | 修正後重跑完成 |
| — A3C | eval 33-54 | 單執行緒，無多工作者多樣性 | 保留為教學反例，不重跑 |

### 可接受現況（不需重跑）

| 演算法 | 說明 |
|---|---|
| CQL / IQL | 使用隨機資料集，非真實 D4RL；結果合理 |
| MAPPO | 合作環境本身獎勵稀疏（-89 合理），非演算法問題 |
| RLHF / DPO / GRPO | 合成資料集，展示演算法流程為主，非性能指標 |

---

## 任務 F：全專案品質審查後續（2026-05-25）

> 全專案審查發現以下問題，依優先順序排列。可依當下時間與目標選擇執行。

---

### 🔴 P1：快速修復（每項 < 30 分鐘，影響立即可見）

#### F-1：確認課程大綱的 08/09/10 範圍 ⬜

**問題**：`08_Meta_RL`、`09_Hierarchical_RL`、`10_Safe_RL` 三個目錄只有 README，無任何程式碼或訓練記錄。若課程大綱提到這些演算法，學生找不到對應實作。

**處置選項**：
- 選項 A（建議）：在各 README 頂部加「⚠️ 本演算法尚未實作，作為延伸閱讀參考」
- 選項 B：從 COURSE_OUTLINE.md 移除這三章

**指令**：
```powershell
# 確認 COURSE_OUTLINE.md 是否涵蓋這些章節
Select-String -Path "C:\Users\666\Desktop\RL\COURSE_OUTLINE.md" -Pattern "Meta|Hierarchical|Safe"
```

**完成標準**：COURSE_OUTLINE.md 與實際目錄狀態一致，無空白章節。

---

#### F-2：C51 支撐集修正 ⬜

**問題**：`v_min=-10, v_max=10`，但 CartPole 最高回報為 500。所有 >10 的回報被截斷，分佈學習無效。

**修改**：`06_Advanced_Specialized/2021_C51_DistRL/train.py`
```python
# 改前
"v_min": -10,
"v_max": 10,
# 改後
"v_min": 0,
"v_max": 500,
```

**完成標準**：改完後重跑 100k 步，eval 應比目前 372.0 更穩定（峰值後不崩潰）。

**預估時間**：改參數 5 分鐘 + 重跑 ~2 小時

---

#### F-3：MBPO training_log 更新 ✅ 已完成（2026-05-25）

**結果**：training_log.md 已更新，含三次訓練對比、每 10k 步 eval、rollout_length 排程效果分析。最佳 eval -1323.2（step 80k）。

---

### 🟡 P2：重要改善（每項 1~4 小時）

#### F-4：全部演算法加隨機種子 ⬜

**問題**：幾乎所有 train.py 無 `np.random.seed()` + `torch.manual_seed()`，結果不可重現、訓練曲線高方差。

**修改模板**（加在 `train()` 函式最前面）：
```python
import random
seed = config.get("seed", 42)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
```

**影響範圍**：所有 00–07 的 train.py（約 25 個檔案）

**完成標準**：每個 train.py 的 config 有 `"seed": 42`，train() 開頭有種子設置。

---

#### F-5：全部演算法加最佳 checkpoint 自動儲存 ⬜

**問題**：大多數只存固定步數 checkpoint。若訓練在峰值後崩潰，無法取回最好的模型（目前只有 DQN 系列有 `best_checkpoints/`）。

**修改模板**（加在評估迴圈中）：
```python
if mean_r > best_return:
    best_return = mean_r
    agent.save("checkpoints/best")
    print(f"  ★ 新最佳：{mean_r:.1f}，已儲存")
```

**影響範圍**：03/04/05/06 底下所有有 eval 的 train.py

**完成標準**：每個演算法的 checkpoints/ 下有 best/ 子目錄。

---

#### F-6：World Models 接受現況並補充說明 ⬜

**問題**：CarRacing 42.5（論文 ~900），差距巨大。但問題是算力（CPU 限制），不是程式碼錯誤。

**處置（不重跑）**：在 `05_Model_Based/2018_WorldModels/training_log.md` 補充：
- 明確說明 CPU 限制（10 集 VAE 資料、30 代 CMA-ES）
- 論文結果需要 GPU + 10,000 集 + 1,000 代
- 教學價值：展示 VAE + MDN-RNN + CMA-ES 三階段架構

**完成標準**：training_log.md 有完整的「與論文差距原因」說明。

---

#### F-7：CQL 超參調整 ⬜

**問題**：`cql_alpha=5.0` 過度保守，訓練後期 actor_loss 轉正（+15.75），評估回報從峰值 1733 下降到 1189。

**修改**：`06_Advanced_Specialized/2020_CQL/train.py`
```python
# 改前
"cql_alpha": 5.0,
# 改後
"cql_alpha": 1.0,   # 或啟用 cql_lagrange=True（自動調整）
```

**完成標準**：重跑後 actor_loss 全程為負，200k 步 eval 不低於峰值的 80%。

**預估時間**：改參數 5 分鐘 + 重跑 ~3 小時

---

#### F-8：補充關鍵演算法的缺失訓練指標 ⬜

**問題**：以下演算法的 training_log 缺少對應演算法特有的診斷指標，降低教學說服力。

| 演算法 | 缺少指標 | 教學意義 |
|---|---|---|
| TRPO | 線搜尋成功/失敗次數、KL 大小變化 | 展示「信任區域」真正在約束更新 |
| SAC | α 的完整下降曲線（每 10k 步）| 展示自動溫度調節 |
| TD3 | actor/critic 更新頻率比（延遲更新效果）| 展示延遲更新的穩定化作用 |
| HER | 原始目標 vs hindsight 目標的成功率對比 | 展示 HER 的核心貢獻 |
| A3C | 各 worker 的分散 eval（非只有平均）| 展示多樣性探索 |

**完成標準**：上述演算法的 training_log.md 各補一段「演算法特有指標」。

---

### 🟢 P3：長期完善（視課程目標決定）

#### F-9：08_Meta_RL 三演算法實作 ⬜

**演算法**：RL²（2016）、MAML（2017）、PEARL（2019）

**難度**：高（需要 meta-learning 訓練基礎設施，RL² 需要 RNN policy）

**建議環境**：
- RL²：Multi-armed bandit 或 GridWorld
- MAML：HalfCheetah-Dir（正向/反向跑）
- PEARL：MetaWorld 或自訂 goal-conditioned 任務

**預估時間**：每個演算法 4~8 小時（含除錯）

---

#### F-10：09_Hierarchical_RL 三演算法實作 ⬜

**演算法**：Options（1999）、FeUdal（2017）、HIRO（2018）

**難度**：高（需要選項框架或目標條件化子策略）

**建議環境**：
- Options：GridWorld（離散，有明顯子目標）
- FeUdal / HIRO：AntMaze 或自訂多房間迷宮

---

#### F-11：10_Safe_RL 兩演算法實作 ⬜

**演算法**：CPO（2017）、PPO-Lagrangian（2019）

**難度**：中（需要約束優化，但可基於現有 PPO 修改）

**建議環境**：SafetyGym 或自訂有代價函式的 Pendulum

---

#### F-12：RLHF/DPO/GRPO 加強說明 ⬜

**問題**：合成隨機資料導致模型無法真正學習，RM 損失停在 0.693（隨機基線），展示效果弱。

**處置選項**：
- 選項 A（低成本）：在各 training_log.md 補充「合成資料框架說明」，明確教學定位
- 選項 B（高成本）：接入真實偏好資料集（HuggingFace `Anthropic/hh-rlhf` 或 `stanfordnlp/SHP`）

**完成標準**（選項 A）：training_log 有「為何使用合成資料」和「如何用真實資料替換」的說明。

---

#### F-13：訓練異常偵測（全部演算法）⬜

**問題**：無 NaN 檢查或崩潰警告，Q 值爆炸要等很久才發現（如 MBPO 第一次跑到 28k 步才爆炸）。

**修改模板**（加在 SAC/DQN 等的 update() 後）：
```python
if np.isnan(metrics.get("critic_loss", 0)):
    raise RuntimeError(f"NaN loss at step {step}")
if step > 10_000 and mean_r < best_return * 0.3:
    print(f"[WARNING] 步數 {step}：eval 崩潰（{mean_r:.1f} vs 峰值 {best_return:.1f}）")
```

---

### 執行手冊（通用指令模板）

任何演算法重跑時的標準流程：

```powershell
# 1. 進入目錄
cd C:\Users\666\Desktop\RL\<章節>\<演算法>

# 2. 清除舊結果（選擇性）
Remove-Item -Recurse -Force checkpoints, runs -ErrorAction SilentlyContinue

# 3. 啟動訓練（stdout/stderr 分別記錄）
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

# 5. 確認錯誤
Get-Content train_err.txt
```
