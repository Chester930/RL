# RL 課程教學指南

**適用對象**：兩種學生並行
- 初學者（第一次接觸 RL）— 需要公式推導 + 直覺建立
- 有基礎者（學過 ML，初學 RL）— 跳過 ML 概念，聚焦 RL 特有機制

---

## 一、學習路線圖

```
[01 表格式基礎]
  DP → MC → TD(λ) → Q-Learning → SARSA
       ↓
[02 Value-Based Deep]               關鍵問題：「如何從表格 Q 跨越到神經網路 Q？」
  DQN → DoubleDQN → PER → DuelingDQN → A3C → Rainbow
       ↓
[03 Policy Gradient]                關鍵問題：「為什麼不直接最佳化策略？」
  REINFORCE → A2C → TRPO → PPO
       ↓
[04 Actor-Critic 連續]              關鍵問題：「如何處理連續動作空間？」
  DDPG → TD3 → SAC
       ↓
[05 Model-Based]                    關鍵問題：「能不能讓模型學世界的規律？」
  DynaQ → WorldModels → Dreamer → MuZero → MBPO
       ↓
[06 進階專題]                       各自獨立，可依時間選擇
  C51 / HER / ICM / CQL / IQL / MADDPG / MAPPO
       ↓
[07 Modern RLHF]                    關鍵問題：「如何讓 LLM 對齊人類偏好？」
  RLHF/InstructGPT → DPO → GRPO
```

---

## 二、如何帶學生看 Training Log

### 通用看法順序

1. **配置表**：先確認環境 + 關鍵超參（學習率、折扣因子、buffer 大小）
2. **第一次更新**：看初始 loss/Q 值，讓學生預測為什麼是這個數字
3. **訓練曲線**：找三個時刻——「尚未學習」「快速進步」「收斂/崩潰」
4. **演算法特有指標**：每個演算法都有不同的「靈魂指標」（見下方各算法）
5. **與前後演算法對比**：強調每個算法解決了什麼前算法的問題

### 互動技巧

- 在里程碑時刻停下來問：「你覺得 loss 接下來會怎麼變？」
- 讓學生手算一個具體步驟（見各演算法的「課堂計算」）
- 使用日誌裡的「失敗案例」（A3C 崩潰、MBPO Q 爆炸）當反例

---

## 三、各演算法教學指引

---

### 01 表格式基礎

#### Q-Learning（1989）
📁 `01_Tabular_Basics/1989_QLearning/training_log.md`

**靈魂指標**：Q-table 演化（各格子的 Q 值如何從 0 逐漸傳播）

**教學流程**：
1. 從 FrozenLake 地圖開始，問學生「如果完全不知道獎勵，要怎麼找到終點？」
2. 展示第 52 集的逐步軌跡（training_log 第 108–163 行）
3. **課堂計算**（最關鍵）：帶著學生做步驟 21 的 TD 更新
   ```
   Q(14, ↑) 從 0.0 → 0.1
   δ = r + γ · max Q(s') − Q(s,a) = 1 + 0.99×0 − 0 = 1.0
   ΔQ = α × δ = 0.1 × 1.0 = 0.1
   ```
4. 展示 5000/10000/20000 集的 Q-table，讓學生觀察「價值如何從終點往起點傳播」
5. 問：「為什麼 Q 值最大才 0.88，不是 1.0？」→ 引出折扣因子概念

**銜接 DQN 的橋梁**：「CartPole 的狀態是連續的 4 維向量，FrozenLake 的表格裝不下了」

---

#### MC（蒙特卡羅）vs TD（時序差分）
📁 `01_Tabular_Basics/1980s_MC/` 和 `01_Tabular_Basics/1988_TD_Lambda/`

**核心對比**：
| 方法 | 更新時機 | 偏差 | 方差 |
|---|---|---|---|
| MC | 集數結束後 | 零偏差 | 高方差 |
| TD(0) | 每步 | 有偏差（仰賴估計）| 低方差 |
| TD(λ) | n 步折衷 | 可調 | 可調 |

**課堂計算**：給一個 5 步軌跡，讓學生同時計算 MC 和 TD(0) 的更新，比較差異

---

### 02 Value-Based Deep

#### DQN（2013）
📁 `02_Value_Based_Deep/2013_DQN/training_log.md`

**靈魂指標**：mean Q 的高估演化 vs eval 回報的實際演化

**教學流程**：
1. 展示第 1000 步的第一次更新（training_log 第 93–121 行）
2. **課堂計算**（核心）：為什麼初始 loss ≈ 0.5？
   ```
   y = r + γ × max Q_target(s') ≈ 1.0 + 0 = 1.0
   Q_online(s, a) ≈ 0.0
   Huber(1.0 - 0.0) = 0.5² / 2 = 0.5
   ```
3. 展示三個里程碑（50K/100K/150K）的對比表，重點討論：
   - mean Q 從 95 → 313 → 224（高估問題）
   - 理論最大 Q ≈ 99.3，實際 mean Q 卻到 313（3倍高估）
4. 展示四個典型狀態的 Q 值快照（訓練結束後），讓學生判斷策略是否合理

**銜接 DoubleDQN**：「mean Q 高達 313，理論只有 99——這就是要解決的問題」

---

#### DoubleDQN（2015）
📁 `02_Value_Based_Deep/2015_DoubleDQN/training_log.md`

**靈魂指標**：DQN 目標 vs DDQN 目標的差值（高估量）

**課堂計算**：展示「選動作」和「評估動作」分離的邏輯
```
DQN：   a* = argmax Q_target(s')  ← 同一網路選 + 評估 → 選到高估的動作
DDQN：  a* = argmax Q_online(s')  ← online 選動作
        y  = Q_target(s', a*)     ← target 評估 → 兩個獨立誤差難以疊加
```

**注意**：DQN-DDQN 目標差值在 log 裡顯示 0.0000——CartPole 太簡單，高估差異不明顯。
告訴學生：「要看到 DDQN 真正的優勢，需要 Atari 等複雜環境」

---

#### PER（2015）
📁 `02_Value_Based_Deep/2015_PER/training_log.md`

**靈魂指標**：平均 |TD 誤差| 的演化 + β 退火進度

**課堂計算**：取樣機率計算
```
P(i) ∝ |TD 誤差_i + ε|^α    （α=0.4）

若 TD 誤差_A = 5.0，TD 誤差_B = 0.2：
  P(A) ∝ 5.0^0.4 ≈ 1.90
  P(B) ∝ 0.2^0.4 ≈ 0.56
  P(A) / P(B) ≈ 3.4×   ← A 被取樣頻率是 B 的 3.4 倍
```

β 退火圖：展示 β 從 0.4（50K 步）→ 0.99（150K 步）→ 1.0（200K 步）的重要性修正進度

---

#### A3C（2016）—— 教學反例
📁 `02_Value_Based_Deep/2016_A3C/training_log.md`

**靈魂指標**：評論家損失爆炸（11M → 106M）+ 策略熵崩潰（0.69 → 0.009）

**教學定位**：這是一個「刻意保留的失敗案例」

**課堂展示**：
1. 展示 50K 步的評論家損失 11,167,526（震撼學生）
2. 展示策略最終「永遠推左」（P(←)=1.0，P(→)=0.0）
3. 問：「為什麼增加平行環境能解決這個問題？」→ 展示 training_log 裡的「多工作者多樣性」分析
4. 對比 A2C VecEnv 版本（同樣演算法，4 個平行環境 → 500 滿分）

**銜接 A2C**：「差距完全來自『並行度』，不是演算法公式」

---

### 03 Policy Gradient

#### REINFORCE（1992）—— 教學反例
📁 `03_Policy_Gradient/1992_REINFORCE/training_log.md`

**靈魂指標**：5000 集後 eval 仍在 9.5（從未超過 100）

**課堂計算**：展示第 4762 集（唯一接近 100 分的集數）的 G_t 計算
```
G_0 = 66.2，G_1 = 65.9，...，G_107 = 1.0
均值 = 39.3，標準差 = 18.8

歸一化後 G_0 = (66.2 - 39.3) / 18.8 = +1.43（正優勢 → 加強此步動作）
歸一化後 G_107 = (1.0 - 39.3) / 18.8 = -2.04（負優勢 → 抑制此步動作）
```

**高方差問題（training_log「核心機制詳解」節）**：
- 展示同一動作在短集（17步）vs 長集（108步）的 advantage 差異：+2.7 vs +25.2（9.3×）
- 說明「低回報-低梯度-低回報」惡性循環（診斷 5000 集不收斂的根本原因）
- 對比表：REINFORCE（G_t，高方差）→ A2C（TD 誤差，低方差）→ PPO（GAE，折衷）

**銜接 PPO**：「REINFORCE 每集只更新 1 次、無法重複使用資料——PPO 解決了這個問題」

---

#### PPO（2017）
📁 `03_Policy_Gradient/2017_PPO/training_log.md`

**靈魂指標**：
- clip_frac（被裁剪的更新比例，約 10%）
- approx_kl（單次更新的策略差異，約 0.003–0.009）

**課堂計算**：展示三種 clip 情況（training_log「Clip 操作實際計算範例」節）
- 情況 A：好動作，ratio=2.01 → 被限制到 1.2（防過大更新）
- 情況 B：好動作，ratio=1.1 → 正常更新
- 情況 C：壞動作，ratio=0.27 → 被限制（防過度懲罰）

**可以講的計算（白板）**：
```
已知：log π_old(a|s) = -1.0，log π_new(a|s) = -0.3，A = +4.2
ratio = exp(-0.3 - (-1.0)) = exp(0.7) = 2.01
surr1 = 2.01 × 4.2 = 8.44
surr2 = clip(2.01, 0.8, 1.2) × 4.2 = 1.2 × 4.2 = 5.04
L_CLIP = min(8.44, 5.04) = 5.04（被裁剪）
```

**銜接 TRPO**：「PPO 的 clip 比 TRPO 的精確 KL 計算快 10 倍，效果媲美」

---

#### TRPO（2015）
📁 `03_Policy_Gradient/2015_TRPO/training_log.md`

**靈魂指標**：線搜尋行為 + KL 約束效果（training_log「線搜尋行為分析」節）

**教學重點**：
- ep 750 的崩潰（eval=27）→ ep 800 的自動恢復（eval=216）
- 線搜尋全部拒絕時策略凍結，等環境隨機性改變梯度方向
- 對比 PPO：TRPO 精確但慢，PPO 近似但實用

---

#### A2C（2016）
📁 `03_Policy_Gradient/2016_A2C/training_log.md`

**靈魂指標**：三次訓練版本的系統性對比（單環境 → VecEnv → 調參後 VecEnv）

**課堂展示**：展示三次訓練的決策過程
```
版本 1（單環境）：峰值 357.5 → 崩潰至 43
  → 診斷：無信任域，更新太大

版本 2（VecEnv, lr=7e-4）：峰值 260.7 → 崩潰至 14
  → 診斷：批次大 4× 但 lr 未降，等效 lr 偏高

版本 3（VecEnv, lr=2e-4, n_steps=10）：300K 步達 500 ✅
  → lr 降至 2e-4 是關鍵
```

---

### 04 Actor-Critic 連續

#### DDPG（2015）
📁 `04_Actor_Critic_Continuous/2015_DDPG/training_log.md`

**靈魂指標**：Actor Loss 的演化（負值 = 策略在改善）

**課堂計算**（核心：確定性策略梯度）：
```
DQN argmax Q：只適用離散動作（A = {left, right, up...}）

DDPG：μ_θ(s) → a（連續向量，如力矩 [-2, 2]）
actor_loss = -Q(s, μ(s))    ← 最大化 Q 就是最佳化 Actor

梯度：∂J/∂θ = ∂Q/∂a · ∂μ/∂θ
              ↑ Critic 給梯度  ↑ Actor 接梯度
```

**OU 雜訊**：展示 training_log 的雜訊軌跡，說明為什麼需要時間相關探索

**銜接 TD3**：展示「三個問題 → 三個改善」的對應關係表

---

#### TD3（2018）
📁 `04_Actor_Critic_Continuous/2018_TD3/training_log.md`

**靈魂指標**：Actor vs Critic 更新頻率比（1:2）

**課堂計算**（延遲更新）：
```
200K 步訓練（learning_starts=5K）：
  Critic 更新次數 = 200,000 - 5,000 = 195,000 次
  Actor 更新次數 = 195,000 / 2 = 97,500 次
  
比率 1:2 的意義：
  Critic 先做 2 次更新，Q 估計更穩定後，Actor 才跟著更新
  → 防止 Actor 追跑不準確的 Q 值
```

---

#### SAC（2018）
📁 `04_Actor_Critic_Continuous/2018_SAC/training_log.md`

**靈魂指標**：α 自動溫度調節曲線（Pendulum vs LunarLander 的差異）

**課堂討論**：
- Pendulum：α 從 0.32 單調降至 0.02（簡單環境，策略快速確定化）
- LunarLander：α 在 100K 升至峰值 0.091（複雜環境需要更長的高探索期）
- 問：「為什麼同一個演算法在不同環境下 α 行為截然不同？」

---

### 05 Model-Based

#### DynaQ（1990）
📁 `05_Model_Based/1990_DynaQ/training_log.md`

**核心概念**：「真實經驗」+ 「模型生成的虛擬經驗」同時使用

**銜接 MBPO**：「DynaQ 用表格記錄模型，MBPO 用神經網路」

---

#### WorldModels（2018）
📁 `05_Model_Based/2018_WorldModels/training_log.md`

**靈魂指標**：三階段訓練結果（VAE 損失 / MDN-RNN NLL / CMA-ES 獎勵）

**教學重點**（V-M-C 架構）：
```
V（Vision）：VAE 將 64×64 畫面壓縮為 32 維 z
M（Memory）：MDN-RNN 預測下一個 z（5 個高斯混合）
C（Controller）：線性層 288D → 動作（CMA-ES 最佳化）

「在夢中學習」：完全在 M 的想像中訓練 C，不接觸真實環境
```

**解釋 42.5 vs 論文 ~900**：展示 training_log 的算力對比表（CPU 單核 vs 64 CPU 核心）

---

#### Dreamer（2019）
📁 `05_Model_Based/2019_Dreamer/training_log.md`

**靈魂指標**：RSSM 世界模型的潛在空間學習

**課堂展示**：展示 state-based 版本（-1237 → -868）的學習趨勢，解釋：
- RSSM 先驗/後驗分佈如何分離「過去記憶」和「當前觀測」
- KL 約束防止潛在空間崩潰

---

#### MuZero（2019）
📁 `05_Model_Based/2019_MuZero/training_log.md`

**教學定位**：骨架展示（不展示訓練結果，展示三網路架構）

**重點展示**：
```
h（Representation）：s → hidden_state（學習有意義的隱藏空間）
g（Dynamics）：       hidden_state + action → next_hidden + reward（世界模型）
f（Prediction）：     hidden_state → policy + value（MCTS 的評估函式）
```

**為什麼結果是隨機基線？**（展示 training_log 的根本原因分析）
1. Policy target 是隨機 one-hot，非 MCTS 造訪次數分佈
2. 表格集數太短（~9步），MCTS 無法建立有意義的搜尋樹

---

#### MBPO（2019）
📁 `05_Model_Based/2020_MBPO/training_log.md`

**靈魂指標**：三次訓練的 Q 值爆炸過程（第一次 step 28k 時 critic_loss = 10 億）

**課堂展示**：展示「Q 值爆炸的正反饋迴圈」
```
real_ratio=0.05 → 95% 來自模型緩衝區
模型誤差累積 → Q 目標高估 → alpha 爆炸至 5,615
策略隨機化 → 模型誤差更大 → 無法自我修復
```

**修復方案**：展示 rollout_length 排程（1→5 每 10k 步）+ real_ratio=0.5 的效果

---

### 06 進階專題

#### C51（2017）—— 分佈式 RL
📁 `06_Advanced_Specialized/2017_C51_DistRL/training_log.md`

**靈魂指標**：三次訓練的支撐集修正效果

**課堂展示（最直觀）**：展示 v_min/v_max 修正的效果
```
舊版（v_min=-10, v_max=10）：
  CartPole 回報 500 → 截斷，分佈學習完全無效
  51 個原子分佈在 [-10, 10]，對 0-500 的回報無意義

新版（v_min=0, v_max=500）：
  51 個原子均勻分佈 0, 10, 20, ..., 500
  ε=0.01 後首個 eval = 500（對比舊版 = 185）
```

---

#### HER（2017）—— 後見之明經驗回放
📁 `06_Advanced_Specialized/2017_HER/training_log.md`

**靈魂指標**：原始目標 vs Hindsight 目標的成功率密度

**課堂計算**（核心）：展示稀疏獎勵的機率問題
```
FetchReach 目標球半徑 5cm = 0.05m
工作空間約 0.5m × 0.5m × 0.5m = 0.125 m³

隨機成功概率 ≈ (4/3)π(0.05)³ / 0.125 ≈ 0.04%
期望：隨機策略每 2500 集才偶發 1 次正獎勵

HER：重標記後正獎勵密度從 0.04% → ~80%（每集 80% 的轉換都能得到 r=0）
```

---

#### ICM（2017）—— 內在好奇心
📁 `06_Advanced_Specialized/2017_ICM/training_log.md`

**靈魂指標**：內在獎勵在首次達頂（~800K 步）後驟升的現象

**課堂展示**：展示特徵坍縮 bug（F.normalize 導致 r_i = 0）
```
MountainCar 狀態 = 2 維（x, ẋ）
F.normalize 將所有 2D 向量投影到單位圓
→ 幾乎所有狀態都映射到相同單位向量
→ 前向模型輸出相同 → 預測誤差 ≈ 0 → r_i = 0 → 好奇心失效

修正：移除 F.normalize，保留原始特徵空間
修正後：r_i 立即上升至 1.37
```

---

#### CQL（2020）—— 保守離線 RL
📁 `06_Advanced_Specialized/2020_CQL_Offline/training_log.md`

**靈魂指標**：actor_loss 從負轉正的時機（100K 步）

**課堂展示**：展示 cql_alpha 的影響
```
cql_alpha=5.0：
  critic_loss ≈ bellman(4) + 5.0×cql_penalty(13) ≈ 69（懲罰主導）
  actor_loss 在 100K 轉正 → 策略過保守 → eval 下降

cql_alpha=1.0：
  critic_loss ≈ bellman(4) + 1.0×cql_penalty(13) ≈ 17（正常）
  actor_loss 全程負值 → 策略持續改善 → eval 穩定在 2000+
```

---

#### MADDPG（2017）& MAPPO（2021）—— 多智能體
📁 `06_Advanced_Specialized/2017_MADDPG_MARL/` 和 `2021_MAPPO_MARL/`

**核心概念（CTDE）**：
```
集中式訓練（Centralised Training）：
  Critic 輸入所有智能體的全域狀態 → 優勢估計更準
  Critic_i([obs_1, obs_2, act_1, act_2]) → Q_i（12 維輸入）

分散式執行（Decentralised Execution）：
  每個 Actor 只需自己的觀測 → 不需要通訊
  Actor_1([obs_1]) → act_1（4 維輸入）

核心解決的問題（training_log「CTDE 解法」節）：
  單 agent DDPG 面對多 agent 時「非穩態（Non-Stationarity）」問題：
  其他 agent 策略持續改變 → 從 Q_i 角度環境無故變化 → 不收斂
  CTDE 通過集中式 Critic 將其他 agent 策略納入條件 → 恢復穩態
```

**對比 IPPO（完全分散）**：MAPPO 的全域 Critic 讓優勢估計更準確

---

### 07 Modern RLHF

#### RLHF/InstructGPT（2022）
📁 `07_Modern_RLHF/2022_RLHF_InstructGPT/training_log.md`

**靈魂指標**：三階段的損失演化（SFT：20→12 / RM：~0.693 / PPO：平均獎勵 -17.5）

**課堂展示**：展示三階段串接
```
SFT（監督式微調）：  在示範資料上微調 → 建立初始行為
RM（獎勵模型）：     從偏好對 chosen > rejected 學習獎勵函式
PPO（強化學習）：    以 RM 分數為獎勵 + KL 懲罰 → 最佳化策略

RM 損失 ~0.693（log 2 = 隨機猜測基線）的解釋：
  合成資料沒有真正的偏好信號
  真實 RLHF 需要人工標注資料（Anthropic/hh-rlhf 等）
```

**如何用真實資料替換**：展示 training_log 的「合成資料框架說明」和程式碼範例

---

#### DPO（2023）
📁 `07_Modern_RLHF/2023_DPO/training_log.md`

**核心概念**：跳過 RM 訓練，直接從偏好對最佳化策略

**課堂展示（training_log「DPO 核心機制詳解」節）**：
```
RLHF：策略 → RM 評分 → PPO 更新（三階段，複雜）
DPO：  直接比較 chosen vs rejected 的對數機率（一階段，簡單）

DPO 損失推導：
  RLHF KL 約束目標 → 閉式最優策略 π*(y|x) = π_ref × exp(r/β) / Z(x)
  帶入 Bradley-Terry 偏好模型（Z(x) 消去）→ 閉式損失函式：
  L_DPO = -log σ( β·log(π_θ(y_w)/π_ref(y_w)) − β·log(π_θ(y_l)/π_ref(y_l)) )

為什麼 β=0.1：KL 懲罰強度（β 大→緊貼 SFT，β 小→可大幅偏離）
```

**關鍵指標**：reward margin = chosen 隱式獎勵 − rejected 隱式獎勵（真實資料應持續上升）

---

#### GRPO（2024）
📁 `07_Modern_RLHF/2024_GRPO/training_log.md`

**核心概念**：Group Relative Policy Optimization（DeepSeek-R1 使用的演算法）

**課堂計算（training_log「GRPO 核心機制詳解」節）**：
```
PPO：需要 Critic 網路估計 V(s)（額外參數 + 計算）
GRPO：在同一個 prompt 上生成 G=8 個回應，用相對排名當優勢
      advantage_i = (r_i - mean(r)) / std(r)    ← 不需要 Critic

合成資料：G=8，所有 r_i = 0.1 → std = 0 → advantage = 0 → 零梯度（正確行為）
真實資料：G=8，{1,0,1,0,0,1,0,0}，mean=0.375，std≈0.484
         A_1（正確）= +1.29，A_2（錯誤）= -0.78
```

**DeepSeek-R1-Zero 效果**：MATH 準確率 15% → 70%（純規則式獎勵，無人工標注）

---

## 四、常見學生疑問解答

### Q1：為什麼訓練中 loss 下降，但 eval 反而上升後崩潰？

**標準回答**：以 C51 v1 為例
```
步數 50K–110K：loss 下降 + eval 升高（正常學習）
步數 110K–200K：loss 繼續下降，eval 卻崩潰（過度擬合 buffer 中舊資料）

關鍵：eval 衡量「無探索時的策略」，loss 衡量「有探索時的 Q 估計」
ε 固定在 0.01 後，策略轉換到純貪婪，但 buffer 裡還是探索期的資料 → 策略失去方向
```

### Q2：為什麼 CartPole 的 Q 值會是 200–300，但理論上最高只有 99？

**標準回答**：以 DQN 100K 步（mean Q = 312）為例
```
理論值：Q* ≈ Σ_{t=0}^{499} 0.99^t × 1 = (1 − 0.99^500) / (1 − 0.99) ≈ 99.3

DQN 實際 mean Q ≈ 312（3 倍高估）：
  TD target = r + γ × max Q_target(s')
  max 操作每步系統性選到被高估的動作 → Q 值持續膨脹

DDQN 解法：用 online 選動作，用 target 評估 → 兩個獨立誤差不容易疊加
```

### Q3：REINFORCE 用了 5000 集還沒學會，PPO 20K 步就滿分，差在哪？

**標準回答**：
```
REINFORCE：每集資料只用一次梯度更新 → 樣本效率 = 1×
PPO：每批 2048 步資料使用 10 個 epoch，每個 epoch 約 64 個 minibatch
     等效樣本效率 ≈ 10× REINFORCE
     + clip 防止崩潰 → 可以安全地多次使用同一批資料
```

### Q4：SAC 的 α 為什麼在訓練初期反而上升？

**標準回答**：以 Pendulum 10K 步時 α: 0.20→0.32 為例
```
α 的更新目標：H(π) ≥ H_target（維持策略熵）

訓練初期策略更新很快，熵快速下降（策略快速確定化）
熵 < H_target → α 自動上調，強制策略保持探索
10K 步後策略開始穩定 → α 開始下降（0.32 → 0.02）
```

### Q5：HER 為什麼比「稀疏獎勵 + DDPG」強這麼多？

**標準回答**：
```
純稀疏 DDPG：每集 50 步幾乎全是 r=-1，每 2500 集才偶發 1 次 r=0
→ 梯度方向幾乎為零，策略無法學習

HER：每集結束後，把軌跡中 achieved_goal 當成「假設的目標」
→ 每集有 80% 的轉換都能得到 r=0（成功）
→ 正獎勵密度從 0.04% 提升至 ~80%
→ 梯度方向清晰，策略快速學習
```

---

## 五、演算法 Best Checkpoint 清單

| 演算法 | 最佳 eval | Checkpoint 路徑 |
|---|---|---|
| DDQN | 500.0（100K）| `02_Value_Based_Deep/2015_DoubleDQN/best_checkpoints/double_dqn.pt` |
| PER | 490.1（180K）| `02_Value_Based_Deep/2015_PER/best_checkpoints/per_dqn.pt` |
| DuelingDQN | 500.0 | `02_Value_Based_Deep/2016_DuelingDQN/checkpoints/dueling_dqn.pt` |
| PPO | 500.0（20K）| `03_Policy_Gradient/2017_PPO/checkpoints/ppo.pt` |
| DDPG | -101.6（100K）| `04_Actor_Critic_Continuous/2015_DDPG/checkpoints/` |
| TD3 | -119.8（70K）| `04_Actor_Critic_Continuous/2018_TD3/checkpoints/` |
| SAC | -171.8（100K）| `04_Actor_Critic_Continuous/2018_SAC/checkpoints/` |
| TRPO | 245.7（350ep）| `03_Policy_Gradient/2015_TRPO/best_checkpoints/trpo.pt` |
| A2C | 500.0（300K）| `03_Policy_Gradient/2016_A2C/checkpoints/a2c_vecenv_step300000/` |
| Dreamer | -868.2（475ep）| `05_Model_Based/2019_Dreamer/best_checkpoints/dreamer_state.pt` |
| C51 | 500.0（80K）| `06_Advanced_Specialized/2017_C51_DistRL/checkpoints/best/` |
| HER | 100%（500ep）| `06_Advanced_Specialized/2017_HER/checkpoints/her_epoch500` |
| CQL | 2370.4（50K）| `06_Advanced_Specialized/2020_CQL_Offline/checkpoints/best/` |
| MADDPG | -2.04（41800ep）| `06_Advanced_Specialized/2017_MADDPG_MARL/checkpoints/` |

---

## 六、計算量估算（帶學生做的項目）

以下是每個演算法建議「讓學生現場計算」的數字，選擇 1–2 個就夠。

| 演算法 | 現場計算 | 難度 |
|---|---|---|
| Q-Learning | TD 更新：δ = r + γ maxQ(s') − Q(s,a)，ΔQ = α·δ | ★ |
| DQN | Huber loss 推導：初始 loss ≈ 0.5 的由來 | ★★ |
| REINFORCE | G_t 折扣回報 + 歸一化 + 高方差示例（短集 +2.7 vs 長集 +25.2）| ★★ |
| PPO | ratio = exp(log π_new − log π_old)，clip 操作三種情況 | ★★ |
| PPO GAE | 倒序計算 5 步 GAE：A_4=−7.80, A_3=−6.93, A_2=−4.81 | ★★★ |
| DDPG | 軟更新半衰期：(1−τ)^n = 0.5 → n = ln(0.5)/ln(0.995) ≈ 138 步 | ★★ |
| TD3 | Actor vs Critic 更新次數計算（policy_delay=2）| ★ |
| HER | 隨機成功概率計算（球體體積/工作空間 = 0.04%）| ★ |
| SAC | α 更新方向（熵 < 目標 → α 上升）| ★★ |
| C51 | 支撐集映射：回報 r → 原子索引 = (r − v_min) / Δz | ★★★ |
| MBPO | real_ratio 的效果：95% 模型資料 → Q 爆炸的正反饋迴圈 | ★★ |
| DPO | 損失計算：Δ = β·(log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l))，Δ=0.10 → loss≈0.644 | ★★★ |
| GRPO | 組內正規化：G=8，{1,0,1,0,0,1,0,0}，mean=0.375，A_1=(1−0.375)/0.484=+1.29 | ★★ |
| MADDPG | 集中式 Critic 輸入維度：[obs_1(4), obs_2(4), act_1(2), act_2(2)] = 12 維 | ★ |
