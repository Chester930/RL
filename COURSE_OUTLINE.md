# 機器人強化學習實作課程大綱

**目標**：從模仿學習到強化學習，帶領學生理解如何讓機器人手臂學會動作
**對象**：有一些 RL 理論基礎，尚未深入實作的學生
**方法數**：7 種

---

## 核心敘事弧線

```
人類示範 → 模仿失敗 → 需要 RL → 離散不夠 → 連續動作 → 多關節穩定
   BC         分佈偏移    DQN       DDPG         TD3/SAC
```

每個方法都在回答前一個方法留下的問題，形成一條完整因果鏈。

---

## 第一幕：為什麼需要 RL？

### 方法 1｜Behavioral Cloning（模仿學習）

**核心問題**：「最直覺的想法：讓機器人看人怎麼做，照著學。」

**理論複習**
- 概念：收集人類示範軌跡 → 監督學習 → 學出 policy
- 本質是把 RL 問題轉化成監督學習：
  ```python
  loss = MSE(policy(state), expert_action)
  ```

**實作重點**
- 環境：Pendulum-v1（模擬單關節扭矩控制）
- 收集專家示範 → 訓練 policy network → 測試

**引導到限制**：Distribution Shift（分佈偏移）
- 訓練時只看過「好的狀態」，測試時稍微偏離就崩潰
- 比喻：只教學生考試範本，碰到新題就不會
- 結論：BC 是起點，但機器人需要能**自己探索、從錯誤中修正**的能力

---

### 方法 2｜Q-Learning

**核心問題**：「給機器人一個獎懲訊號，讓它自己學——RL 的基礎語言。」

**理論複習**（三件事）
1. Q(s, a)：在狀態 s 採取動作 a 的長期期望回報
2. Bellman Equation：
   ```
   Q(s, a) ← Q(s, a) + α [r + γ·max Q(s', a') − Q(s, a)]
   ```
3. ε-greedy：ε 機率探索，1-ε 機率利用

**實作重點**
- 環境：CartPole（快速展示 Q-table 收斂）
- 展示 Q-table 的大小隨狀態維度爆炸

**引導到限制**：機器手臂有無限多種姿態（連續狀態空間）→ Q-table 裝不下

---

### 方法 3｜DQN（Deep Q-Network）

**核心問題**：「用神經網路取代 Q-table，解決無限狀態問題。」

**理論複習**（兩個關鍵設計）
- **Experience Replay**：打破時序相關性，穩定訓練
- **Target Network**：固定學習目標，防止「追著自己跑」
  ```python
  # TD target 由慢更新的 target net 決定
  y = r + γ · max_a Q_target(s', a)
  ```

**實作重點**
- 環境：CartPole-v1
- 展示訓練結果：eval = 500（滿分）
- 程式碼差異：Q-table 的 lookup → `online_net(state).max()`

**引導到限制**：輸出是離散動作（左/右），無法控制馬達的連續角度

---

## 第二幕：連續動作——進入真實機器人領域

### 方法 4｜DDPG（Deep Deterministic Policy Gradient）

**核心問題**：「如何控制連續的馬達扭矩？」

**理論複習**：Actor-Critic 架構
```
Actor  網路：state → action（連續值，例如扭矩 ∈ [-2, 2]）
Critic 網路：(state, action) → Q 值（評估這個動作好不好）
```

與機器手臂連結：
- Actor = 關節角度控制器（「現在這個姿態，施加多少扭矩？」）
- Critic = 評估這個姿態的長期價值

**實作重點**
- 環境：Pendulum-v1（單關節，連續扭矩控制）
- 展示訓練結果：eval ≈ -101（接近最優 -100）
- 關鍵程式碼：
  ```python
  action = actor(state)                          # 連續動作
  q_value = critic(state, action)                # 評估
  actor_loss = -critic(state, actor(state)).mean()  # 讓 Critic 給分更高
  ```

**引導到限制**：訓練不穩定，超引數敏感，Q 值容易高估

---

### 方法 5｜TD3（Twin Delayed Deep Deterministic）

**核心問題**：「DDPG 的三個已知問題，TD3 一次修掉。」

**理論複習**：三個改進
| 問題 | TD3 的解法 |
|---|---|
| Q 值高估 | **Twin Critics**：取兩個 Critic 的最小值 |
| Actor 更新太快 | **Delayed Actor Update**：每 2 步才更新一次 Actor |
| 過擬合特定動作 | **Target Policy Smoothing**：目標動作加入雜訊 |

**實作重點**
- 環境：Pendulum-v1
- 對比 DDPG：相同環境，TD3 eval ≈ -119，DDPG ≈ -101（差異不大但訓練更穩）
- 程式碼差異：相對 DDPG 只增加 ~20 行

**引導到限制**：探索仍依賴人工加雜訊，多關節任務容易陷入區域性最優

---

### 方法 6｜SAC（Soft Actor-Critic）

**核心問題**：「如何讓機器人自動平衡探索與利用？引入資訊熵。」

**理論複習**：最大熵 RL
```
目標：最大化 Σ [r_t + α · H(π(·|s_t))]
           ↑ 一般 reward    ↑ 鼓勵高 entropy（多樣性）
```

為什麼 entropy 對機器人重要？
- 高 entropy = 保持探索 = 不陷入區域性最優
- 機器手臂比喻：不只學「一種解法」，而是學會「任何合理姿態都行」
- 更強的泛化能力，部署到真實硬體時更穩健

**實作重點**
- 環境 1：Pendulum-v1，eval ≈ -171（注意：Pendulum 太簡單，entropy 正則化反而過度探索，導致 SAC 在此不如 DDPG/TD3；SAC 的優勢在複雜任務）
- 環境 2：LunarLanderContinuous-v3，eval ≈ 262 @ 100K steps（SAC 50K 就突破 >200，而 PPO 需 ~163K）
- 自動調整 α 的機制（α: 0.32 → 0.02，不需要手動調探索率）

**三種連續控制演算法在 Pendulum-v1 的對比**：

| 演算法 | 最佳 Eval | 步數 | 說明 |
|---|---|---|---|
| DDPG | **-101.6** | 100K | 單一 Critic，訓練不穩但此環境峰值最高 |
| TD3 | -119.8 | 70K | 收斂更快，Twin Critic 穩定性提升 |
| SAC | -171.8 | 100K | 過度探索；優勢要在 LunarLander+ 才顯現 |

> 結論：SAC 的競爭力不在極簡任務，而在**多軸連續控制**（LunarLander、MuJoCo）。

**SAC 為何是現在機器人 RL 的業界標準**：
- 樣本效率高（off-policy，可重用舊資料）
- 訓練穩定（entropy 正則化防止策略崩潰）
- 對超引數不敏感

---

## 第三幕：穩定訓練，走向實際部署

### 方法 7｜PPO（Proximal Policy Optimization）

**核心問題**：「為什麼 OpenAI、DeepMind 的機器人論文幾乎都用 PPO？」

**理論複習**：先展示 REINFORCE 的問題
- REINFORCE：直接對 log π × G_t 做梯度上升
- 問題：每次更新幅度無限制 → 策略崩潰 → eval 停在 9-10（實際結果展示）

PPO 的核心創新：Clipped Surrogate Objective
```python
ratio = π_new(a|s) / π_old(a|s)
loss = -min(ratio * A,  clip(ratio, 1-ε, 1+ε) * A)
# 「每次更新不能走太遠，
#   新策略和舊策略的差距要在安全範圍內。」
```

**實作重點**
- 環境：CartPole → LunarLander-v3
- 展示結果：CartPole 500 + LunarLander eval ≈ 284
- 對比 REINFORCE（eval=9.5）讓 Clip 的效果一目瞭然

**SAC vs PPO 使用場景對比**

| | SAC | PPO |
|---|---|---|
| 訓練方式 | Off-policy（可重用舊資料）| On-policy（每次更新丟舊資料）|
| 樣本效率 | 高（真實機器人收集資料成本高）| 低 |
| 訓練穩定性 | 高 | 很高 |
| 適用場景 | **真實機器人部署** | **模擬環境快速迭代** |

---

## 課程總覽

| 幕 | 方法 | 核心問題 | 環境 |
|---|---|---|---|
| 一 | Behavioral Cloning | 示範夠用嗎？| Pendulum |
| 一 | Q-Learning | RL 基礎語言 | CartPole |
| 一 | DQN | 無限狀態怎麼辦？| CartPole |
| 二 | DDPG | 連續動作怎麼辦？| Pendulum |
| 二 | TD3 | 為何 DDPG 不穩？| Pendulum |
| 二 | SAC | 如何自動探索？| Pendulum + LunarLander |
| 三 | PPO | 如何穩定部署？| CartPole + LunarLander |

---

## 每種演算法的標準展示格式

```
① 30 秒「上一個方法的問題是⋯⋯」
② 核心理論複習（1-2 張投影片）
③ 核心程式碼差異（相對前一個只改幾行）
④ 訓練曲線 / eval 數字展示
⑤ 30 秒「這個方法的限制是⋯⋯」→ 引出下一個方法
```

---

## 延伸方向（課後或進階班）

- **HER（Hindsight Experience Replay）**：稀疏獎勵問題，適合 FetchReach 機器手臂
- **GAIL（Generative Adversarial Imitation Learning）**：BC 的進階版，結合 RL 與模仿
- **Sim-to-Real Transfer**：模擬訓練 → 真實硬體部署的 gap 問題
- **Model-Based RL（Dreamer / MuZero）**：用世界模型減少真實資料需求
