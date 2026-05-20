"""在 CartPole-v1 上訓練 REINFORCE，並產生 training_log.md。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from collections import deque
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import REINFORCEAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

ACTION_NAMES = {0: "←（推左）", 1: "→（推右）"}

CONFIG = {
    "env_id": "CartPole-v1",
    "n_episodes": 5000,
    "lr": 1e-3,
    "gamma": 0.99,
    "use_baseline": True,
    "normalize_returns": False,
    "log_freq": 100,
    "eval_freq": 200,
    "milestone_freq": 500,
    "window": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

THRESHOLD_GOOD = 100   # 第一個值得記錄的里程碑集數（CartPole 理論解決標準為 195，但 REINFORCE 高方差，先記 100）


def train(config: dict):
    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        use_baseline=config["use_baseline"],
        normalize_returns=config["normalize_returns"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"reinforce_{config['env_id']}")

    recent = deque(maxlen=config["window"])
    milestone_blocks: list[str] = []
    first_good_data = None
    first_good_logged = False

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_length = 0
        done = False

        ep_actions: list[int] = []
        ep_rewards: list[float] = []
        ep_states: list[np.ndarray] = []

        while not done:
            ep_states.append(obs.copy())
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward(float(reward))
            ep_actions.append(action)
            ep_rewards.append(float(reward))
            ep_return += float(reward)
            ep_length += 1
            obs = next_obs

        # 在 update() 前捕捉回報（供日誌使用）
        rewards_snap = list(agent._rewards)

        metrics = agent.update()

        recent.append(ep_return)
        logger.log_episode(ep_return, ep_length, step=episode)
        if metrics and episode % config["log_freq"] == 0:
            logger.log_scalars(metrics, step=episode)

        if episode % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, step=episode)
            print(f"Episode {episode:5d}  Eval: {mean_r:.1f} ± {std_r:.1f}  "
                  f"Recent100: {np.mean(recent):.1f}")

        # ── 捕捉第一個「好」集數（回報 >= threshold）────────────────
        if not first_good_logged and ep_return >= THRESHOLD_GOOD:
            first_good_logged = True
            # 從捕捉的 rewards_snap 重算 G_t
            T = len(rewards_snap)
            G_vals: list[float] = [0.0] * T
            G = 0.0
            for t in reversed(range(T)):
                G = rewards_snap[t] + config["gamma"] * G
                G_vals[t] = G
            first_good_data = {
                "episode": episode,
                "ep_return": ep_return,
                "ep_length": ep_length,
                "ep_states": ep_states,
                "ep_actions": ep_actions,
                "ep_rewards": rewards_snap,
                "G_vals": G_vals,
                "loss": metrics.get("loss", float("nan")),
                "mean_return": metrics.get("mean_return", float("nan")),
            }

        # ── 里程碑快照 ───────────────────────────────────────────────
        if episode % config["milestone_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            recent_mean = float(np.mean(recent))
            milestone_blocks.append(
                _milestone_block(episode, recent_mean, mean_r, std_r, metrics)
            )

    logger.close()
    env.close()
    eval_env.close()
    return agent, milestone_blocks, first_good_data


def _milestone_block(episode, recent_mean, eval_mean, eval_std, metrics) -> str:
    loss_str = f"{metrics.get('loss', float('nan')):.4f}" if metrics else "N/A"
    return f"""---

## 第 {episode:,} 集進度摘要

| 指標 | 數值 |
|:---|:---|
| 最近 {CONFIG['window']} 集平均回報 | {recent_mean:.1f} |
| 評估回報（10 集）| {eval_mean:.1f} ± {eval_std:.1f} |
| 本集損失 L | {loss_str} |
| 評估是否達到里程碑（≥{THRESHOLD_GOOD}）| {"✅ 是" if eval_mean >= THRESHOLD_GOOD else "❌ 否"} |
"""


def build_log(agent, milestone_blocks, first_good_data, config) -> str:
    sections = []

    # ── 說明 ──────────────────────────────────────────────────────────
    sections.append(f"""# REINFORCE 訓練日誌

環境：{config['env_id']}（最大 500 步）
訓練集數：{config['n_episodes']:,}
學習率 α：{config['lr']}｜折扣因子 γ：{config['gamma']}
use_baseline：{config['use_baseline']}｜normalize_returns：{config['normalize_returns']}

---

## 說明

### CartPole-v1 環境

```
觀測值（連續 4 維）：
  x         ：小車位置（-4.8 ~ 4.8）
  ẋ         ：小車速度
  θ         ：桿子角度（-24° ~ 24°）
  θ̇         ：桿子角速度

動作（離散 2 維）：
  0 = 向左推力
  1 = 向右推力

獎勵：每活過一步得 r = 1.0
終止條件：桿子傾斜超過 ±12° 或小車超出邊界，或達到 500 步
```

里程碑標準：單集回報 ≥ {THRESHOLD_GOOD} 步（CartPole 官方解決標準為平均 195，REINFORCE 高方差，先記 100 步為首個里程碑）

### REINFORCE 更新流程

```
① 完整集數收集（必須走完才能更新）
   for t = 0, 1, ..., T-1:
       a_t ~ π(·|s_t; θ)          # 從策略取樣動作
       記錄 log π(a_t|s_t; θ)

② 倒序計算折扣回報（蒙特卡羅）
   G_T = 0
   for t = T-1, T-2, ..., 0:
       G_t = r_{{t+1}} + γ × G_{{t+1}}

③ 回報歸一化（方差縮減）
   G_t ← (G_t - mean(G)) / (std(G) + ε)

④ 減去基準值（可選，方差縮減）
   advantage_t = G_t - mean(G)

⑤ 策略梯度損失 + 更新
   L = -Σ_t advantage_t × log π(a_t|s_t; θ)
   θ ← θ - α × ∇L
```

### 網路結構

```
輸入（4 維）→ 全連線 128 → ReLU → 全連線 128 → ReLU → 輸出（2 維 logits）
→ Categorical 分佈 → 取樣動作
```
""")

    # ── 第一個「好」集數 ──────────────────────────────────────────────
    if first_good_data:
        d = first_good_data
        T = d["ep_length"]
        show_n = min(10, T)

        # 前 show_n 步的詳情
        step_rows = [
            "| 步驟 | x | θ（角度°）| 動作 | r | G_t（歸一化前）|",
            "|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for i in range(show_n):
            s = d["ep_states"][i]
            a = d["ep_actions"][i]
            r = d["ep_rewards"][i]
            G = d["G_vals"][i]
            theta_deg = float(s[2]) * (180 / 3.14159)
            step_rows.append(
                f"| {i+1} | {s[0]:+.3f} | {theta_deg:+.2f}° "
                f"| {ACTION_NAMES[a]} | {r:.0f} | {G:.4f} |"
            )
        if T > show_n:
            step_rows.append(f"| ... | ... | ... | ... | ... | ... |（共 {T} 步）|")

        step_table = "\n".join(step_rows)

        # G_t 的統計
        G_arr = d["G_vals"]
        G_mean = float(np.mean(G_arr))
        G_std = float(np.std(G_arr))
        G_max = float(np.max(G_arr))
        G_min = float(np.min(G_arr))

        sections.append(f"""---

## 第一個優秀集數（第 {d['episode']} 集，回報 = {d['ep_return']:.0f}）

此集共持續 **{T} 步**（回報 = 步數，CartPole 每步得 1 分）。

### 前 {show_n} 步詳情

{step_table}

### G_t 統計（歸一化前）

| 指標 | 數值 |
|:---|:---|
| G_0（從第 0 步看的總折扣回報）| {G_arr[0]:.4f} |
| G_t 最大值 | {G_max:.4f} |
| G_t 最小值 | {G_min:.4f} |
| G_t 均值 | {G_mean:.4f} |
| G_t 標準差 | {G_std:.4f} |

### 本集梯度更新

| 指標 | 數值 |
|:---|:---|
| 策略梯度損失 L | {d['loss']:.6f} |
| 歸一化後平均 G_t | {d['mean_return']:.6f} |

**本集重點說明**

- G_0 ≈ {G_arr[0]:.1f}：整集的折扣總回報，γ=0.99 幾乎不衰減，所以 G_0 ≈ 步數 × 0.99^平均。
- G_t 從前往後遞減（每少一步能收集的獎勵就少一步）。
- 歸一化後的 G_t 分佈在 0 附近，前半步（G_t > 0）被強化，後半步（G_t < 0）被抑制。
- 損失 L 為負值意味梯度往提高 log π 的方向走——即讓「帶來高 G_t 的動作」機率升高。
""")

    # ── 里程碑 ────────────────────────────────────────────────────────
    for block in milestone_blocks:
        sections.append(block)

    # ── 最終總結 ──────────────────────────────────────────────────────
    sections.append(f"""---

## 訓練總結

| 訓練設定 | 數值 |
|:---|:---|
| 總集數 | {config['n_episodes']:,} |
| 學習率 | {config['lr']} |
| 折扣因子 γ | {config['gamma']} |
| 基準值 (Baseline) | {"使用" if config['use_baseline'] else "不使用"} |
| 回報歸一化 | {"使用" if config['normalize_returns'] else "不使用"} |

### REINFORCE 的核心限制

1. **高方差**：G_t 是整集所有隨機事件的累積，同一狀態不同集的 G_t 差異很大。
   歸一化和基準值只是部分緩解——Actor-Critic 用 V(s) 作為基準，方差更低。

2. **樣本效率低**：每條軌跡收集後只用一次梯度更新，然後丟棄。
   PPO 的 clip ratio 允許每條軌跡做多次更新。

3. **必須等整集結束**：無法做線上（on-step）學習。
   TD 系列方法（SARSA、A2C）每步更新，不需要等整集。
""")

    return "\n".join(sections)


if __name__ == "__main__":
    print("訓練中...")
    agent, milestone_blocks, first_good_data = train(CONFIG)

    log_path = os.path.join(os.path.dirname(__file__), "training_log.md")
    log_content = build_log(agent, milestone_blocks, first_good_data, CONFIG)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"訓練完成！日誌已寫入：{log_path}")
