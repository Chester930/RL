"""
在 CartPole-v1 上訓練 A3C（同步單執行緒版本，等價於 A2C）。

關於完整非同步實作的說明：
    真實的 A3C 使用 torch.multiprocessing + 共享記憶體：
        - global_net.share_memory() 建立全域網路
        - 生成 N 個工作者程式，各有獨立環境
        - 每個工作者複製全域權重、收集 n_steps、計算梯度並更新全域網路
        - torch.optim.SharedAdam 支援跨程式引數更新

執行方式：
    python train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym
from collections import deque

from agent import A3CAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

# ──────────────────────────────────────────────
CONFIG = {
    "env_id":          "CartPole-v1",
    "total_steps":     150_000,
    "lr":              7e-4,
    "gamma":           0.99,
    "n_steps":         5,
    "c_v":             0.5,
    "c_e":             0.01,
    "milestone_freq":  50_000,
    "eval_n_eps":      10,
    "first_good_thr":  200,
    "checkpoint_dir":  "checkpoints",
    "log_path":        "training_log.md",
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}

TEST_STATES = [
    ([0.0,   0.0,  0.00,  0.0],  "正中央靜止（初始）"),
    ([0.0,   0.0, -0.10,  0.0],  "桿往左傾（需推左）"),
    ([0.0,   0.0, +0.10,  0.0],  "桿往右傾（需推右）"),
    ([-0.1, -0.5,  0.00,  0.0],  "小車往左移（需推右）"),
]


# ──────────────────────────────────────────────
# Analysis helper
# ──────────────────────────────────────────────

def _get_policy_info(agent, states_list):
    """回傳每個測試狀態的 P(←), P(→), V(s), H(π)。"""
    results = []
    for state_vec, label in states_list:
        s = torch.FloatTensor(state_vec).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            logits, value = agent.global_net(s)
            probs = torch.softmax(logits, dim=-1)[0]
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy().item()
        results.append({
            "label":   label,
            "p_left":  float(probs[0].item()),
            "p_right": float(probs[1].item()),
            "value":   float(value.item()),
            "entropy": float(entropy),
            "action":  "←" if probs[0] > probs[1] else "→",
        })
    return results


# ──────────────────────────────────────────────
# Markdown builders
# ──────────────────────────────────────────────

def _build_header(config):
    return f"""# A3C 訓練日誌（同步單執行緒 A2C）

環境：{config['env_id']}（最大 500 步）｜訓練步數：{config['total_steps']:,}
學習率 α：{config['lr']}｜折扣因子 γ：{config['gamma']}
n-步回報長度：{config['n_steps']}｜評論家係數 c_v：{config['c_v']}｜熵獎勵係數 c_e：{config['c_e']}

---

## 核心概念

### A3C vs DQN 家族的根本差異

```
DQN 家族（離線策略，Off-policy）：
  - 儲存所有過去的轉移到 Replay Buffer
  - 從 Buffer 隨機抽樣進行批次更新
  - 策略選擇：ε-greedy（探索 vs 利用）
  - 更新目標：Q(s,a) = r + γ × max_a' Q(s',a')

A3C（線上策略，On-policy）：
  - 不使用 Replay Buffer，直接用當前策略收集資料
  - 每 n 步更新一次，然後丟棄舊資料
  - 策略選擇：隨機取樣 a ~ π(a|s)（softmax 分佈）
  - 更新目標：最大化 E[log π(a|s) × A(s,a)]
```

### 演員-評論家架構（Actor-Critic）

```
輸入 s（4維）
  │
  ▼
共享骨幹（MLP, 256維）
  │
  ├──▶ 演員 (Actor) head → logits → softmax → π(a|s)   [策略梯度]
  │
  └──▶ 評論家 (Critic) head → V(s)                      [值函式估計]
```

### 損失函式分解

```
① 演員損失 (Actor Loss)：
   L_actor = -E[log π(a|s) × A(s,a)]

   A(s,a) = G_t - V(s_t)    ← 優勢函式 = n-步回報 - 評論家估計

   A > 0：此動作比平均好 → 提高 π(a|s)
   A < 0：此動作比平均差 → 降低 π(a|s)

② 評論家損失 (Critic Loss)：
   L_critic = E[(V(s_t) - G_t)^2]    ← MSE 損失

   G_t = r_t + γ*r_{{t+1}} + ... + γ^{{n-1}}*r_{{t+n-1}} + γ^n * V(s_{{t+n}})

③ 熵獎勵 (Entropy Bonus)：
   L_entropy = -E[H(π(s))] = E[Σ π(a|s) log π(a|s)]

   鼓勵策略保持隨機性，避免過早收斂到次優的確定性策略
   均勻分佈時熵最大：H_max = ln(2) ≈ 0.693

④ 總損失：
   L_total = L_actor + {config['c_v']} × L_critic + {config['c_e']} × L_entropy
```

### n-步回報 vs TD(0) vs Monte Carlo

```
n=1（TD(0)）：G_t = r_t + γV(s_{{t+1}})
  → 低方差，高偏差（仰賴 V 估計，可能傳播誤差）

n=∞（Monte Carlo）：G_t = Σ_{{k=0}}^∞ γ^k * r_{{t+k}}
  → 高方差，零偏差（完整軌跡，但雜訊大）

n={config['n_steps']}（A3C 預設）：
  → 折衷取捨：比 TD 偏差小，比 Monte Carlo 方差小
```

---

"""


def _build_first_update_section(d, config):
    return f"""---

## 第一次網路更新（步數 {d['step']:,}）

| 指標 | 數值 |
|:---|:---|
| 更新使用的步數 | {config['n_steps']} |
| 演員損失 | {d['actor_loss']:.6f} |
| 評論家損失 | {d['critic_loss']:.6f} |
| 策略熵 H(π) | {d['entropy']:.6f} |
| 總損失 | {d['total_loss']:.6f} |

初始策略接近均勻分佈（H ≈ ln(2) ≈ 0.693），演員損失接近 0（優勢 ≈ 0），
評論家損失較大（V(s) ≈ 0，但真實回報 >> 0）。
---

"""


def _build_first_good_ep_section(d, config):
    rows = ""
    for i, step_info in enumerate(d["steps"]):
        direction = "←（推左）" if step_info["action"] == 0 else "→（推右）"
        rows += f"| {i+1} | {step_info['x']:+.3f} | {step_info['theta']:+.2f}° | {direction} | 1 |\n"
    rows += f"| ... | | | | |（共 {d['ep_return']:.0f} 步）|"

    return f"""---

## 第一個優秀集數（第 {d['episode']:,} 集，步數 {d['step']:,}，回報 = {d['ep_return']:.0f}）

此集共持續 **{d['ep_return']:.0f} 步**（CartPole 回報 = 存活步數）。

| 指標 | 數值 |
|:---|:---|
| 訓練步數 | {d['step']:,} |
| 集數 | {d['episode']:,} |
| 資料來源 | 即時收集（無 Replay Buffer）|

### 前 10 步詳情

| 步驟 | x | θ（°）| 動作 | r |
|:---:|:---:|:---:|:---:|:---:|
{rows}
---

"""


def _build_milestone_section(d, step, config):
    return f"""---

## 步數 {step:,} 進度快照

| 指標 | 數值 |
|:---|:---|
| eval 回報（{config['eval_n_eps']} 集均值）| {d['eval_mean']:.1f} ± {d['eval_std']:.1f} |
| 最近 100 集均值 | {d['recent100']:.1f} |
| 最新演員損失 | {d['actor_loss']:.6f} |
| 最新評論家損失 | {d['critic_loss']:.6f} |
| 最新策略熵 H(π) | {d['entropy']:.4f} |
| 總更新次數 | {d['updates']:,} |
---

"""


def _build_final_section(agent, config):
    results = _get_policy_info(agent, TEST_STATES)

    rows = ""
    for r in results:
        rows += (
            f"| {r['label']} "
            f"| {r['value']:.3f} "
            f"| {r['p_left']:.4f} "
            f"| {r['p_right']:.4f} "
            f"| {r['entropy']:.4f} "
            f"| {r['action']}（{'推左' if r['action'] == '←' else '推右'}）|\n"
        )

    return f"""---

## 最終策略快照（訓練完成後）

| 測試狀態 | V(s) | P(←) | P(→) | H(π) | 選擇動作 |
|:---|:---:|:---:|:---:|:---:|:---:|
{rows}
**解讀重點：**
- `V(s)`：評論家估計的狀態回報（越高代表狀態越好）
- `P(←)/P(→)`：演員輸出的策略分佈（隨機，非確定性）
- `H(π)`：策略熵（H_max=0.693；越低越確定；趨近 0 = 完全確定性）
- 桿往左傾 → P(←) 應 > P(→)；桿往右傾 → P(→) 應 > P(←)
- A3C 的策略是機率分佈，不像 DQN 的 argmax 那樣完全確定

---

## 訓練總結

| 設定 | 數值 |
|:---|:---|
| 總訓練步數 | {config['total_steps']:,} |
| n-步回報長度 | {config['n_steps']} |
| 評論家係數 c_v | {config['c_v']} |
| 熵獎勵係數 c_e | {config['c_e']} |

### A3C vs DQN 家族完整對比

| 面向 | DQN / Double DQN | Dueling DQN | A3C |
|:---|:---|:---|:---|
| 策略型別 | 確定性（ε-greedy）| 確定性（ε-greedy）| 隨機（softmax）|
| 資料重用 | Replay Buffer | Replay Buffer | 無（即收即丟）|
| 探索機制 | ε 衰減 | ε 衰減 | 熵獎勵 |
| 網路輸出 | Q(s,a) | V(s) + A(s,a) | π(a|s) + V(s) |
| 更新頻率 | 每步 1 次 | 每步 1 次 | 每 {config['n_steps']} 步 1 次 |
| 目標函式 | TD 誤差最小化 | TD 誤差最小化 | 優勢加權對數機率 |
| 優勢估計 | 隱含在 Q 差 | 隱含在 Q 差 | 顯式 A(s,a) = G_t - V(s_t) |
"""


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train(config: dict) -> A3CAgent:
    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = A3CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        c_v=config["c_v"],
        c_e=config["c_e"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"a3c_{config['env_id']}")

    global_step = 0
    episode = 0
    update_count = 0
    recent100 = deque(maxlen=100)

    first_update_data = None
    first_good_ep_data = None
    milestone_data = {}
    next_milestone = config["milestone_freq"]
    last_metrics = {}

    print(f"訓練中：{config['env_id']}（A3C），共 {config['total_steps']:,} 步...")

    while global_step < config["total_steps"]:
        obs, _ = env.reset()
        ep_return = 0.0
        ep_length = 0
        done = False
        episode += 1
        step_details = []

        while not done and global_step < config["total_steps"]:
            # 收集 n_steps 資料（或直到集數結束）
            for _ in range(config["n_steps"]):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store(obs, action, reward, done)

                if len(step_details) < 10:
                    step_details.append({
                        "x":     float(obs[0]),
                        "theta": float(np.degrees(obs[2])),
                        "action": int(action),
                    })

                obs = next_obs
                ep_return += reward
                ep_length += 1
                global_step += 1

                if done:
                    break

            # 每 n_steps（或集數結束）更新一次
            metrics = agent.update(
                next_state=None if done else obs,
                last_done=done,
            )

            if metrics:
                update_count += 1
                last_metrics = metrics
                logger.log_scalars(metrics, global_step)

                if first_update_data is None:
                    first_update_data = {
                        "step":        global_step,
                        "actor_loss":  metrics["actor_loss"],
                        "critic_loss": metrics["critic_loss"],
                        "entropy":     metrics["entropy"],
                        "total_loss":  metrics["total_loss"],
                    }

        # 集數結束
        recent100.append(ep_return)
        logger.log_episode(ep_return, ep_length, global_step)

        if first_good_ep_data is None and ep_return >= config["first_good_thr"]:
            first_good_ep_data = {
                "episode":   episode,
                "step":      global_step,
                "ep_return": ep_return,
                "steps":     step_details,
            }

        # 里程碑快照（一集可能跨越多個里程碑）
        while global_step >= next_milestone and next_milestone <= config["total_steps"]:
            eval_mean, eval_std = evaluate(agent, eval_env, n_episodes=config["eval_n_eps"])
            logger.log_scalar("eval/mean_return", eval_mean, global_step)
            print(
                f"步數 {next_milestone:>7,}  "
                f"eval={eval_mean:.1f}±{eval_std:.1f}  "
                f"recent100={np.mean(recent100):.1f}"
            )

            milestone_data[next_milestone] = {
                "eval_mean":   eval_mean,
                "eval_std":    eval_std,
                "recent100":   float(np.mean(recent100)),
                "actor_loss":  last_metrics.get("actor_loss", 0.0),
                "critic_loss": last_metrics.get("critic_loss", 0.0),
                "entropy":     last_metrics.get("entropy", 0.0),
                "updates":     update_count,
            }

            agent.save(config["checkpoint_dir"])
            print(f"  → checkpoint 已存（步數 {next_milestone:,}）")
            next_milestone += config["milestone_freq"]

    logger.close()
    env.close()
    eval_env.close()

    # ── 寫入訓練日誌 ──
    sections = [_build_header(config)]

    if first_update_data:
        sections.append(_build_first_update_section(first_update_data, config))

    if first_good_ep_data:
        sections.append(_build_first_good_ep_section(first_good_ep_data, config))
    else:
        sections.append("*(此次訓練未出現回報 ≥ 200 的集數)*\n\n")

    for step in sorted(milestone_data):
        sections.append(_build_milestone_section(milestone_data[step], step, config))

    sections.append(_build_final_section(agent, config))

    log_text = "".join(sections)
    log_path = config["log_path"]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    print(f"日誌已寫入：{os.path.abspath(log_path)}")
    return agent


if __name__ == "__main__":
    train(CONFIG)
