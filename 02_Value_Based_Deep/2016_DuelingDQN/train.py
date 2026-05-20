"""在 CartPole-v1 上訓練 Dueling DQN，並產生 training_log.md。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym
from collections import deque

from agent import DuelingDQNAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

ACTION_NAMES = {0: "←（推左）", 1: "→（推右）"}

CONFIG = {
    "env_id":          "CartPole-v1",
    "total_steps":     300_000,
    "lr":              5e-4,
    "gamma":           0.99,
    "buffer_size":     100_000,
    "batch_size":      64,
    "target_update":   200,
    "epsilon_start":   1.0,
    "epsilon_end":     0.01,
    "epsilon_steps":   50_000,
    "learning_starts": 1_000,
    "log_freq":        1_000,
    "eval_freq":       10_000,
    "milestone_freq":  50_000,
    "window":          100,
    "good_threshold":  200,
    "checkpoint_dir":  "checkpoints",
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}


# ── checkpoint metadata ───────────────────────────────────────────────

def _meta_path(ckpt_dir):
    return os.path.join(ckpt_dir, "train_meta.json")


def _load_meta(ckpt_dir):
    p = _meta_path(ckpt_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_meta(ckpt_dir, meta):
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(_meta_path(ckpt_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _append_log(path, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)


# ── V / A 分解分析 ────────────────────────────────────────────────────

def _decompose(agent, states_list):
    """回傳各測試狀態的 V(s)、A(s,←)、A(s,→)、Q(s,←)、Q(s,→)。"""
    results = []
    for label, s in states_list:
        s_t = torch.FloatTensor(s).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            features = agent.online_net.backbone(s_t)
            V = agent.online_net.value_stream(features)          # (1,1)
            A = agent.online_net.advantage_stream(features)      # (1,2)
            Q = V + A - A.mean(dim=1, keepdim=True)              # (1,2)
        v = float(V.squeeze().item())
        a0, a1 = float(A[0, 0].item()), float(A[0, 1].item())
        q0, q1 = float(Q[0, 0].item()), float(Q[0, 1].item())
        best = ACTION_NAMES[int(Q.argmax(dim=1).item())]
        results.append((label, v, a0, a1, q0, q1, best))
    return results


# ── Log 段落建構 ──────────────────────────────────────────────────────

def _build_header(config):
    return f"""# Dueling DQN 訓練日誌

環境：{config['env_id']}（最大 500 步）｜訓練步數：{config['total_steps']:,}
學習率 α：{config['lr']}｜折扣因子 γ：{config['gamma']}
Replay Buffer：{config['buffer_size']:,} 筆｜Batch size：{config['batch_size']}
Target 更新：每 {config['target_update']} 步｜ε：{config['epsilon_start']} → {config['epsilon_end']}（前 {config['epsilon_steps']:,} 步）

---

## 核心概念

### DQN 的問題：Q(s,a) 難以區分「狀態本身的好壞」與「動作的相對優勢」

```
Q(s, a) = 「在狀態 s 下選動作 a 的期望折扣總回報」

問題：在許多狀態下，不管選哪個動作結果都差不多
（例如：CartPole 桿子幾乎直立時，往左或往右影響不大）
→ 普通 Q 網路需要對每個 (s,a) 對分別學習，效率低落
```

### Dueling DQN：將 Q 值分解為 V(s) 與 A(s,a)

```
Q(s, a) = V(s) + A(s, a) − mean_{{a'}} A(s, a')

V(s)    ： 狀態價值函式，衡量「不管採取什麼動作，這個狀態有多好」
A(s, a) ： 優勢函式，衡量「動作 a 相較於平均動作好多少」
mean(A) ： 減去平均優勢，確保 V 與 A 的分解具有唯一性
```

### 架構：共享骨幹 → 兩條獨立的流

```
輸入 s（4維）
  │
  ▼
共享骨幹（MLP, 256維）
  │
  ├──▶ 價值流 Value stream：Linear(256) → ReLU → Linear(1)    → V(s)
  │
  └──▶ 優勢流 Advantage stream：Linear(256) → ReLU → Linear(2) → A(s,←), A(s,→)

Q(s,←) = V(s) + A(s,←) − [A(s,←) + A(s,→)] / 2
Q(s,→) = V(s) + A(s,→) − [A(s,←) + A(s,→)] / 2
```

### 為什麼有效？

```
① V(s) 可以在「不需要動作標籤」的情況下更新
  → 每次更新都能改善 V(s)，學習效率更高

② 在「動作差異小」的狀態，A(s,a) ≈ 0
  → 策略由 V(s) 主導，不浪費容量在無意義的動作區分

③ 在「稀疏獎勵」環境特別有效
  → 能更快收斂到好的 V(s) 估計，即使 A(s,a) 尚未準確
```

---

"""


def _build_first_update_section(d, config):
    return f"""---

## 緩衝區填充階段（前 {config['learning_starts']:,} 步）

前 {config['learning_starts']:,} 步：ε = 1.0，完全隨機動作，**只收集資料不更新網路**。

### 第一次網路更新（步數 {d['step']:,}）

| 指標 | 數值 |
|:---|:---|
| 緩衝區大小 | {d['buffer_size']:,} 筆 |
| 當前 ε | {d['epsilon']:.4f} |
| 初始 loss | {d['loss']:.6f} |
| 初始 mean Q | {d['mean_q']:.4f} |

初始 Q 接近 0，V(s) 與 A(s,a) 尚未分化，網路兩條流的輸出幾乎相同。
"""


def _build_first_good_section(d, config):
    T = d["ep_length"]
    show_n = min(10, T)
    rows = ["| 步驟 | x | θ（°）| 動作 | r |",
            "|:---:|:---:|:---:|:---:|:---:|"]
    for i in range(show_n):
        s = d["ep_states"][i]
        a = d["ep_actions"][i]
        r = d["ep_rewards"][i]
        theta = float(s[2]) * (180.0 / 3.14159)
        rows.append(f"| {i+1} | {s[0]:+.3f} | {theta:+.2f}° | {ACTION_NAMES[a]} | {r:.0f} |")
    if T > show_n:
        rows.append(f"| ... | | | | |（共 {T} 步）|")
    table = "\n".join(rows)
    return f"""---

## 第一個優秀集數（第 {d['ep_num']} 集，步數 {d['step']:,}，回報 = {d['ep_return']:.0f}）

此集共持續 **{T} 步**（CartPole 回報 = 存活步數）。

| 指標 | 數值 |
|:---|:---|
| 訓練步數 | {d['step']:,} |
| ε | {d['epsilon']:.4f} |
| 緩衝區 | {d['buffer_size']:,} 筆 |

### 前 {show_n} 步詳情

{table}
"""


def _milestone_block(step, eval_mean, eval_std, recent_mean, eps,
                     buf_size, metrics, config):
    loss_str  = f"{metrics.get('loss', float('nan')):.6f}" if metrics else "N/A"
    meanq_str = f"{metrics.get('mean_q', float('nan')):.4f}" if metrics else "N/A"
    return f"""---

## 步數 {step:,} 進度快照

| 指標 | 數值 |
|:---|:---|
| eval 回報（10 集均值）| {eval_mean:.1f} ± {eval_std:.1f} |
| 最近 {config['window']} 集均值 | {recent_mean:.1f} |
| 當前 ε | {eps:.4f} |
| 緩衝區大小 | {buf_size:,} / {config['buffer_size']:,} |
| 最新 loss | {loss_str} |
| 最新 mean Q | {meanq_str} |
| 目標網路已同步次數 | {step // config['target_update']} |
"""


def _build_final_section(agent, config):
    test_states = [
        ("正中央靜止（初始）",    np.array([0.0,  0.0,  0.0,  0.0], dtype=np.float32)),
        ("桿往左傾（需推左）",    np.array([0.0,  0.0, -0.1, -0.1], dtype=np.float32)),
        ("桿往右傾（需推右）",    np.array([0.0,  0.0, +0.1, +0.1], dtype=np.float32)),
        ("小車往左移（需推右）",  np.array([-1.0, -0.5,  0.0,  0.0], dtype=np.float32)),
    ]
    decomp = _decompose(agent, test_states)

    q_rows = ["| 測試狀態 | V(s) | A(←) | A(→) | Q(←) | Q(→) | 選擇動作 |",
              "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for label, v, a0, a1, q0, q1, best in decomp:
        q_rows.append(
            f"| {label} | {v:.3f} | {a0:+.3f} | {a1:+.3f} | {q0:.3f} | {q1:.3f} | {best} |"
        )
    q_table = "\n".join(q_rows)

    return f"""---

## 最終 V/A/Q 分解快照（訓練完成後）

{q_table}

**解讀重點：**
- `V(s)`：反映狀態本身的好壞，與動作無關——越靠近目標的狀態 V 應越高
- `A(s,a)`：正值代表此動作優於平均，負值代表劣於平均
- `Q(s,a) = V(s) + A(s,a) − mean(A)`：最終決策依據
- 桿往左傾時 A(←) 應 > A(→)；桿往右傾時 A(→) 應 > A(←)

---

## 訓練總結

| 設定 | 數值 |
|:---|:---|
| 總訓練步數 | {config['total_steps']:,} |
| 目標網路同步次數 | {config['total_steps'] // config['target_update']} |
| ε 衰減區間 | 前 {config['epsilon_steps']:,} 步 |

### Dueling DQN 相對於 Double DQN 的改進

| 面向 | Double DQN | Dueling DQN |
|:---|:---|:---|
| 網路輸出 | Q(s,a) 直接輸出 | Q = V(s) + A(s,a) − mean(A) |
| 更新效率 | 每次只更新選到的動作 | V(s) 每次都更新（更高效）|
| 稀疏獎勵 | 較慢收斂 | 更快建立好的 V(s) 估計 |
| 程式碼改動量 | — | 僅換網路架構，訓練邏輯不變 |
"""


# ── 訓練主迴圈 ────────────────────────────────────────────────────────

def train(config):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, config["checkpoint_dir"])
    log_path = os.path.join(base_dir, "training_log.md")

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = DuelingDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        target_update=config["target_update"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_steps=config["epsilon_steps"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"dueling_dqn_{config['env_id']}")

    meta = _load_meta(ckpt_dir)
    resume_step = meta.get("train_step", 0)
    first_good_logged = meta.get("first_good_logged", False)
    first_update_logged = meta.get("first_update_logged", False)
    ep_num = meta.get("ep_num", 0)

    if resume_step > 0:
        agent.load(ckpt_dir)
        print(f"找到 checkpoint（步數 {resume_step:,}），暖機中...")
        warmup_obs, _ = env.reset()
        while len(agent.buffer) < config["learning_starts"]:
            a = int(np.random.randint(agent.action_dim))
            w_next, w_r, w_t, w_u, _ = env.step(a)
            w_done = w_t or w_u
            agent.buffer.push(warmup_obs, a, float(w_r), w_next, w_done)
            warmup_obs = env.reset()[0] if w_done else w_next
        print(f"從步數 {resume_step + 1:,} 繼續...")
    else:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(_build_header(config))
        print(f"訓練中：{config['env_id']}（Dueling DQN），共 {config['total_steps']:,} 步...")

    obs, _ = env.reset()
    ep_return = ep_length = 0
    ep_states = [obs.copy()]
    ep_actions = []
    ep_rewards = []
    recent_returns = deque(maxlen=config["window"])
    last_metrics = {}

    for step in range(resume_step + 1, config["total_steps"] + 1):

        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.push(obs, action, float(reward), next_obs, done)
        ep_actions.append(action)
        ep_rewards.append(float(reward))
        ep_return += float(reward)
        ep_length += 1
        obs = next_obs
        if not done:
            ep_states.append(obs.copy())

        if done:
            ep_num += 1
            recent_returns.append(ep_return)
            logger.log_episode(ep_return, ep_length, step)

            if not first_good_logged and ep_return >= config["good_threshold"]:
                first_good_logged = True
                _append_log(log_path, _build_first_good_section({
                    "ep_num": ep_num, "step": step,
                    "ep_return": ep_return, "ep_length": ep_length,
                    "ep_states": ep_states[:], "ep_actions": ep_actions[:],
                    "ep_rewards": ep_rewards[:],
                    "epsilon": agent.epsilon_schedule.get(step),
                    "buffer_size": len(agent.buffer),
                }, config))

            obs, _ = env.reset()
            ep_return = ep_length = 0
            ep_states = [obs.copy()]
            ep_actions = []
            ep_rewards = []

        if step >= config["learning_starts"]:
            metrics = agent.update()
            if metrics:
                last_metrics = metrics

                if not first_update_logged:
                    first_update_logged = True
                    _append_log(log_path, _build_first_update_section({
                        "step": step,
                        "buffer_size": len(agent.buffer),
                        "loss": metrics["loss"],
                        "mean_q": metrics["mean_q"],
                        "epsilon": metrics.get("epsilon", 0.0),
                    }, config))

                if step % config["log_freq"] == 0:
                    logger.log_scalars(metrics, step)

        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=10)
            logger.log_scalar("eval/mean_return", mean_r, step)
            eps = agent.epsilon_schedule.get(step)
            recent_mean = float(np.mean(recent_returns)) if recent_returns else 0.0
            print(f"步數 {step:8,}  eval={mean_r:.1f}±{std_r:.1f}  "
                  f"recent{config['window']}={recent_mean:.1f}  ε={eps:.3f}")

        if step % config["milestone_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=10)
            recent_mean = float(np.mean(recent_returns)) if recent_returns else 0.0
            eps = agent.epsilon_schedule.get(step)
            _append_log(log_path, _milestone_block(
                step, mean_r, std_r, recent_mean, eps,
                len(agent.buffer), last_metrics, config
            ))
            agent.save(ckpt_dir)
            _save_meta(ckpt_dir, {
                "train_step": step, "ep_num": ep_num,
                "first_good_logged": first_good_logged,
                "first_update_logged": first_update_logged,
            })
            print(f"  → checkpoint 已存（步數 {step:,}）")

    logger.close()
    env.close()
    eval_env.close()
    return agent


# ── 入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = train(CONFIG)
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.md")
    _append_log(log_path, _build_final_section(agent, CONFIG))
    print(f"日誌已寫入：{log_path}")
