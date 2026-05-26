"""在 CartPole-v1 上訓練 Double DQN，並產生 training_log.md。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import json
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym
from collections import deque

from agent import DoubleDQNAgent
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
    "best_checkpoint_dir": "best_checkpoints",
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "seed":            42,
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


# ── 高估量計算 ────────────────────────────────────────────────────────

def _compute_overestimation(agent):
    """
    比較 DQN 目標與 DDQN 目標的均值差，量化高估偏差量。

    DQN  目標：max_a Q_target(s', a)                          ← 容易高估
    DDQN 目標：Q_target(s', argmax_a Q_online(s', a))         ← 解耦後偏差較低
    """
    if len(agent.buffer) < agent.batch_size:
        return None
    n = min(512, len(agent.buffer))
    batch = agent.buffer.sample(n)
    next_states = torch.FloatTensor(batch["next_states"]).to(agent.device)
    with torch.no_grad():
        dqn_next_q = agent.target_net(next_states).max(dim=1)[0]
        best_a = agent.online_net(next_states).argmax(dim=1, keepdim=True)
        ddqn_next_q = agent.target_net(next_states).gather(1, best_a).squeeze(1)
    return float((dqn_next_q - ddqn_next_q).mean().item())


# ── Log 段落建構 ──────────────────────────────────────────────────────

def _build_header(config):
    return f"""# Double DQN 訓練日誌

環境：{config['env_id']}（最大 500 步）｜訓練步數：{config['total_steps']:,}
學習率 α：{config['lr']}｜折扣因子 γ：{config['gamma']}
Replay Buffer：{config['buffer_size']:,} 筆｜Batch size：{config['batch_size']}
Target 更新：每 {config['target_update']} 步｜ε：{config['epsilon_start']} → {config['epsilon_end']}（前 {config['epsilon_steps']:,} 步）

---

## 核心概念

### 為什麼 DQN 會高估 Q 值？

DQN 計算 TD 目標時：

```
y = r + γ × max_a Q_target(s', a)
```

問題在於 **max 運算元** 同時負責兩件事：
- **選擇**最佳動作：a* = argmax Q_target(s', a)
- **評估**該動作的價值：Q_target(s', a*)

神經網路對每個動作的估計都有隨機誤差（高估或低估）。max 運算元
**系統性地選到當前被高估的動作**，導致 Q 值持續膨脹，學習不穩定。

### Double DQN：解耦「選擇」與「評估」

```
步驟 1：線上網路選動作    a* = argmax_a Q_online(s', a)
步驟 2：目標網路評估價值  y  = r + γ × Q_target(s', a*)
```

關鍵洞見：Q_online 與 Q_target 有不同的隨機誤差模式。
即使 Q_online 選了一個「看起來最好」的動作，
Q_target 不太可能對 **同一個** 動作也恰好高估。
兩個獨立誤差不太可能同向疊加，高估大幅降低。

### 程式碼改動（相對 DQN 只改 3 行）

```python
# DQN（1 行）：
next_q = target_net(s').max(dim=1)[0]

# Double DQN（3 行）：
best_a  = online_net(s').argmax(dim=1, keepdim=True)   # online 選動作
next_q  = target_net(s').gather(1, best_a).squeeze(1)  # target 評估價值
```

---

"""


def _build_first_update_section(d, config):
    return f"""---

## 緩衝區填充階段（前 {config['learning_starts']:,} 步）

前 {config['learning_starts']:,} 步：ε = 1.0，完全隨機動作，**只收集資料不更新網路**。

| 時刻 | 事件 |
|:---|:---|
| 第 1 步 | 開始收集，緩衝區空（0 筆）|
| 第 {config['learning_starts']:,} 步 | 緩衝區達 {config['learning_starts']:,} 筆，**首次觸發更新** |

### 第一次網路更新（步數 {d['step']:,}）

| 指標 | 數值 |
|:---|:---|
| 緩衝區大小 | {d['buffer_size']:,} 筆 |
| 當前 ε | {d['epsilon']:.4f} |
| 初始 loss | {d['loss']:.6f} |
| 初始 mean Q | {d['mean_q']:.4f} |

初始 loss 高、mean Q 接近 0——網路剛初始化，Q 值估計完全不準。
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
                     buf_size, metrics, overest_gap, config):
    loss_str  = f"{metrics.get('loss', float('nan')):.6f}" if metrics else "N/A"
    meanq_str = f"{metrics.get('mean_q', float('nan')):.4f}" if metrics else "N/A"
    over_str  = f"{overest_gap:.4f}" if overest_gap is not None else "N/A"
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
| Q 高估量（DQN − DDQN 目標差）| {over_str} |
| 目標網路已同步次數 | {step // config['target_update']} |
"""


def _build_final_section(agent, config):
    test_states = [
        ("正中央靜止（初始）",    np.array([0.0,  0.0,  0.0,  0.0], dtype=np.float32)),
        ("桿往左傾（需推左）",    np.array([0.0,  0.0, -0.1, -0.1], dtype=np.float32)),
        ("桿往右傾（需推右）",    np.array([0.0,  0.0, +0.1, +0.1], dtype=np.float32)),
        ("小車往左移（需推右）",  np.array([-1.0, -0.5,  0.0,  0.0], dtype=np.float32)),
    ]
    q_rows = ["| 測試狀態 | Q(←) | Q(→) | 選擇動作 |",
              "|:---|:---:|:---:|:---:|"]
    for label, s in test_states:
        s_t = torch.FloatTensor(s).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            qs = agent.online_net(s_t).squeeze().cpu().numpy()
        best = int(np.argmax(qs))
        q_rows.append(f"| {label} | {qs[0]:.3f} | {qs[1]:.3f} | {ACTION_NAMES[best]} |")
    q_table = "\n".join(q_rows)

    overest = _compute_overestimation(agent)
    over_str = f"{overest:.4f}" if overest is not None else "N/A"

    return f"""---

## 最終 Q 值快照（訓練完成後）

{q_table}

**解讀**：
- 桿往左傾時，Q(←) 應 > Q(→)（往左推讓桿回正）
- 桿往右傾時，Q(→) 應 > Q(←)（往右推讓桿回正）

---

## 高估偏差分析（訓練結束）

| 指標 | 數值 |
|:---|:---|
| DQN 目標均值 − DDQN 目標均值 | {over_str} |

> 此差值代表若使用標準 DQN，每步 TD 目標被額外高估的量。
> 值越大代表 Double DQN 的修正效果越顯著。

---

## 訓練總結

| 設定 | 數值 |
|:---|:---|
| 總訓練步數 | {config['total_steps']:,} |
| 目標網路同步次數 | {config['total_steps'] // config['target_update']} |
| ε 衰減區間 | 前 {config['epsilon_steps']:,} 步 |

### Double DQN 相對於 DQN 的改進

| 面向 | DQN | Double DQN |
|:---|:---|:---|
| TD 目標公式 | `max_a Q_target(s', a)` | `Q_target(s', argmax_a Q_online(s', a))` |
| 高估傾向 | 明顯（max 運算元偏差）| 大幅降低 |
| 程式碼改動量 | — | 僅 3 行 |
| 訓練穩定性 | 一般 | 更穩定 |
"""


# ── 訓練主迴圈 ────────────────────────────────────────────────────────

def train(config):
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, config["checkpoint_dir"])
    log_path = os.path.join(base_dir, "training_log.md")

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = DoubleDQNAgent(
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

    logger = Logger(log_dir="runs", run_name=f"double_dqn_{config['env_id']}")

    # ── 嘗試接續 checkpoint ──────────────────────────────────────────
    best_ckpt_dir = os.path.join(base_dir, config["best_checkpoint_dir"])
    meta = _load_meta(ckpt_dir)
    resume_step = meta.get("train_step", 0)
    first_good_logged = meta.get("first_good_logged", False)
    first_update_logged = meta.get("first_update_logged", False)
    ep_num = meta.get("ep_num", 0)
    best_eval = meta.get("best_eval", -float("inf"))

    if resume_step > 0:
        agent.load(ckpt_dir)
        print(f"找到 checkpoint（步數 {resume_step:,}），暖機中（隨機動作重填 buffer）...")
        warmup_obs, _ = env.reset()
        while len(agent.buffer) < config["learning_starts"]:
            a = int(np.random.randint(agent.action_dim))
            w_next, w_r, w_t, w_u, _ = env.step(a)
            w_done = w_t or w_u
            agent.buffer.push(warmup_obs, a, float(w_r), w_next, w_done)
            warmup_obs = env.reset()[0] if w_done else w_next
        print(f"暖機完成，從步數 {resume_step + 1:,} 繼續訓練...")
    else:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(_build_header(config))
        print(f"訓練中：{config['env_id']}，共 {config['total_steps']:,} 步...")

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

        # ── 更新 ──────────────────────────────────────────────────────
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

        # ── 評估 ──────────────────────────────────────────────────────
        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=10)
            logger.log_scalar("eval/mean_return", mean_r, step)
            eps = agent.epsilon_schedule.get(step)
            recent_mean = float(np.mean(recent_returns)) if recent_returns else 0.0
            if mean_r > best_eval:
                best_eval = mean_r
                agent.save(best_ckpt_dir)
                print(f"步數 {step:8,}  eval={mean_r:.1f}±{std_r:.1f}  "
                      f"recent{config['window']}={recent_mean:.1f}  ε={eps:.3f}  ★ best={best_eval:.1f}")
            else:
                print(f"步數 {step:8,}  eval={mean_r:.1f}±{std_r:.1f}  "
                      f"recent{config['window']}={recent_mean:.1f}  ε={eps:.3f}")

        # ── 里程碑：寫 log + 存 checkpoint ────────────────────────────
        if step % config["milestone_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=10)
            recent_mean = float(np.mean(recent_returns)) if recent_returns else 0.0
            eps = agent.epsilon_schedule.get(step)
            overest = _compute_overestimation(agent)
            _append_log(log_path, _milestone_block(
                step, mean_r, std_r, recent_mean, eps,
                len(agent.buffer), last_metrics, overest, config
            ))
            agent.save(ckpt_dir)
            _save_meta(ckpt_dir, {
                "train_step": step, "ep_num": ep_num,
                "first_good_logged": first_good_logged,
                "first_update_logged": first_update_logged,
                "best_eval": best_eval,
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
