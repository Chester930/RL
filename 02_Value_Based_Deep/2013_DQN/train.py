"""在 CartPole-v1 上訓練 DQN，並產生 training_log.md。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym
from collections import deque

from agent import DQNAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

ACTION_NAMES = {0: "←（推左）", 1: "→（推右）"}

CONFIG = {
    "env_id":          "CartPole-v1",
    "total_steps":     150_000,
    "lr":              1e-3,
    "gamma":           0.99,
    "buffer_size":     50_000,
    "batch_size":      64,
    "target_update":   500,
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


# ── 輔助：checkpoint metadata ─────────────────────────────────────────

def _meta_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "train_meta.json")


def _load_meta(ckpt_dir: str) -> dict:
    p = _meta_path(ckpt_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_meta(ckpt_dir: str, meta: dict) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(_meta_path(ckpt_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ── 輔助：增量寫 log ──────────────────────────────────────────────────

def _append_log(path: str, content: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)


# ── Log 段落建構 ──────────────────────────────────────────────────────

def _build_header(config: dict) -> str:
    return f"""# DQN 訓練日誌

環境：{config['env_id']}（最大 500 步）
訓練步數：{config['total_steps']:,}
學習率 α：{config['lr']}｜折扣因子 γ：{config['gamma']}
Replay Buffer：{config['buffer_size']:,} 筆｜Batch size：{config['batch_size']}
Target 更新頻率：每 {config['target_update']} 步｜ε：{config['epsilon_start']} → {config['epsilon_end']}（前 {config['epsilon_steps']:,} 步線性衰減）

---

## 說明

### DQN 兩大創新

**① 經驗回放（Experience Replay）**

```
每步：  D.push(s, a, r, s', done)           ← 存入回放緩衝區
每步：  batch = D.sample(64)                 ← 隨機抽取迷你批次
        梯度更新 ← 基於 batch（非當前步驟）  ← 打破時間相關性
```

**② 目標網路（Target Network）**

```
每 {config['target_update']} 步：Q_theta-  ←  Q_theta    ← 固定目標，避免追移動標靶
計算 TD 目標時：y = r + γ × max_a Q_theta-(s')            ← 用 target_net，非 online_net
```

### 損失函式

```
y_i = r_i + γ × max_{{a'}} Q_theta-(s'_i)    （若 done 則 y_i = r_i）
L   = (1/B) × Σ Huber(y_i − Q_theta(s_i, a_i))

Huber 損失：誤差小時 ≈ MSE（平滑梯度），誤差大時 ≈ L1（有界梯度，防止梯度爆炸）
```

### 網路結構（CartPole MLP）

```
輸入（4 維：位置、速度、角度、角速度）
→ 全連線 256 → LayerNorm → ReLU
→ 全連線 256 → LayerNorm → ReLU
→ 全連線 2（輸出：Q(s, ←) 和 Q(s, →)）
```
"""


def _build_first_update_section(d: dict, config: dict) -> str:
    return f"""---

## 緩衝區填充階段（前 {config['learning_starts']:,} 步）

前 {config['learning_starts']:,} 步：ε = 1.0，完全隨機動作，**只收集資料不更新網路**。

| 時刻 | 事件 |
|:---|:---|
| 第 1 步 | 開始收集，緩衝區空（0 筆）|
| 第 {config['learning_starts']:,} 步 | 緩衝區達到 {config['learning_starts']:,} 筆，**首次觸發更新** |

### 第一次網路更新（步數 {d['step']:,}）

| 指標 | 數值 |
|:---|:---|
| 緩衝區大小 | {d['buffer_size']:,} 筆 |
| 當前 ε | {d['epsilon']:.4f} |
| 初始 loss | {d['loss']:.6f} |
| 初始 mean Q | {d['mean_q']:.4f} |

初始 loss 高、mean Q 接近 0——因為網路剛初始化，Q 值估計完全不準。
loss 和 mean Q 會隨訓練逐步下降/收斂。
"""


def _build_first_good_section(d: dict, config: dict) -> str:
    T = d["ep_length"]
    show_n = min(10, T)

    step_rows = ["| 步驟 | x | θ（°）| 動作 | r |",
                 "|:---:|:---:|:---:|:---:|:---:|"]
    for i in range(show_n):
        s = d["ep_states"][i]
        a = d["ep_actions"][i]
        r = d["ep_rewards"][i]
        theta_deg = float(s[2]) * (180.0 / 3.14159)
        step_rows.append(
            f"| {i+1} | {s[0]:+.3f} | {theta_deg:+.2f}°"
            f" | {ACTION_NAMES[a]} | {r:.0f} |"
        )
    if T > show_n:
        step_rows.append(f"| ... | | | | |（共 {T} 步）|")
    step_table = "\n".join(step_rows)

    return f"""---

## 第一個優秀集數（第 {d['ep_num']} 集，步數 {d['step']:,}，回報 = {d['ep_return']:.0f}）

此集共持續 **{T} 步**（CartPole 回報 = 存活步數）。

| 指標 | 數值 |
|:---|:---|
| 當時訓練步數 | {d['step']:,} |
| 當時 ε | {d['epsilon']:.4f} |
| 當時緩衝區大小 | {d['buffer_size']:,} 筆 |

### 前 {show_n} 步詳情

{step_table}

**重點說明**

- ε = {d['epsilon']:.4f}：大多數動作仍由策略選擇，少部分隨機探索。
- 桿子角度（θ）一直保持在 ±12° 以內才能存活這麼多步。
- DQN 已學會「桿子往哪邊傾就往哪邊推」的基本控制邏輯。
"""


def _milestone_block(step, eval_mean, eval_std, recent_mean, eps,
                     buf_size, metrics, config) -> str:
    loss_str = f"{metrics.get('loss', float('nan')):.6f}" if metrics else "N/A"
    meanq_str = f"{metrics.get('mean_q', float('nan')):.4f}" if metrics else "N/A"
    return f"""---

## 步數 {step:,} 進度快照

| 指標 | 數值 |
|:---|:---|
| eval 回報（10 集均值）| {eval_mean:.1f} ± {eval_std:.1f} |
| 最近 {config['window']} 集均值 | {recent_mean:.1f} |
| 當前 ε | {eps:.4f} |
| 回放緩衝區大小 | {buf_size:,} / {config['buffer_size']:,} |
| 最新 loss（Huber）| {loss_str} |
| 最新 mean Q | {meanq_str} |
| 目標網路已同步次數 | {step // config['target_update']} |
"""


def _build_final_section(agent: DQNAgent, config: dict) -> str:
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

    return f"""---

## 最終 Q 值快照（訓練完成後）

訓練結束後，對幾個典型狀態查詢 Q 值，驗證策略是否符合直覺：

{q_table}

**解讀**：
- 桿往左傾時，Q(←) 應 > Q(→)（往左推讓桿回正）
- 桿往右傾時，Q(→) 應 > Q(←)（往右推讓桿回正）
- 若符合上述模式，代表網路已學到正確的控制邏輯。

---

## 訓練總結

| 設定 | 數值 |
|:---|:---|
| 總訓練步數 | {config['total_steps']:,} |
| 最終緩衝區大小 | {config['buffer_size']:,}（已滿）|
| 目標網路同步次數 | {config['total_steps'] // config['target_update']} |
| ε 衰減區間 | 前 {config['epsilon_steps']:,} 步 |

### DQN 相對於 Q-Learning 的提升

| 問題 | Q-Learning 的困境 | DQN 的解法 |
|:---|:---|:---|
| 連續狀態空間 | 表格存不下 | 神經網路泛化 |
| 樣本時間相關 | 梯度震盪 | Replay Buffer 隨機抽樣 |
| 移動學習目標 | Q 值發散 | Target Network 固定目標 |
"""


# ── 訓練主迴圈 ────────────────────────────────────────────────────────

def train(config: dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, config["checkpoint_dir"])
    log_path = os.path.join(base_dir, "training_log.md")

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = DQNAgent(
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

    logger = Logger(log_dir="runs", run_name=f"dqn_{config['env_id']}")

    # ── 嘗試接續 checkpoint ──────────────────────────────────────────
    meta = _load_meta(ckpt_dir)
    resume_step = meta.get("train_step", 0)
    first_good_logged = meta.get("first_good_logged", False)
    first_update_logged = meta.get("first_update_logged", False)
    ep_num = meta.get("ep_num", 0)

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
    ep_return = 0.0
    ep_length = 0
    ep_states: list = [obs.copy()]
    ep_actions: list = []
    ep_rewards: list = []
    recent_returns = deque(maxlen=config["window"])
    last_metrics: dict = {}

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
            ep_return = 0.0
            ep_length = 0
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
                        "epsilon": metrics["epsilon"],
                    }, config))

                if step % config["log_freq"] == 0:
                    logger.log_scalars(metrics, step)

        # ── 評估 ──────────────────────────────────────────────────────
        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=10)
            logger.log_scalar("eval/mean_return", mean_r, step)
            eps = agent.epsilon_schedule.get(step)
            recent_mean = float(np.mean(recent_returns)) if recent_returns else 0.0
            print(f"步數 {step:8,}  eval={mean_r:.1f}±{std_r:.1f}  "
                  f"recent{config['window']}={recent_mean:.1f}  "
                  f"ε={eps:.3f}  buf={len(agent.buffer):,}")

        # ── 里程碑：寫 log + 存 checkpoint ────────────────────────────
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
