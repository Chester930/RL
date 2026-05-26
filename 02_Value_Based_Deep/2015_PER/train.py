"""在 CartPole-v1 上訓練 Double DQN + PER，並產生 training_log.md。"""

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

from agent import PERDQNAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

ACTION_NAMES = {0: "←（推左）", 1: "→（推右）"}

CONFIG = {
    "env_id":            "CartPole-v1",
    "total_steps":       300_000,
    "lr":                5e-4,
    "gamma":             0.99,
    "buffer_size":       100_000,
    "batch_size":        64,
    "target_update":     200,
    "epsilon_start":     1.0,
    "epsilon_end":       0.01,
    "epsilon_steps":     50_000,
    "per_alpha":         0.4,
    "per_beta_start":    0.4,
    "per_beta_anneal_steps": 150_000,  # 整個訓練期間線性退火到 1.0
    "learning_starts":   2_000,        # PER 需要更多初始資料（SumTree 取樣品質）
    "log_freq":          1_000,
    "eval_freq":         10_000,
    "milestone_freq":    50_000,
    "window":            100,
    "good_threshold":    200,
    "checkpoint_dir":    "checkpoints",
    "best_checkpoint_dir": "best_checkpoints",
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
    "seed":              42,
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


# ── Log 段落建構 ──────────────────────────────────────────────────────

def _build_header(config):
    return f"""# PER-DQN 訓練日誌（Double DQN + 優先經驗回放）

環境：{config['env_id']}（最大 500 步）｜訓練步數：{config['total_steps']:,}
學習率 α：{config['lr']}｜折扣因子 γ：{config['gamma']}
Replay Buffer：{config['buffer_size']:,} 筆｜Batch size：{config['batch_size']}
Target 更新：每 {config['target_update']} 步｜ε：{config['epsilon_start']} → {config['epsilon_end']}（前 {config['epsilon_steps']:,} 步）
PER α（優先指數）：{config['per_alpha']}｜β：{config['per_beta_start']} → 1.0（前 {config['per_beta_anneal_steps']:,} 步）

---

## 核心概念

### 均勻回放的問題

標準 DQN 對所有轉移資料一視同仁，均勻取樣：

```
問題 ①：許多「已學好」的轉移（TD 誤差 ≈ 0）仍佔用寶貴的取樣配額
問題 ②：罕見但重要的轉移（TD 誤差大、策略尚未學會）取樣頻率不足
→ 樣本效率低落，尤其在稀疏獎勵或非均勻資料分佈時
```

### PER 的解法：比例優先取樣（Proportional Prioritization）

```
取樣機率 P(i) ∝ |TD 誤差_i + ε|^α

α = 0  → 均勻取樣（退化為標準 DQN）
α = 1  → 完全依優先權取樣
α = 0.6 → 折衷：重視高 TD 誤差，但仍保留一定隨機性

實作：SumTree 資料結構  → O(log N) 取樣與更新
```

### 重要性取樣（IS）修正偏差

```
優先取樣 ≠ 均勻取樣 → 梯度估計有偏差（某些樣本被過度強調）

IS 修正權重：w_i = (N × P(i))^{{-β}}     （β 從 0.4 逐漸增加到 1.0）

β = 0   → 不修正（完全偏差）
β = 1   → 完全修正（無偏梯度估計）

訓練初期：β 較小，允許優先學習（偏差換效率）
訓練末期：β → 1.0，確保收斂時梯度無偏
```

### 結合 Double DQN

```
PER 決定「取哪些樣本」（取樣策略）
Double DQN 決定「如何計算目標」（目標公式）

y_i = r_i + γ × Q_target(s'_i, argmax_a Q_online(s'_i, a))

兩者正交，可直接疊加：
  PER 提升樣本效率（學得更快）
  Double DQN 消除高估偏差（學得更正確）
```

---

"""


def _build_first_update_section(d, config):
    return f"""---

## 緩衝區填充階段（前 {config['learning_starts']:,} 步）

前 {config['learning_starts']:,} 步：ε = 1.0，完全隨機動作，**只收集資料不更新網路**。
PER 需要更多初始資料（{config['learning_starts']:,} 筆），確保 SumTree 取樣具代表性。

### 第一次網路更新（步數 {d['step']:,}）

| 指標 | 數值 |
|:---|:---|
| 緩衝區大小 | {d['buffer_size']:,} 筆 |
| 當前 ε | {d['epsilon']:.4f} |
| 當前 β | {d['beta']:.4f} |
| 初始 loss | {d['loss']:.6f} |
| 初始 mean \\|TD 誤差\\| | {d['mean_td_error']:.4f} |

初始 TD 誤差通常較大——新轉移以最大優先權插入，尚未被更新過。
隨訓練進行，常見轉移的 TD 誤差會降低，優先取樣更聚焦於「尚未學好」的轉移。
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
| β（IS 修正進度）| {d['beta']:.4f} |
| 緩衝區 | {d['buffer_size']:,} 筆 |

### 前 {show_n} 步詳情

{table}
"""


def _milestone_block(step, eval_mean, eval_std, recent_mean, eps,
                     buf_size, metrics, config):
    loss_str  = f"{metrics.get('loss', float('nan')):.6f}" if metrics else "N/A"
    td_str    = f"{metrics.get('mean_td_error', float('nan')):.4f}" if metrics else "N/A"
    beta_str  = f"{metrics.get('beta', float('nan')):.4f}" if metrics else "N/A"
    beta_pct  = float(metrics.get('beta', 0.4)) / 1.0 * 100 if metrics else 0.0
    return f"""---

## 步數 {step:,} 進度快照

| 指標 | 數值 |
|:---|:---|
| eval 回報（10 集均值）| {eval_mean:.1f} ± {eval_std:.1f} |
| 最近 {config['window']} 集均值 | {recent_mean:.1f} |
| 當前 ε | {eps:.4f} |
| 緩衝區大小 | {buf_size:,} / {config['buffer_size']:,} |
| 最新 loss | {loss_str} |
| 平均 \\|TD 誤差\\| | {td_str} |
| 當前 β（IS 修正進度）| {beta_str}（已完成 {beta_pct:.0f}%）|
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

    final_beta = agent.buffer.beta

    return f"""---

## 最終 Q 值快照（訓練完成後）

{q_table}

**解讀**：
- 桿往左傾時，Q(←) 應 > Q(→)（往左推讓桿回正）
- 桿往右傾時，Q(→) 應 > Q(←)（往右推讓桿回正）

---

## PER 訓練總結

| 指標 | 數值 |
|:---|:---|
| 總訓練步數 | {config['total_steps']:,} |
| 最終 β 值 | {final_beta:.4f}（目標 1.0）|
| PER α | {config['per_alpha']}（優先取樣強度）|
| 目標網路同步次數 | {config['total_steps'] // config['target_update']} |

### PER 相對於標準 DQN 的改進

| 面向 | 標準 DQN | PER-DQN |
|:---|:---|:---|
| 取樣方式 | 均勻隨機 | 按 \\|TD 誤差\\|^α 比例 |
| 樣本效率 | 低（所有轉移同等對待）| 高（聚焦難以學好的轉移）|
| 偏差修正 | 不需要 | IS 權重 w_i = (N·P(i))^{{-β}} |
| 資料結構 | deque | SumTree（O(log N) 取樣）|
| 超引數 | 無 | α（優先強度）、β（IS 退火）|

### β 退火的作用

```
訓練初期（β ≈ 0.4）：允許較大的 IS 偏差
  → 優先學習可加速早期收斂，輕微偏差可接受

訓練末期（β → 1.0）：完全消除 IS 偏差
  → 保證梯度估計在收斂時漸近無偏，提升最終策略品質
```
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

    agent = PERDQNAgent(
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
        alpha=config["per_alpha"],
        beta_start=config["per_beta_start"],
        beta_anneal_steps=config["per_beta_anneal_steps"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"per_dqn_{config['env_id']}")

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
        print(f"訓練中：{config['env_id']}（PER）共 {config['total_steps']:,} 步...")

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
                    "beta": agent.buffer.beta,
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
            if metrics and np.isnan(metrics.get("loss", 0)):
                raise RuntimeError(f"NaN loss detected at step {step}, stopping training.")
            if metrics:
                last_metrics = metrics

                if not first_update_logged:
                    first_update_logged = True
                    _append_log(log_path, _build_first_update_section({
                        "step": step,
                        "buffer_size": len(agent.buffer),
                        "loss": metrics["loss"],
                        "mean_td_error": metrics["mean_td_error"],
                        "beta": metrics["beta"],
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
            beta = agent.buffer.beta
            if step > 10_000 and mean_r < best_eval * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_eval:.1f}")
            if mean_r > best_eval:
                best_eval = mean_r
                agent.save(best_ckpt_dir)
                print(f"步數 {step:8,}  eval={mean_r:.1f}±{std_r:.1f}  "
                      f"recent{config['window']}={recent_mean:.1f}  "
                      f"ε={eps:.3f}  β={beta:.3f}  ★ best={best_eval:.1f}")
            else:
                print(f"步數 {step:8,}  eval={mean_r:.1f}±{std_r:.1f}  "
                      f"recent{config['window']}={recent_mean:.1f}  "
                      f"ε={eps:.3f}  β={beta:.3f}")

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
