"""
在 CartPole-v1 上訓練 Rainbow DQN。

Rainbow 整合了 6 項改進：
    1. Double DQN      — 解耦動作選擇與 Q 值評估，消除高估偏差
    2. PER             — 按 |TD 誤差|^α 優先取樣，提升樣本效率
    3. Dueling Network — 分離 V(s) 與 A(s,a)，更高效更新狀態價值
    4. N-步回報        — G_t = r + γr' + ... + γ^{n-1}r^{n-1} + γ^n V(s_n)
    5. 分散式 RL(C51)  — 學習回報分佈而非純量期望，更豐富的監督訊號
    6. NoisyNet        — 網路內建可學習雜訊取代 ε-greedy，更有效率的探索

執行方式：
    python train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym
from collections import deque

from agent import RainbowAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate

# ──────────────────────────────────────────────
CONFIG = {
    "env_id":               "CartPole-v1",
    "total_steps":          150_000,
    "lr":                   6.25e-5,
    "gamma":                0.99,
    "buffer_size":          50_000,
    "batch_size":           32,
    "target_update":        2_000,
    "n_atoms":              51,
    "v_min":                0.0,
    "v_max":                500.0,
    "n_step":               3,
    "per_alpha":            0.5,
    "per_beta_start":       0.4,
    "per_beta_anneal_steps": 150_000,
    "learning_starts":      2_000,
    "milestone_freq":       50_000,
    "eval_n_eps":           10,
    "first_good_thr":       200,
    "checkpoint_dir":       "checkpoints",
    "log_path":             "training_log.md",
    "device":               "cuda" if torch.cuda.is_available() else "cpu",
    "seed":                 42,
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

def _get_rainbow_info(agent, states_list):
    """回傳每個測試狀態的 Q(←), Q(→), peak atom 與選擇動作。"""
    agent.online_net.eval()
    results = []
    for state_vec, label in states_list:
        s = torch.FloatTensor(state_vec).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            probs = agent.online_net(s)           # (1, A, n_atoms)
            q_vals = agent.online_net.get_q_values(s)  # (1, A)
            # 最高機率原子（最可能的回報值）
            peak_left  = agent.online_net.support[probs[0, 0].argmax()].item()
            peak_right = agent.online_net.support[probs[0, 1].argmax()].item()
        best = q_vals[0].argmax().item()
        results.append({
            "label":      label,
            "q_left":     float(q_vals[0, 0].item()),
            "q_right":    float(q_vals[0, 1].item()),
            "peak_left":  float(peak_left),
            "peak_right": float(peak_right),
            "action":     "←" if best == 0 else "→",
        })
    return results


# ──────────────────────────────────────────────
# Markdown builders
# ──────────────────────────────────────────────

def _build_header(config):
    delta_z = (config['v_max'] - config['v_min']) / (config['n_atoms'] - 1)
    return f"""# Rainbow DQN 訓練日誌

環境：{config['env_id']}（最大 500 步）｜訓練步數：{config['total_steps']:,}
學習率 α：{config['lr']}｜折扣因子 γ：{config['gamma']}
Replay Buffer：{config['buffer_size']:,} 筆｜Batch size：{config['batch_size']}
Target 更新：每 {config['target_update']} 步｜N-步回報：n={config['n_step']}
C51 原子數：{config['n_atoms']}｜支撐範圍：[{config['v_min']}, {config['v_max']}]｜原子間距：{delta_z:.4f}
PER α：{config['per_alpha']}｜β：{config['per_beta_start']} → 1.0（前 {config['per_beta_anneal_steps']:,} 步）

---

## 核心概念

### Rainbow = 6 項改進的整合

```
改進項              貢獻
──────────────────────────────────────────────────────────────
1. Double DQN    解耦選擇（線上網路）與評估（目標網路）→ 消除高估偏差
2. PER           按 |TD 誤差|^α 優先取樣 → 樣本效率提升
3. Dueling       Q = V(s) + A(s,a) - mean(A) → 狀態價值更高效更新
4. N-步回報      G = r + γr' + γ²r'' + γ³V(s₃) → 更遠的信用分配
5. C51（分散式） 學習回報的完整分佈，而非純量期望值
6. NoisyNet      可學習雜訊 σ 取代 ε-greedy → 更智慧的探索
```

### C51 分散式 RL：學習回報分佈

```
傳統 DQN：Q(s,a) = E[G_t | s_t=s, a_t=a]    ← 純量期望值

C51：     Z(s,a) = 在 {{z_1,...,z_51}} 上的機率分佈
          z_i = v_min + (i-1) × Δz          Δz = {delta_z:.4f}
          z_1 = {config['v_min']:.1f}（最小回報）, z_51 = {config['v_max']:.1f}（最大回報）

         優點：保留回報的不確定性資訊（方差、多峰分佈等）
         學習目標：最小化投影後的分佈與目標分佈的 KL 散度
```

### NoisyNet：引數化探索

```
普通線性層：y = Wx + b                  ← 確定性
NoisyLinear：y = (μ_w + σ_w ⊙ ε_w)x + (μ_b + σ_b ⊙ ε_b)

μ：可學習均值（策略主體）
σ：可學習雜訊強度（自動調整探索力度）
ε：每次前向傳遞重新取樣的隨機雜訊

好處：
  - 探索隨狀態而異（比 ε-greedy 更精細）
  - 不需要手動設計 ε 衰減排程
  - 訓練後 σ 自動趨近 0（探索自然消退）
```

### 貝爾曼分佈投影（C51 目標計算）

```
1. 線上網路選最佳動作（Double DQN）：
   a* = argmax_a E[Z_online(s', a)]

2. 目標網路取分佈（N-步折扣）：
   p(z_j | s', a*) from target net

3. 將目標原子投影至支撐集：
   z'_j = clip(r + γ^n * z_j, v_min, v_max)
   分配機率質量到相鄰原子

4. 最小化交叉熵（線上分佈 vs 投影目標）：
   L = -Σ_j m_j * log p(z_j | s, a)
   （使用 PER IS 權重加權）
```

---

"""


def _build_first_update_section(d, config):
    return f"""---

## 緩衝區填充階段（前 {config['learning_starts']:,} 步）

前 {config['learning_starts']:,} 步：NoisyNet 探索，**只收集資料不更新網路**。
（Rainbow 無 ε-greedy；NoisyNet 在取樣動作時自動新增雜訊）

### 第一次網路更新（步數 {d['step']:,}）

| 指標 | 數值 |
|:---|:---|
| 緩衝區大小 | {d['buffer_size']:,} 筆 |
| 當前 β（IS 修正）| {d['beta']:.4f} |
| 初始 loss（交叉熵）| {d['loss']:.6f} |
| 初始 mean Q | {d['mean_q']:.4f} |

初始分佈接近均勻（51 個原子各有 1/51 ≈ 0.0196 機率），
loss ≈ -log(1/51) ≈ 3.93；mean Q 接近支撐集中點。
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
| 緩衝區 | {d['buffer_size']:,} 筆 |
| 當前 β | {d['beta']:.4f} |

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
| 緩衝區大小 | {d['buffer_size']:,} / {config['buffer_size']:,} |
| 最新 loss（交叉熵）| {d['loss']:.6f} |
| 最新 mean Q | {d['mean_q']:.4f} |
| 當前 β（IS 修正進度）| {d['beta']:.4f}（已完成 {d['beta_pct']:.0f}%）|
| 目標網路已同步次數 | {d['target_updates']:,} |
---

"""


def _build_final_section(agent, config):
    results = _get_rainbow_info(agent, TEST_STATES)

    rows = ""
    for r in results:
        rows += (
            f"| {r['label']} "
            f"| {r['q_left']:.3f} "
            f"| {r['q_right']:.3f} "
            f"| {r['peak_left']:.1f} "
            f"| {r['peak_right']:.1f} "
            f"| {r['action']}（{'推左' if r['action'] == '←' else '推右'}）|\n"
        )

    return f"""---

## 最終 Q 值快照（訓練完成後）

| 測試狀態 | Q(←) | Q(→) | Peak(←) | Peak(→) | 選擇動作 |
|:---|:---:|:---:|:---:|:---:|:---:|
{rows}
**解讀重點：**
- `Q(←)/Q(→)`：回報分佈的期望值（E[Z(s,a)] = Σ p_i × z_i）
- `Peak(←/→)`：最高機率原子的位置（最可能的回報值）
- C51 的 Q 值比傳統 DQN 更「有意義」：包含分佈形狀資訊
- 桿往左傾 → Q(←) 應 > Q(→)；桿往右傾 → Q(→) 應 > Q(←)

---

## 訓練總結

| 設定 | 數值 |
|:---|:---|
| 總訓練步數 | {config['total_steps']:,} |
| C51 原子數 | {config['n_atoms']} |
| N-步回報 n | {config['n_step']} |
| PER α | {config['per_alpha']} |
| 最終 β | {agent.buffer.beta:.4f} |
| 目標網路更新間隔 | {config['target_update']} 步 |

### Rainbow 各改進項貢獻對比（論文資料，Atari-57 中位數）

| 組合 | 人類標準化分數 |
|:---|:---:|
| DQN（基線）| 79% |
| + Double DQN | 84% |
| + PER | 96% |
| + Dueling | 103% |
| + N-步回報 | 110% |
| + NoisyNet | 117% |
| + C51（完整 Rainbow）| **223%** |

### 四演演算法比較（CartPole-v1，150K 步）

| 演演算法 | 探索方式 | 目標公式 | 分散式 | 特點 |
|:---|:---:|:---:|:---:|:---|
| Double DQN | ε-greedy | TD(1) | 否 | 基線，消除高估 |
| PER-DQN | ε-greedy | TD(1) | 否 | 優先取樣 |
| Dueling DQN | ε-greedy | TD(1) | 否 | V/A 分解 |
| A3C | 熵獎勵 | n-步（線上）| 否 | 演員-評論家 |
| **Rainbow** | **NoisyNet** | **n-步+分散式** | **是** | **六項整合** |
"""


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train(config: dict) -> RainbowAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = RainbowAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        target_update=config["target_update"],
        n_atoms=config["n_atoms"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        n_step=config["n_step"],
        alpha=config["per_alpha"],
        beta=config["per_beta_start"],
        beta_anneal_steps=config["per_beta_anneal_steps"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"rainbow_{config['env_id']}")

    best_return = -float("inf")

    obs, _ = env.reset()
    ep_return = ep_length = episode = 0
    recent100 = deque(maxlen=100)
    step_details = []

    first_update_data = None
    first_good_ep_data = None
    milestone_data = {}
    next_milestone = config["milestone_freq"]
    last_metrics = {"loss": 0.0, "mean_q": 0.0, "beta": config["per_beta_start"]}

    print(f"訓練中：{config['env_id']}（Rainbow DQN），共 {config['total_steps']:,} 步...")

    for step in range(1, config["total_steps"] + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 記錄前 10 步（用於第一個優秀集數）
        if len(step_details) < 10:
            step_details.append({
                "x":     float(obs[0]),
                "theta": float(np.degrees(obs[2])),
                "action": int(action),
            })

        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_return += reward
        ep_length += 1

        if done:
            episode += 1
            recent100.append(ep_return)
            logger.log_episode(ep_return, ep_length, step)

            if first_good_ep_data is None and ep_return >= config["first_good_thr"]:
                first_good_ep_data = {
                    "episode":    episode,
                    "step":       step,
                    "ep_return":  ep_return,
                    "steps":      step_details,
                    "buffer_size": len(agent.buffer),
                    "beta":       agent.buffer.beta,
                }

            obs, _ = env.reset()
            ep_return = ep_length = 0
            step_details = []

        # 網路更新
        if step >= config["learning_starts"]:
            metrics = agent.update()
            if metrics:
                last_metrics = metrics
                logger.log_scalars(metrics, step)

                if first_update_data is None:
                    first_update_data = {
                        "step":        step,
                        "buffer_size": len(agent.buffer),
                        "loss":        metrics["loss"],
                        "mean_q":      metrics["mean_q"],
                        "beta":        metrics["beta"],
                    }

        # 里程碑
        if step == next_milestone:
            eval_mean, eval_std = evaluate(agent, eval_env, n_episodes=config["eval_n_eps"])
            logger.log_scalar("eval/mean_return", eval_mean, step)
            beta_pct = (agent.buffer.beta - config["per_beta_start"]) / (1.0 - config["per_beta_start"]) * 100
            print(
                f"步數 {step:>7,}  "
                f"eval={eval_mean:.1f}±{eval_std:.1f}  "
                f"recent100={np.mean(recent100):.1f}  "
                f"β={agent.buffer.beta:.4f}"
            )
            if eval_mean > best_return:
                best_return = eval_mean
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳：{eval_mean:.1f}，已儲存")

            milestone_data[step] = {
                "eval_mean":     eval_mean,
                "eval_std":      eval_std,
                "recent100":     float(np.mean(recent100)),
                "buffer_size":   len(agent.buffer),
                "loss":          last_metrics["loss"],
                "mean_q":        last_metrics["mean_q"],
                "beta":          agent.buffer.beta,
                "beta_pct":      beta_pct,
                "target_updates": agent.total_steps // config["target_update"],
            }

            agent.save(config["checkpoint_dir"])
            print(f"  → checkpoint 已存（步數 {step:,}）")
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
