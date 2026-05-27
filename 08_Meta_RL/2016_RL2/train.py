"""
在 N 臂 Bandit 任務上訓練 RL²。

Meta-learning 設定：
  每個「任務」= 一個隨機 N 臂 Bandit（每臂的獎勵機率不同）
  每個任務內：agent 有 T_steps 步探索並剝削
  訓練目標：讓 GRU 學會快速識別哪臂最好，並切換到剝削模式

評估指標：
  最優臂平均命中率（後半段）= GRU 有多快找到最佳臂

Baseline 對比：
  - 隨機策略：命中率 = 1/n_arms
  - RL² (訓練後)：命中率應顯著超越隨機
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import pickle
import numpy as np
import torch

from agent import RL2Agent

RESUME_DIR = "checkpoints/resume"


# ------------------------------------------------------------------
# Bandit 環境
# ------------------------------------------------------------------

class BanditEnv:
    """
    N 臂 Bernoulli Bandit。

    每個任務（reset()）隨機重新採樣每臂的獎勵機率。
    Agent 每步選擇一臂，以 Bernoulli(p_arm) 獲得獎勵。
    """

    def __init__(self, n_arms: int = 5):
        self.n_arms = n_arms
        self.probs = None
        self.best_arm = None

    def reset(self) -> None:
        """採樣新任務的臂機率（固定整個任務期間）。"""
        self.probs = np.random.uniform(0.0, 1.0, self.n_arms)
        self.best_arm = int(np.argmax(self.probs))

    def step(self, action: int):
        reward = float(np.random.random() < self.probs[action])
        return reward

    @property
    def optimal_prob(self) -> float:
        return float(self.probs[self.best_arm])


# ------------------------------------------------------------------
# 輔助：收集一個任務的 rollout
# ------------------------------------------------------------------

def collect_task(agent: RL2Agent, env: BanditEnv, t_steps: int):
    """
    在一個任務內收集 T 步資料。

    回傳 task_batch dict（agent.update 所需格式）
    以及本次任務的累積獎勵與最優臂命中率。
    """
    env.reset()
    agent.reset_hidden()

    n_arms = env.n_arms
    inputs, actions, rewards_list, dones_list, log_probs, values = [], [], [], [], [], []

    prev_action, prev_reward, prev_done = 0, 0.0, 0.0
    total_reward = 0.0
    best_arm_hits = 0

    for t in range(t_steps):
        # 組裝 GRU 輸入（使用前一步資訊）
        onehot = np.zeros(n_arms, dtype=np.float32)
        onehot[prev_action] = 1.0
        inp = np.concatenate([onehot, [prev_reward, prev_done]])

        action, log_prob, value = agent.select_action()
        reward = env.step(action)
        done = (t == t_steps - 1)

        agent.observe(action, reward, done)

        inputs.append(inp)
        actions.append(action)
        rewards_list.append(reward)
        dones_list.append(float(done))
        log_probs.append(log_prob)
        values.append(value)

        total_reward += reward
        if action == env.best_arm:
            best_arm_hits += 1

        prev_action = action
        prev_reward = reward
        prev_done = float(done)

    return {
        "inputs":    np.array(inputs, dtype=np.float32),
        "actions":   np.array(actions, dtype=np.int64),
        "rewards":   np.array(rewards_list, dtype=np.float32),
        "dones":     np.array(dones_list, dtype=np.float32),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "values":    np.array(values, dtype=np.float32),
    }, total_reward, best_arm_hits / t_steps


# ------------------------------------------------------------------
# 評估
# ------------------------------------------------------------------

def evaluate_meta(agent: RL2Agent, n_arms: int, t_steps: int, n_tasks: int = 50):
    """
    在新任務上評估 RL² 的「後期最優臂命中率」。

    後半段（步數 t_steps//2 之後）的命中率代表 GRU 識別任務後的剝削品質。
    """
    env = BanditEnv(n_arms)
    hit_rates_late = []

    for _ in range(n_tasks):
        env.reset()
        agent.reset_hidden()
        prev_action, prev_reward, prev_done = 0, 0.0, 0.0
        late_hits = 0
        half = t_steps // 2

        for t in range(t_steps):
            agent._prev_action = prev_action
            agent._prev_reward = prev_reward
            agent._prev_done = prev_done
            action, _, _ = agent.select_action(evaluate=True)
            reward = env.step(action)
            done = (t == t_steps - 1)
            agent.observe(action, reward, done)
            if t >= half and action == env.best_arm:
                late_hits += 1
            prev_action, prev_reward, prev_done = action, reward, float(done)

        hit_rates_late.append(late_hits / (t_steps - half))

    return float(np.mean(hit_rates_late))


# ------------------------------------------------------------------
# 訓練
# ------------------------------------------------------------------

def train(config: dict) -> RL2Agent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_arms = config["n_arms"]
    t_steps = config["t_steps_per_task"]
    n_tasks_per_update = config["n_tasks_per_update"]
    n_updates = config["n_updates"]

    agent = RL2Agent(
        n_arms=n_arms,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        clip_eps=config["clip_eps"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        n_epochs=config["n_epochs"],
        device=config["device"],
    )

    env = BanditEnv(n_arms)
    best_hit_rate = 0.0
    start_update = 1

    # 接續訓練
    resume_meta = os.path.join(RESUME_DIR, "train_meta.pkl")
    resume_ckpt = os.path.join(RESUME_DIR, "rl2.pt")
    if os.path.exists(resume_ckpt) and os.path.exists(resume_meta):
        agent.load_resume(RESUME_DIR)
        with open(resume_meta, "rb") as f:
            meta = pickle.load(f)
        start_update = meta["update"] + 1
        best_hit_rate = meta["best_hit_rate"]
        random.setstate(meta["random_state"])
        np.random.set_state(meta["np_state"])
        torch.set_rng_state(meta["torch_state"])
        print(f"[RESUME] 從更新 {meta['update']} 繼續，歷史最佳命中率 {best_hit_rate:.3f}")

    random_baseline = 1.0 / n_arms
    print(
        f"RL² on {n_arms}-Armed Bandit | {n_updates} updates × {n_tasks_per_update} tasks × {t_steps} steps/task"
        f" | 隨機基線命中率: {random_baseline:.3f}"
    )

    for update in range(start_update, n_updates + 1):
        # 收集多個任務的 rollout
        task_batches = []
        ep_rewards, ep_hits = [], []

        for _ in range(n_tasks_per_update):
            tb, total_r, hit_rate = collect_task(agent, env, t_steps)
            task_batches.append(tb)
            ep_rewards.append(total_r)
            ep_hits.append(hit_rate)

        metrics = agent.update(task_batches)

        if update % config["log_freq"] == 0:
            mean_r = np.mean(ep_rewards)
            mean_hit = np.mean(ep_hits)
            print(
                f"更新 {update:4d}/{n_updates} | "
                f"平均獎勵 {mean_r:.2f} | "
                f"最優臂命中率 {mean_hit:.3f} | "
                f"entropy {metrics.get('entropy', 0):.3f} | "
                f"KL {metrics.get('approx_kl', 0):.4f}"
            )

        if update % config["eval_freq"] == 0:
            hit_rate_late = evaluate_meta(agent, n_arms, t_steps)
            print(
                f"  >>> Eval 後半段命中率: {hit_rate_late:.3f} "
                f"（隨機基線: {random_baseline:.3f}）"
            )
            if hit_rate_late > best_hit_rate:
                best_hit_rate = hit_rate_late
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳命中率: {best_hit_rate:.3f}，已儲存")

        if update % config["save_freq"] == 0:
            agent.save_resume(RESUME_DIR)
            meta = {
                "update": update,
                "best_hit_rate": best_hit_rate,
                "random_state": random.getstate(),
                "np_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
            }
            with open(resume_meta, "wb") as f:
                pickle.dump(meta, f)
            print(f"  [RESUME] 暫停點已儲存（更新 {update}）")

    return agent


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        # 任務設定
        "n_arms": 5,                # Bandit 臂數
        "t_steps_per_task": 50,     # 每任務步數（= 50 次臂拉動）

        # 訓練規模
        "n_updates": 2000,          # PPO 更新次數（延長至 2000，前 1000 已有 checkpoint）
        "n_tasks_per_update": 30,   # 每次更新前收集的任務數（↑30 穩定梯度估計）

        # RL² 網路
        "hidden_dim": 64,

        # PPO 超參數
        "lr": 1e-3,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": 0.02,           # ↓0.05→0.02，後期減少熵鼓勵剝削
        "vf_coef": 0.5,
        "n_epochs": 4,

        # 日誌 / 存檔
        "log_freq": 20,
        "eval_freq": 100,
        "save_freq": 200,

        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
