"""
在 SafePendulum 上訓練 CPO（Constrained Policy Optimization）。

與 PPO-Lagrangian 使用相同環境，方便直接對比結果：
  - 同一個安全約束（|θ̇| > 2.0 rad/s → cost=1）
  - 同一個 cost_limit（d = 25 per episode）
  - 不同更新機制（TRPO 信任域 vs PPO 剪裁）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import pickle
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import CPOAgent
from common.utils.logger import Logger

RESUME_DIR = "checkpoints/resume"


# ------------------------------------------------------------------
# 安全環境（與 PPO-Lagrangian 相同定義）
# ------------------------------------------------------------------

class SafePendulumEnv(gym.Wrapper):
    """Pendulum-v1 + 代價信號：|θ̇| > omega_threshold → cost=1。"""

    def __init__(self, omega_threshold: float = 2.0):
        super().__init__(gym.make("Pendulum-v1"))
        self.omega_threshold = omega_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["cost"] = float(abs(obs[2]) > self.omega_threshold)
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------
# 評估
# ------------------------------------------------------------------

def evaluate_safe(agent: CPOAgent, env: SafePendulumEnv, n_episodes: int = 10):
    reward_returns, cost_returns = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = ep_c = 0.0
        while True:
            action = agent.select_action(obs, evaluate=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            ep_c += info.get("cost", 0.0)
            if term or trunc:
                break
        reward_returns.append(ep_r)
        cost_returns.append(ep_c)
    return float(np.mean(reward_returns)), float(np.std(reward_returns)), float(np.mean(cost_returns))


# ------------------------------------------------------------------
# 訓練
# ------------------------------------------------------------------

def train(config: dict) -> CPOAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    best_return = -float("inf")

    env = SafePendulumEnv(config["omega_threshold"])
    eval_env = SafePendulumEnv(config["omega_threshold"])
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = CPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        max_kl=config["max_kl"],
        cost_limit=config["cost_limit"],
        damping=config["damping"],
        cg_iters=config["cg_iters"],
        ls_iters=config["ls_iters"],
        lr_critic=config["lr_critic"],
        n_critic_epochs=config["n_critic_epochs"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name="cpo_SafePendulum")

    obs, _ = env.reset()
    ep_reward = ep_cost = 0.0
    global_step = 0
    ep_count = 0
    start_update = 1

    # 接續訓練
    resume_meta = os.path.join(RESUME_DIR, "train_meta.pkl")
    resume_ckpt = os.path.join(RESUME_DIR, "cpo.pt")
    if os.path.exists(resume_ckpt) and os.path.exists(resume_meta):
        agent.load_resume(RESUME_DIR)
        with open(resume_meta, "rb") as f:
            meta = pickle.load(f)
        start_update = meta["update"] + 1
        global_step = meta["global_step"]
        best_return = meta["best_return"]
        obs = meta["obs"]
        random.setstate(meta["random_state"])
        np.random.set_state(meta["np_state"])
        torch.set_rng_state(meta["torch_state"])
        print(f"[RESUME] 從更新 {meta['update']} 繼續，最佳 {best_return:.1f}")

    n_steps = config["n_steps_per_update"]
    total_updates = config["total_timesteps"] // n_steps
    print(
        f"CPO on SafePendulum | {total_updates} updates × {n_steps} steps | "
        f"cost_limit={config['cost_limit']} | max_kl={config['max_kl']}"
    )

    for update in range(start_update, total_updates + 1):
        ep_rewards_this = []
        ep_costs_this = []
        done = False

        for _ in range(n_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cost = info.get("cost", 0.0)

            agent.store(obs, action, reward, cost, done)
            obs = next_obs
            ep_reward += reward
            ep_cost += cost
            global_step += 1

            if done:
                ep_rewards_this.append(ep_reward)
                ep_costs_this.append(ep_cost)
                logger.log_episode(ep_reward, n_steps, global_step)
                logger.log_scalar("train/ep_cost", ep_cost, global_step)
                ep_reward = ep_cost = 0.0
                ep_count += 1
                obs, _ = env.reset()

        metrics = agent.update(next_state=obs, last_done=done)

        if metrics and update % config["log_freq"] == 0 and ep_rewards_this:
            mean_r = np.mean(ep_rewards_this)
            mean_c = np.mean(ep_costs_this)
            u_type = "TRPO" if metrics.get("update_type", 0) == 0 else "CPO"
            print(
                f"更新 {update:4d}/{total_updates} | "
                f"步數 {global_step:7d} | "
                f"獎勵 {mean_r:7.1f} | "
                f"代價/集 {mean_c:5.1f} | "
                f"KL {metrics.get('kl', 0):.4f} | "
                f"更新類型 {u_type}"
            )
            logger.log_scalar("train/kl", metrics.get("kl", 0), global_step)
            logger.log_scalar("train/mean_ep_cost", metrics.get("mean_ep_cost", 0), global_step)

        if update % config["eval_freq"] == 0:
            mean_r, std_r, mean_c = evaluate_safe(agent, eval_env)
            logger.log_scalar("eval/mean_reward", mean_r, global_step)
            logger.log_scalar("eval/mean_cost", mean_c, global_step)

            safe_flag = "✓ 安全" if mean_c <= config["cost_limit"] else "✗ 違規"
            print(
                f"  >>> Eval | 獎勵 {mean_r:.1f} ± {std_r:.1f} | "
                f"代價/集 {mean_c:.1f} (限 {config['cost_limit']}) {safe_flag}"
            )
            if global_step > 10_000 and mean_r < best_return * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_return:.1f}")
            if mean_r > best_return:
                best_return = mean_r
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳獎勵：{mean_r:.1f}，已儲存")

        if update % config["save_freq"] == 0:
            agent.save_resume(RESUME_DIR)
            meta = {
                "update": update,
                "global_step": global_step,
                "best_return": best_return,
                "obs": obs,
                "random_state": random.getstate(),
                "np_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
            }
            with open(resume_meta, "wb") as f:
                pickle.dump(meta, f)
            print(f"  [RESUME] 暫停點已儲存（更新 {update}，步數 {global_step}）")

    logger.close()
    env.close()
    eval_env.close()
    return agent


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        # 環境
        "omega_threshold": 2.0,
        "cost_limit": 25.0,

        # 訓練規模（CPO 每次 update 需計算自然梯度，比 PPO 慢）
        "total_timesteps": 300_000,
        "n_steps_per_update": 4096,    # 更多步數讓梯度估計更準確

        # CPO 超參數
        "max_kl": 0.01,                # 信任域半徑
        "damping": 0.1,                # Fisher 阻尼
        "cg_iters": 10,                # 共軛梯度迭代次數
        "ls_iters": 10,                # 回溯線搜尋步數

        # 評論家
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "lr_critic": 3e-4,
        "n_critic_epochs": 5,

        # 日誌 / 存檔
        "log_freq": 5,
        "eval_freq": 15,
        "save_freq": 30,               # 每 ~120K 步存一次接續點

        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
