"""
在 SafePendulum 上訓練 PPO-Lagrangian。

環境說明：
  SafePendulum = Pendulum-v1 + 安全代價信號
    代價 c_t = 1  若 |角速度 θ̇| > omega_threshold
    代價 c_t = 0  否則
  安全預算 d = cost_limit（每集最多允許 cost_limit 步違規）

實驗目標：
  在最大化累積獎勵的同時，讓每集違規步數 ≤ cost_limit。
  λ 自動調整：違規多時加重懲罰，違規少時放寬約束。
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

from agent import PPOLagrangianAgent
from common.utils.logger import Logger

RESUME_DIR = "checkpoints/resume"


# ------------------------------------------------------------------
# 安全環境包裝器
# ------------------------------------------------------------------

class SafePendulumEnv(gym.Wrapper):
    """
    在 Pendulum-v1 上加入代價信號。

    代價定義：若角速度 |θ̇| > omega_threshold，則 c_t = 1，否則 c_t = 0。
    代價透過 info["cost"] 回傳，不影響原始獎勵。

    狀態向量：[cos(θ), sin(θ), θ̇]
      → obs[2] = θ̇（角速度）
    """

    def __init__(self, omega_threshold: float = 2.0):
        super().__init__(gym.make("Pendulum-v1"))
        self.omega_threshold = omega_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = float(abs(obs[2]) > self.omega_threshold)
        info["cost"] = cost
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------
# 評估（同時追蹤獎勵和代價）
# ------------------------------------------------------------------

def evaluate_safe(agent: PPOLagrangianAgent, env: SafePendulumEnv, n_episodes: int = 10):
    """
    評估 agent 的確定性策略。

    回傳：
        mean_reward, std_reward, mean_cost（每集平均代價違規步數）
    """
    reward_returns = []
    cost_returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_cost = 0.0
        while True:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_cost += info.get("cost", 0.0)
            if terminated or truncated:
                break
        reward_returns.append(ep_reward)
        cost_returns.append(ep_cost)

    return (
        float(np.mean(reward_returns)),
        float(np.std(reward_returns)),
        float(np.mean(cost_returns)),
    )


# ------------------------------------------------------------------
# 訓練
# ------------------------------------------------------------------

def train(config: dict) -> PPOLagrangianAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    best_return = -float("inf")

    env = SafePendulumEnv(omega_threshold=config["omega_threshold"])
    eval_env = SafePendulumEnv(omega_threshold=config["omega_threshold"])
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = PPOLagrangianAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        lr=config["lr"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        n_minibatch=config["n_minibatch"],
        clip_eps=config["clip_eps"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        cost_limit=config["cost_limit"],
        lambda_init=config["lambda_init"],
        lr_lambda=config["lr_lambda"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name="ppo_lagrangian_SafePendulum")

    obs, _ = env.reset()
    ep_reward = ep_cost = ep_length = 0.0
    global_step = 0
    done = False
    start_update = 1

    # 自動偵測暫停點並繼續
    resume_meta = os.path.join(RESUME_DIR, "train_meta.pkl")
    resume_ckpt = os.path.join(RESUME_DIR, "ppo_lagrangian.pt")
    if os.path.exists(resume_ckpt) and os.path.exists(resume_meta):
        agent.load_resume(RESUME_DIR)
        with open(resume_meta, "rb") as f:
            meta = pickle.load(f)
        start_update  = meta["update"] + 1
        global_step   = meta["global_step"]
        best_return   = meta["best_return"]
        obs           = meta["obs"]
        done          = meta["done"]
        random.setstate(meta["random_state"])
        np.random.set_state(meta["np_state"])
        torch.set_rng_state(meta["torch_state"])
        print(f"[RESUME] 從更新 {meta['update']} 繼續，λ={agent.lambda_:.4f}，最佳 {best_return:.1f}")

    total_updates = config["total_timesteps"] // config["n_steps"]
    print(
        f"PPO-Lagrangian on SafePendulum | "
        f"{total_updates} updates × {config['n_steps']} steps | "
        f"cost_limit={config['cost_limit']} | "
        f"omega_threshold={config['omega_threshold']}"
    )

    for update in range(start_update, total_updates + 1):
        ep_rewards_this_rollout = []
        ep_costs_this_rollout = []

        for _ in range(config["n_steps"]):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cost = info.get("cost", 0.0)

            agent.store_transition(reward, cost, done)
            obs = next_obs
            ep_reward += reward
            ep_cost += cost
            ep_length += 1
            global_step += 1

            if done:
                ep_rewards_this_rollout.append(ep_reward)
                ep_costs_this_rollout.append(ep_cost)
                logger.log_episode(ep_reward, int(ep_length), global_step)
                logger.log_scalar("train/ep_cost", ep_cost, global_step)
                obs, _ = env.reset()
                ep_reward = ep_cost = ep_length = 0.0

        metrics = agent.update(next_state=obs, last_done=done)

        if metrics and np.isnan(metrics.get("loss", 0.0)):
            raise RuntimeError(f"NaN loss at step {global_step}")

        if metrics and update % config["log_freq_updates"] == 0:
            logger.log_scalars(metrics, global_step)
            logger.log_scalar("train/lambda", agent.lambda_, global_step)

            if ep_rewards_this_rollout:
                mean_r = np.mean(ep_rewards_this_rollout)
                mean_c = np.mean(ep_costs_this_rollout)
                print(
                    f"更新 {update:4d}/{total_updates} | "
                    f"步數 {global_step:7d} | "
                    f"獎勵 {mean_r:7.1f} | "
                    f"代價/集 {mean_c:5.1f} | "
                    f"λ {agent.lambda_:.4f} | "
                    f"KL {metrics.get('approx_kl', 0):.4f}"
                )

        if update % config["eval_freq_updates"] == 0:
            mean_r, std_r, mean_c = evaluate_safe(agent, eval_env)
            logger.log_scalar("eval/mean_reward", mean_r, global_step)
            logger.log_scalar("eval/mean_cost", mean_c, global_step)
            logger.log_scalar("eval/lambda", agent.lambda_, global_step)

            safe_flag = "✓ 安全" if mean_c <= config["cost_limit"] else "✗ 違規"
            print(
                f"  >>> Eval | 獎勵 {mean_r:.1f} ± {std_r:.1f} | "
                f"代價/集 {mean_c:.1f} (限 {config['cost_limit']}) {safe_flag} | "
                f"λ={agent.lambda_:.4f}"
            )

            if global_step > 10_000 and mean_r < best_return * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_return:.1f}")

            if mean_r > best_return:
                best_return = mean_r
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳獎勵：{mean_r:.1f}，已儲存")

        # 每 save_freq_updates 次更新儲存一次接續點
        if update % config["save_freq_updates"] == 0:
            agent.save_resume(RESUME_DIR)
            meta = {
                "update": update,
                "global_step": global_step,
                "best_return": best_return,
                "obs": obs,
                "done": done,
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
        "omega_threshold": 2.0,    # |θ̇| > 2.0 rad/s 視為違規
        "cost_limit": 25.0,        # 每集最多允許 25 步違規（共 200 步）

        # 訓練規模
        "total_timesteps": 300_000,
        "n_steps": 2048,

        # PPO 超參數
        "lr": 3e-4,
        "gamma": 0.99,
        "n_epochs": 10,
        "n_minibatch": 32,
        "clip_eps": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "vf_coef": 0.5,

        # 拉格朗日超參數
        "lambda_init": 0.0,        # 從 0 開始，由 λ 更新自動增加
        "lr_lambda": 0.05,         # λ 學習率（越大，安全調整越激進）

        # 日誌 / 存檔
        "log_freq_updates": 5,
        "eval_freq_updates": 20,
        "save_freq_updates": 50,   # 每 50 次更新（~100K 步）存一次接續點

        # 系統
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
