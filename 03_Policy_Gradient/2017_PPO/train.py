"""在 LunarLander-v2 或 CartPole-v1 上訓練 PPO。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import torch
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import PPOAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> PPOAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    best_return = -float("inf")

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        n_minibatch=config["n_minibatch"],
        clip_eps=config["clip_eps"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"ppo_{config['env_id']}")

    obs, _ = env.reset()
    ep_return = ep_length = 0
    global_step = 0
    n_updates = 0

    total_updates = config["total_timesteps"] // config["n_steps"]
    print(f"正在 {config['env_id']} 上訓練 PPO：共 {total_updates} 次更新，每次收集 {config['n_steps']} 步")

    for update in range(1, total_updates + 1):
        ep_returns_this_rollout = []

        # 收集 n 個步數的資料 (Collect n_steps)
        for step in range(config["n_steps"]):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward_done(reward, done)
            obs = next_obs
            ep_return += reward
            ep_length += 1
            global_step += 1

            if done:
                ep_returns_this_rollout.append(ep_return)
                logger.log_episode(ep_return, ep_length, global_step)
                obs, _ = env.reset()
                ep_return = ep_length = 0

        # 執行更新 (Update)
        metrics = agent.update(next_state=obs, last_done=done)
        n_updates += 1

        if metrics and update % config["log_freq_updates"] == 0:
            logger.log_scalars(metrics, global_step)
            if ep_returns_this_rollout:
                mean_ret = np.mean(ep_returns_this_rollout)
                print(
                    f"更新次數 {update:4d}/{total_updates} | "
                    f"總步數 {global_step:8d} | "
                    f"平均集數回報: {mean_ret:.1f} | "
                    f"近似 KL 散度: {metrics.get('approx_kl', 0):.4f}"
                )

        if update % config["eval_freq_updates"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, global_step)
            print(f"  >>> 評估結果 (Eval): {mean_r:.1f} ± {std_r:.1f}")
            if mean_r > best_return:
                best_return = mean_r
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳：{mean_r:.1f}，已儲存")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "CartPole-v1",
        "total_timesteps": 500_000,
        "lr": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "n_epochs": 10,
        "n_minibatch": 32,
        "clip_eps": 0.2,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "log_freq_updates": 10,
        "eval_freq_updates": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
