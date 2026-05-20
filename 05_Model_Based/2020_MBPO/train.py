"""
MBPO 訓練迴圈。

真實環境步數與模型取樣 (Rollouts) 及 SAC 更新交替執行。

參考文獻：
    Janner et al. (2019). When to Trust Your Model: Model-Based Policy Optimization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import MBPOAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> MBPOAgent:
    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = MBPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        ensemble_members=config["ensemble_members"],
        n_elite=config["n_elite"],
        rollout_length=config["rollout_length"],
        real_ratio=config["real_ratio"],
        gamma=config["gamma"],
        tau=config["tau"],
        lr=config["lr"],
        model_lr=config["model_lr"],
        real_buffer_size=config["real_buffer_size"],
        model_buffer_size=config["model_buffer_size"],
        batch_size=config["batch_size"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"mbpo_{config['env_id']}")

    obs, _ = env.reset()
    ep_return = 0.0
    ep_length = 0

    print(f"正在 {config['env_id']} 環境上進行 MBPO 訓練，總步數為 {config['total_steps']} 步...")

    for step in range(1, config["total_steps"] + 1):
        # 收集真實轉換資料 (Collect real transition)
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.real_buffer.push(obs, action, reward, next_obs, float(done))
        obs = next_obs
        ep_return += reward
        ep_length += 1

        if done:
            logger.log_episode(ep_return, ep_length, step)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_length = 0

        if step < config["random_steps"]:
            continue

        # 定期重新訓練動態模型 (Retrain dynamics model)
        if step % config["model_train_freq"] == 0:
            model_metrics = agent.update_model(n_epochs=config["model_epochs"])
            if model_metrics:
                logger.log_scalars(model_metrics, step)

            # 生成模型取樣 (Generate model rollouts)
            agent.model_rollout(n_rollouts=config["rollout_batch_size"])

        # SAC 梯度更新步數 (SAC gradient steps)
        for _ in range(config["sac_updates_per_step"]):
            metrics = agent.update()
            if metrics and step % config["log_freq"] == 0:
                logger.log_scalars(metrics, step)

        # 評估 (Evaluation)
        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=5)
            logger.log_scalar("eval/mean_return", mean_r, step)
            print(f"步數 {step:8d}  評估回報: {mean_r:.1f} ± {std_r:.1f}")

        # 檢查點 (Checkpoint)
        if step % config["save_freq"] == 0:
            agent.save(f"checkpoints/mbpo_step{step}")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "Hopper-v4",
        "total_steps": 300_000,
        "random_steps": 5_000,
        "hidden_dim": 256,
        "ensemble_members": 7,
        "n_elite": 5,
        "rollout_length": 1,        # 隨訓練過程增加（未顯示排程邏輯）
        "rollout_batch_size": 400,  # 每次模型訓練時的分支狀態數量
        "real_ratio": 0.05,
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 3e-4,
        "model_lr": 1e-3,
        "real_buffer_size": 100_000,
        "model_buffer_size": 400_000,
        "batch_size": 256,
        "model_train_freq": 250,
        "model_epochs": 5,
        "sac_updates_per_step": 20,
        "log_freq": 1000,
        "eval_freq": 10_000,
        "save_freq": 50_000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }
    train(config)
