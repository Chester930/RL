"""
在離線強化學習 (Offline RL) 資料集（D4RL 格式）上訓練 CQL。

離線強化學習：訓練期間不與環境進行任何互動。
資料集由某種行為策略 (Behavior policy) 預先收集而成。

參考文獻：
    Kumar, A., et al. (2020). Conservative Q-Learning for Offline RL. NeurIPS 2020.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import CQLAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def generate_random_dataset(env, n_transitions: int = 100_000) -> dict:
    """
    生成一個簡單的隨機策略資料集用於測試。

    在實踐中，應使用 D4RL：pip install d4rl
    然後執行：dataset = env.get_dataset()
    """
    obs_list, act_list, rew_list, nobs_list, done_list = [], [], [], [], []

    obs, _ = env.reset()
    for _ in range(n_transitions):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_list.append(obs.copy())
        act_list.append(action.copy())
        rew_list.append(reward)
        nobs_list.append(next_obs.copy())
        done_list.append(float(done))

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return {
        "observations": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.float32),
        "rewards": np.array(rew_list, dtype=np.float32),
        "next_observations": np.array(nobs_list, dtype=np.float32),
        "terminals": np.array(done_list, dtype=np.float32),
    }


def train(config: dict) -> CQLAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = CQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        cql_alpha=config["cql_alpha"],
        cql_n_actions=config["cql_n_actions"],
        cql_lagrange=config["cql_lagrange"],
        device=config["device"],
    )

    # 載入離線資料集 (Load offline dataset)
    print("正在載入離線資料集...")
    try:
        # D4RL 資料集
        dataset = env.get_dataset()
    except AttributeError:
        print("找不到 D4RL。正在生成隨機資料集作為演示...")
        dataset = generate_random_dataset(env, n_transitions=config["dataset_size"])

    agent.load_dataset(dataset)

    best_return = -float("inf")

    logger = Logger(log_dir="runs", run_name=f"cql_{config['env_id']}")

    print(f"正在 {config['env_id']} 環境上進行 CQL 離線訓練，總步數為 {config['total_steps']} 步...")

    for step in range(1, config["total_steps"] + 1):
        metrics = agent.update(batch_size=config["batch_size"])

        if step % config["log_freq"] == 0 and metrics:
            logger.log_scalars(metrics, step)
            print(f"步數 {step:6d}  "
                  f"評論家: {metrics['critic_loss']:.4f}  "
                  f"CQL 懲罰項: {metrics['cql_penalty']:.4f}  "
                  f"演員: {metrics['actor_loss']:.4f}")

        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=10)
            logger.log_scalar("eval/mean_return", mean_r, step)
            print(f"  --> 評估回報: {mean_r:.1f} ± {std_r:.1f}")
            if mean_r > best_return:
                best_return = mean_r
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳：{mean_r:.1f}，已儲存")

        if step % config["save_freq"] == 0:
            agent.save(f"checkpoints/cql_step{step}")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "HalfCheetah-v4",
        # 對於 D4RL，請使用: "env_id": "halfcheetah-medium-v2"
        "total_steps": 200_000,
        "dataset_size": 100_000,   # 用於隨機演示資料集 (Random demo dataset)
        "hidden_dim": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "cql_alpha": 1.0,
        "cql_n_actions": 10,
        "cql_lagrange": False,
        "log_freq": 5_000,
        "eval_freq": 25_000,
        "save_freq": 100_000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
