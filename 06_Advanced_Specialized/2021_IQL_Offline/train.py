"""
在離線強化學習 (Offline RL) 資料集上訓練 IQL。

IQL 從不查詢分佈外 (OOD) 動作，這使得離線訓練非常安全，
且不會在 Q 網路更新中產生分佈偏移 (Distributional shift) 問題。

參考文獻：
    Kostrikov, I., Nair, A., & Levine, S. (2021).
    Offline RL with Implicit Q-Learning. ICLR 2022.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import IQLAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def generate_random_dataset(env, n_transitions: int = 100_000) -> dict:
    """生成隨機策略資料集用於測試。實踐中請使用 D4RL。"""
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


def train(config: dict) -> IQLAgent:
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

    agent = IQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        expectile=config["expectile"],
        temperature=config["temperature"],
        clip_score=config["clip_score"],
        device=config["device"],
    )

    print("正在載入離線資料集...")
    try:
        dataset = env.get_dataset()
    except AttributeError:
        print("找不到 D4RL。正在使用隨機資料集進行演示...")
        dataset = generate_random_dataset(env, config["dataset_size"])

    agent.load_dataset(dataset)

    best_return = -float("inf")

    logger = Logger(log_dir="runs", run_name=f"iql_{config['env_id']}")
    print(f"正在 {config['env_id']} 環境上進行 IQL 訓練，總步數為 {config['total_steps']} 步...")

    for step in range(1, config["total_steps"] + 1):
        metrics = agent.update(batch_size=config["batch_size"])
        if metrics and np.isnan(metrics.get("qf_loss", 0)):
            raise RuntimeError(f"NaN loss detected at step {step}, stopping training.")

        if step % config["log_freq"] == 0 and metrics:
            logger.log_scalars(metrics, step)
            print(f"步數 {step:6d}  "
                  f"V 損失: {metrics['vf_loss']:.4f}  "
                  f"Q 損失: {metrics['qf_loss']:.4f}  "
                  f"演員損失: {metrics['actor_loss']:.4f}  "
                  f"平均優勢: {metrics['mean_adv']:.3f}")

        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env, n_episodes=10)
            logger.log_scalar("eval/mean_return", mean_r, step)
            print(f"  --> 評估回報: {mean_r:.1f} ± {std_r:.1f}")
            if step > 10_000 and mean_r < best_return * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_return:.1f}")
            if mean_r > best_return:
                best_return = mean_r
                agent.save("checkpoints/best")
                print(f"  ★ 新最佳：{mean_r:.1f}，已儲存")

        if step % config["save_freq"] == 0:
            agent.save(f"checkpoints/iql_step{step}")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "HalfCheetah-v4",
        "total_steps": 200_000,
        "dataset_size": 100_000,
        "hidden_dim": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "expectile": 0.7,     # 論文中的 tau — 越高代表越接近最大值
        "temperature": 3.0,   # 論文中的 beta — 越高代表策略提取越積極
        "clip_score": 100.0,  # 截斷 exp(beta*A) 以確保穩定性
        "log_freq": 5_000,
        "eval_freq": 25_000,
        "save_freq": 100_000,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
