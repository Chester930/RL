"""在 Pendulum-v1 或 LunarLanderContinuous-v2 上訓練 DDPG。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import torch
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import DDPGAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> DDPGAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    best_return = -float("inf")

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        gamma=config["gamma"],
        tau=config["tau"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        noise_sigma=config["noise_sigma"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"ddpg_{config['env_id']}")
    obs, _ = env.reset()
    ep_return = ep_length = 0

    for step in range(1, config["total_steps"] + 1):
        if step < config["learning_starts"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_return += reward
        ep_length += 1

        if done:
            agent.noise.reset()
            logger.log_episode(ep_return, ep_length, step)
            obs, _ = env.reset()
            ep_return = ep_length = 0

        if step >= config["learning_starts"]:
            metrics = agent.update()
            if metrics and np.isnan(metrics.get("critic_loss", 0)):
                raise RuntimeError(f"NaN loss detected at step {step}, stopping training.")
            if metrics and step % config["log_freq"] == 0:
                logger.log_scalars(metrics, step)

        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, step)
            print(f"Step {step:8d}  Eval: {mean_r:.1f} ± {std_r:.1f}")
            if step > 10_000 and mean_r < best_return * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_return:.1f}")
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
        "env_id": "Pendulum-v1",
        "total_steps": 200_000,
        "lr_actor": 1e-4,
        "lr_critic": 1e-3,
        "gamma": 0.99,
        "tau": 0.005,
        "buffer_size": 100_000,
        "batch_size": 256,
        "noise_sigma": 0.1,
        "learning_starts": 5000,
        "log_freq": 1000,
        "eval_freq": 10_000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
