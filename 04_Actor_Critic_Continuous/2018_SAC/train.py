"""在 Pendulum-v1 或 HalfCheetah-v4 上訓練 SAC。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym
from agent import SACAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> SACAgent:
    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, action_scale=action_scale,
        lr=config["lr"], gamma=config["gamma"], tau=config["tau"],
        buffer_size=config["buffer_size"], batch_size=config["batch_size"],
        auto_alpha=config["auto_alpha"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"sac_{config['env_id']}")
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
            logger.log_episode(ep_return, ep_length, step)
            obs, _ = env.reset()
            ep_return = ep_length = 0

        if step >= config["learning_starts"]:
            metrics = agent.update()
            if metrics and step % config["log_freq"] == 0:
                logger.log_scalars(metrics, step)

        if step % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, step)
            print(f"Step {step:8d}  Eval: {mean_r:.1f} ± {std_r:.1f}  Alpha: {agent.alpha:.4f}")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "Pendulum-v1",
        "total_steps": 200_000,
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "buffer_size": 200_000,
        "batch_size": 256,
        "auto_alpha": True,
        "learning_starts": 5000,
        "log_freq": 1000,
        "eval_freq": 10_000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    train(config)
