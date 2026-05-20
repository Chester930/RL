"""在 CartPole-v1 上訓練 A2C。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
# pyrefly: ignore [missing-import]
import gymnasium as gym
from agent import A2CAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> A2CAgent:
    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = A2CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        gae_lambda=config["gae_lambda"],
        c_v=config["c_v"],
        c_e=config["c_e"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"a2c_{config['env_id']}")
    obs, _ = env.reset()
    ep_return = ep_length = 0
    global_step = 0

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        ep_return = ep_length = 0
        done = False

        while not done:
            for _ in range(config["n_steps"]):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_reward_done(reward, done)
                obs = next_obs
                ep_return += reward
                ep_length += 1
                global_step += 1
                if done:
                    break

            metrics = agent.update(next_state=None if done else obs, last_done=done)
            if metrics and global_step % config["log_freq"] == 0:
                logger.log_scalars(metrics, global_step)

        logger.log_episode(ep_return, ep_length, global_step)

        if episode % config["eval_freq_ep"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, global_step)
            print(f"Episode {episode:5d} | Step {global_step:8d} | Eval: {mean_r:.1f} ± {std_r:.1f}")

    logger.close()
    env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "CartPole-v1",
        "n_episodes": 2000,
        "lr": 7e-4,
        "gamma": 0.99,
        "n_steps": 5,
        "gae_lambda": 0.95,
        "c_v": 0.5,
        "c_e": 0.01,
        "log_freq": 500,
        "eval_freq_ep": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    train(config)
