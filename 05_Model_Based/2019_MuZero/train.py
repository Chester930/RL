"""在 CartPole-v1 環境上訓練 MuZero。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import random
import torch
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym

from agent import MuZeroAgent
from common.utils.logger import Logger
from common.utils.evaluator import evaluate


def train(config: dict) -> MuZeroAgent:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])

    agent = MuZeroAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        num_simulations=config["num_simulations"],
        gamma=config["gamma"],
        device=config["device"],
    )

    logger = Logger(log_dir="runs", run_name=f"muzero_{config['env_id']}")
    best_eval = -float("inf")
    os.makedirs("best_checkpoints", exist_ok=True)

    print(f"MuZero 重跑：num_simulations={config['num_simulations']}, n_episodes={config['n_episodes']}")

    for episode in range(1, config["n_episodes"] + 1):
        obs, _ = env.reset()
        game_history = []
        ep_return = ep_length = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            game_history.append({
                "obs": obs, "action": action,
                "reward": reward, "done": done,
            })
            ep_return += reward
            ep_length += 1
            obs = next_obs

        agent.store(game_history)
        metrics = agent.update()
        if metrics and np.isnan(metrics.get("value_loss", 0)):
            raise RuntimeError(f"NaN loss detected at episode {episode}, stopping training.")
        logger.log_episode(ep_return, ep_length, episode)

        if episode % config["eval_freq"] == 0:
            mean_r, std_r = evaluate(agent, eval_env)
            logger.log_scalar("eval/mean_return", mean_r, episode)
            print(f"Episode {episode:5d} | Eval: {mean_r:.1f} ± {std_r:.1f} | Buffer: {len(agent.replay_buffer)}")
            if episode > 1000 and mean_r < best_eval * 0.3:
                print(f"  [WARNING] eval 崩潰：{mean_r:.1f} vs 峰值 {best_eval:.1f}")
            if mean_r > best_eval:
                best_eval = mean_r
                agent.save("best_checkpoints")
                print(f"  ** 新最佳 {best_eval:.1f}，已儲存 checkpoint **")

    logger.close()
    env.close()
    eval_env.close()
    print(f"\n訓練完成。最佳 eval: {best_eval:.1f}")
    return agent


if __name__ == "__main__":
    config = {
        "env_id": "CartPole-v1",
        "n_episodes": 3000,
        "hidden_dim": 64,
        "lr": 1e-3,
        "num_simulations": 50,
        "gamma": 0.997,
        "eval_freq": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    train(config)
