"""
Policy evaluation utility.

Usage:
    mean_return, std_return = evaluate(agent, env, n_episodes=10)
"""

import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym
from typing import Tuple


def evaluate(
    agent,
    env: gym.Env,
    n_episodes: int = 10,
    max_steps: int = 10_000,
    render: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate an agent's greedy policy over several episodes.

    The agent's `select_action(state, evaluate=True)` method is called at
    every step. No gradient computation is performed.

    Args:
        agent:       Any agent implementing select_action(state, evaluate=True).
        env:         A gymnasium environment (not modified).
        n_episodes:  Number of evaluation episodes to run.
        max_steps:   Maximum steps per episode (safety cap).
        render:      If True, call env.render() at each step.

    Returns:
        (mean_return, std_return): Statistics over the n_episodes.
    """
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        step = 0

        while True:
            if render:
                env.render()

            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            step += 1

            if terminated or truncated or step >= max_steps:
                break

        returns.append(ep_return)

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    return mean_return, std_return


def evaluate_and_log(
    agent,
    env: gym.Env,
    logger,
    step: int,
    n_episodes: int = 10,
    max_steps: int = 10_000,
    tag: str = "eval/mean_return",
) -> Tuple[float, float]:
    """
    Evaluate and immediately log results to a Logger.

    Args:
        agent:      Agent to evaluate.
        env:        Evaluation environment.
        logger:     Logger instance with log_scalar() method.
        step:       Current training step (used as x-axis in TensorBoard).
        n_episodes: Number of episodes.
        max_steps:  Max steps per episode.
        tag:        TensorBoard tag for mean return.

    Returns:
        (mean_return, std_return)
    """
    mean_r, std_r = evaluate(agent, env, n_episodes, max_steps)
    logger.log_scalar(tag, mean_r, step=step)
    logger.log_scalar(tag.replace("mean", "std"), std_r, step=step)
    print(
        f"[Eval @ step {step:8d}] "
        f"Mean Return: {mean_r:.2f} ± {std_r:.2f}  ({n_episodes} eps)"
    )
    return mean_r, std_r
