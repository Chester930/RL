"""
Gymnasium environment wrappers for Atari preprocessing.

Follows the standard preprocessing pipeline from the Nature DQN paper:
    1. NoopResetEnv       — random no-ops at episode start
    2. MaxAndSkipEnv      — frame skip with max pooling
    3. EpisodicLifeEnv    — treat life loss as episode end
    4. FireResetEnv       — press FIRE on reset for games that require it
    5. WarpFrame          — resize to 84x84 grayscale
    6. ScaledFloatFrame   — divide pixel values by 255
    7. FrameStack         — stack k consecutive frames

Use make_atari_env() to apply the full pipeline in one call.
"""

import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from gymnasium import spaces
from collections import deque
from typing import Optional
import cv2


# ---------------------------------------------------------------------------
# Individual wrappers
# ---------------------------------------------------------------------------

class NoopResetEnv(gym.Wrapper):
    """
    Sample a random number of no-ops at episode start.
    This helps break correlations between episodes.

    Args:
        noop_max: Maximum number of no-ops to execute (uniformly sampled).
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_noops = self.np_random.integers(1, self.noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame, taking the pixel-wise max over
    the last two frames to handle Atari's sprite flickering.

    Args:
        skip: Number of frames to repeat the selected action.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        obs_shape = env.observation_space.shape
        self._obs_buffer = np.zeros((2, *obs_shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Treat every life loss as an episode end for the agent,
    but only truly reset when the game is over.
    This encourages the agent to avoid dying.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # Life lost — signal episode end without resetting
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Step with no-op to advance the game without full reset
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Press FIRE on environment reset for games that require it to start
    (e.g., Breakout, Pong).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """
    Convert observations to 84x84 grayscale images.

    Input:  RGB frame of any resolution
    Output: (84, 84, 1) uint8 grayscale array
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]   # add channel dimension


class ScaledFloatFrame(gym.ObservationWrapper):
    """Divide pixel values by 255 to get float observations in [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.array(obs, dtype=np.float32) / 255.0


class FrameStack(gym.Wrapper):
    """
    Stack the last `k` observations along a new first axis.

    Input:  (H, W, C)
    Output: (k*C, H, W)  — channel-first format expected by NatureCNN
    """

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self._frames: deque = deque(maxlen=k)

        obs_space = env.observation_space
        # Original shape: (H, W, C) -> stacked: (k*C, H, W)
        H, W, C = obs_space.shape
        self.observation_space = spaces.Box(
            low=obs_space.low.min(),
            high=obs_space.high.max(),
            shape=(k * C, H, W),
            dtype=obs_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # frames: list of (H, W, C) -> stack to (k*C, H, W)
        frames = np.array(list(self._frames), dtype=np.float32)  # (k, H, W, C)
        frames = frames.transpose(0, 3, 1, 2)  # (k, C, H, W)
        return frames.reshape(-1, *frames.shape[2:])  # (k*C, H, W)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_atari_env(
    env_id: str,
    render_mode: Optional[str] = None,
    noop_max: int = 30,
    frame_skip: int = 4,
    frame_stack: int = 4,
    episodic_life: bool = True,
    fire_reset: bool = True,
    scale: bool = True,
) -> gym.Env:
    """
    Apply the standard Atari preprocessing pipeline.

    Args:
        env_id:        Gymnasium environment ID (e.g. "ALE/Breakout-v5").
        render_mode:   e.g. "rgb_array" or "human".
        noop_max:      Max random no-ops on reset.
        frame_skip:    Number of frames to repeat each action.
        frame_stack:   Number of consecutive frames to stack.
        episodic_life: Treat life loss as episode end.
        fire_reset:    Auto-press FIRE on reset.
        scale:         Divide pixels by 255 to get float in [0,1].

    Returns:
        Fully preprocessed gymnasium environment.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if fire_reset and "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    env = FrameStack(env, k=frame_stack)
    return env
