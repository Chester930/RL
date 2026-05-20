"""
Standard experience replay buffer using a circular deque.

Usage:
    buf = ReplayBuffer(capacity=100_000)
    buf.push(state, action, reward, next_state, done)
    batch = buf.sample(batch_size=256)
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Fixed-size FIFO replay buffer.

    Stores transitions (s, a, r, s', done) and returns random mini-batches
    as numpy arrays ready for conversion to tensors.
    """

    def __init__(self, capacity: int = 100_000):
        """
        Args:
            capacity: Maximum number of transitions to store.
                      Oldest transitions are discarded when full.
        """
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer."""
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict:
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            dict with keys: states, actions, rewards, next_states, dones
            All values are numpy arrays of appropriate dtype.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Not enough transitions to sample: "
                f"buffer has {len(self._buffer)}, requested {batch_size}."
            )

        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # float32 works for both continuous (SAC/TD3/DDPG) and discrete (DQN)
        # actions. Discrete agents call torch.LongTensor() which handles the
        # float→long conversion correctly.
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Return True when buffer contains enough data to sample."""
        return len(self._buffer) >= batch_size

    def clear(self) -> None:
        """Remove all transitions from the buffer."""
        self._buffer.clear()
