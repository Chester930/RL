"""
Prioritized Experience Replay (PER) buffer.

Implements the proportional variant from:
    Schaul et al., "Prioritized Experience Replay", ICLR 2016.
    https://arxiv.org/abs/1511.05952

A SumTree segment tree allows O(log N) sampling and O(log N) priority updates.
"""

import random
import numpy as np


class SumTree:
    """
    Binary segment tree where each leaf stores a priority p_i,
    and each internal node stores the sum of its children.

    - Update priority of leaf i:  O(log N)
    - Sample proportional to priority: O(log N)
    - Query total priority: O(1)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity          # number of leaves
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data = [None] * capacity     # stores actual transitions
        self.data_pointer = 0             # circular write pointer
        self.n_entries = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, leaf_idx: int, delta: float) -> None:
        """Propagate priority change up to the root."""
        parent = (leaf_idx) // 2
        while parent >= 1:
            self.tree[parent] += delta
            parent //= 2

    def _retrieve(self, node: int, value: float) -> int:
        """Walk down the tree to find the leaf whose prefix sum >= value."""
        left = 2 * node
        right = left + 1
        if left >= len(self.tree):
            return node
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total(self) -> float:
        """Sum of all priorities (root of the tree)."""
        return float(self.tree[1])

    def add(self, priority: float, data) -> None:
        """Insert a transition with the given priority."""
        leaf_idx = self.data_pointer + self.capacity  # 1-indexed tree
        self.update(leaf_idx, priority)
        self.data[self.data_pointer] = data
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update the priority of an existing leaf."""
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, value: float):
        """
        Sample a leaf whose cumulative priority covers `value`.

        Returns:
            (leaf_idx, priority, data)
        """
        leaf_idx = self._retrieve(1, value)
        data_idx = leaf_idx - self.capacity
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer backed by a SumTree.

    Transitions are sampled with probability proportional to |TD error|^alpha.
    Importance-sampling weights are returned to correct for the sampling bias.

    Args:
        capacity: Maximum number of stored transitions.
        alpha:    Priority exponent (0 = uniform, 1 = full prioritization).
        beta:     IS weight exponent (annealed from beta_start -> 1.0).
        epsilon:  Small constant to avoid zero priorities.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self._beta_start = beta  # fixed reference for linear annealing
        self.epsilon = epsilon
        self._max_priority = 1.0
        self.tree = SumTree(capacity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition with maximum current priority (ensures it gets sampled)."""
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict:
        """
        Sample a prioritized mini-batch.

        Returns:
            dict with keys: states, actions, rewards, next_states, dones,
                            weights (IS correction), indices (for priority update).
        """
        assert len(self) >= batch_size, (
            f"Buffer has {len(self)} transitions, need {batch_size}."
        )

        batch = []
        indices = []
        priorities = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            leaf_idx, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(leaf_idx)
            priorities.append(priority)

        # Importance-sampling weights
        sampling_probs = np.array(priorities, dtype=np.float64) / self.tree.total
        # Clip to avoid division by zero
        sampling_probs = np.clip(sampling_probs, 1e-10, 1.0)
        weights = (len(self) * sampling_probs) ** (-self.beta)
        weights /= weights.max()   # normalize so max weight = 1

        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "weights": np.array(weights, dtype=np.float32),
            "indices": indices,
        }

    def update_priorities(self, indices: list, td_errors: np.ndarray) -> None:
        """
        Update priorities after computing new TD errors.

        Args:
            indices:   Leaf indices returned by sample().
            td_errors: Absolute TD errors, shape (batch_size,).
        """
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)

    def anneal_beta(self, step: int, total_steps: int, beta_end: float = 1.0) -> None:
        """Linearly anneal beta from its initial value toward beta_end."""
        self.beta = min(beta_end, self._beta_start + (beta_end - self._beta_start) * step / total_steps)

    def __len__(self) -> int:
        return self.tree.n_entries
