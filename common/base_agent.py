"""
Abstract base class for all RL agents in this project.
Every algorithm-specific agent should inherit from BaseAgent.
"""

from abc import ABC, abstractmethod
import os
import torch


class BaseAgent(ABC):
    """
    Abstract base class defining the interface for all RL agents.

    Subclasses must implement:
        - select_action(state) -> action
        - update(*args, **kwargs) -> dict of losses/metrics
        - save(path)
        - load(path)
    """

    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.total_steps = 0       # incremented by the training loop
        self.episodes_done = 0     # incremented by the training loop

    @abstractmethod
    def select_action(self, state, evaluate: bool = False):
        """
        Choose an action given the current state.

        Args:
            state: Current environment observation (numpy array or tensor).
            evaluate: If True, use greedy/deterministic policy (no exploration).

        Returns:
            action: The selected action (scalar int or numpy array).
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """
        Perform one gradient update step.

        Returns:
            metrics (dict): A dictionary of scalar metrics (e.g., {"loss": 0.1}).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist the agent's parameters to disk.

        Args:
            path: Directory path where checkpoint files will be written.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Restore the agent's parameters from disk.

        Args:
            path: Directory path containing the checkpoint files.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience helpers (optional to override)
    # ------------------------------------------------------------------

    def to(self, device: str):
        """Move agent networks to a different device."""
        self.device = torch.device(device)
        return self

    def train_mode(self):
        """Set all networks to training mode."""
        for attr in self.__dict__.values():
            if isinstance(attr, torch.nn.Module):
                attr.train()

    def eval_mode(self):
        """Set all networks to evaluation mode."""
        for attr in self.__dict__.values():
            if isinstance(attr, torch.nn.Module):
                attr.eval()

    def _ensure_dir(self, path: str) -> None:
        """Create directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"device={self.device})"
        )
