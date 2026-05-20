"""
Nature DQN Convolutional Neural Network for Atari environments.

Architecture from:
    Mnih et al., "Human-level control through deep reinforcement learning",
    Nature 518, 529-533 (2015).

Input:  (batch, 4, 84, 84)  — 4 stacked grayscale frames
Output: (batch, 512)         — flat feature vector, fed into a linear head
"""

import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """
    Three-layer CNN matching the Nature DQN paper architecture.

    Layer 1: Conv 8x8, stride 4, 32 filters
    Layer 2: Conv 4x4, stride 2, 64 filters
    Layer 3: Conv 3x3, stride 1, 64 filters
    FC:      512 units

    Args:
        n_input_channels: Number of stacked frames (default: 4).
        features_dim:     Size of the output feature vector (default: 512).
    """

    def __init__(self, n_input_channels: int = 4, features_dim: int = 512):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, 84, 84)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

        self.features_dim = features_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (batch, C, 84, 84), values in [0, 1].

        Returns:
            features: (batch, features_dim)
        """
        return self.linear(self.cnn(x))


class ImpalaCNN(nn.Module):
    """
    Larger residual CNN from IMPALA / MuZero.

    Uses residual blocks for improved stability on harder Atari games.
    Channel sizes follow the standard IMPALA config: [16, 32, 32].
    Output: (batch, 256) feature vector.
    """

    def __init__(self, n_input_channels: int = 4, features_dim: int = 256):
        super().__init__()

        self.res_blocks = nn.Sequential(
            self._make_block(n_input_channels, 16),
            self._make_block(16, 32),
            self._make_block(32, 32),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, 84, 84)
            flat_dim = self.res_blocks(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(flat_dim, features_dim),
            nn.ReLU(),
        )
        self.features_dim = features_dim

    @staticmethod
    def _make_block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            _ResidualBlock(out_ch),
            _ResidualBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.res_blocks(x))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
