"""
Generic Multi-Layer Perceptron (MLP) building block.

Usage:
    net = MLP(input_dim=4, hidden_dims=[256, 256], output_dim=2)
    out = net(torch.randn(32, 4))   # shape: (32, 2)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Type


class MLP(nn.Module):
    """
    Fully-connected feedforward network with configurable depth and activation.

    Args:
        input_dim:    Dimensionality of the input vector.
        hidden_dims:  List of hidden layer widths (e.g. [256, 256]).
        output_dim:   Dimensionality of the output vector.
        activation:   Activation class applied after every hidden layer (default: ReLU).
        output_activation: Optional activation applied to the final layer (default: None).
        use_layer_norm: If True, LayerNorm is applied after each activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization (common in RL)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration (NoisyNet / Rainbow).

    Reference: Fortunato et al., "Noisy Networks for Exploration", ICLR 2018.
    Replaces epsilon-greedy with learned stochastic weights.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnable)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Factorised Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def sample_noise(self) -> None:
        """Resample noise tensors (call once per forward in training)."""
        p = self._scale_noise(self.in_features)
        q = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(q.outer(p))
        self.bias_epsilon.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)
