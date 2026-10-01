"""Frozen-representation state probes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialLinearStateProbe(nn.Module):
    """A fixed spatial pyramid followed by one trainable linear readout."""

    def __init__(
        self,
        *,
        num_tokens: int = 256,
        latent_dim: int = 384,
        target_dim: int = 8,
        grid_size: int = 16,
        pool_sizes: tuple[int, ...] | list[int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.target_dim = int(target_dim)
        self.grid_size = int(grid_size)
        self.pool_sizes = tuple(int(size) for size in pool_sizes)
        if self.grid_size * self.grid_size != self.num_tokens:
            raise ValueError("grid_size squared must equal num_tokens")
        if not self.pool_sizes or any(size < 1 for size in self.pool_sizes):
            raise ValueError("pool_sizes must contain positive integers")
        pooled_cells = sum(size * size for size in self.pool_sizes)
        self.readout = nn.Linear(pooled_cells * latent_dim, target_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        expected = (self.num_tokens, self.latent_dim)
        if latents.ndim != 3 or tuple(latents.shape[1:]) != expected:
            raise ValueError(
                f"Expected latent shape (B, {expected}), got {latents.shape}"
            )
        feature_map = latents.transpose(1, 2).reshape(
            latents.shape[0], self.latent_dim, self.grid_size, self.grid_size
        )
        pyramid = [
            F.adaptive_avg_pool2d(feature_map, size).flatten(1)
            for size in self.pool_sizes
        ]
        return self.readout(torch.cat(pyramid, dim=1))
