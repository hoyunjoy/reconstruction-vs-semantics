"""Streaming channel statistics and reusable latent normalization."""

from __future__ import annotations

import torch


class StreamingChannelMoments:
    """Numerically stable population moments over every non-channel axis."""

    def __init__(self, channel_dim: int):
        if channel_dim < 1:
            raise ValueError("channel_dim must be positive")
        self.channel_dim = int(channel_dim)
        self.count = 0
        self.mean = torch.zeros(channel_dim, dtype=torch.float64)
        self.m2 = torch.zeros(channel_dim, dtype=torch.float64)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.ndim < 2 or values.shape[-1] != self.channel_dim:
            raise ValueError(
                f"Expected (..., {self.channel_dim}) values, got {tuple(values.shape)}"
            )
        if not torch.isfinite(values).all():
            raise FloatingPointError("Cannot accumulate NaN or Inf values")
        reduction_dims = tuple(range(values.ndim - 1))
        batch_count = values.numel() // self.channel_dim
        work = values.float()
        batch_mean = work.mean(dim=reduction_dims).double().cpu()
        batch_variance = work.var(
            dim=reduction_dims, correction=0
        ).double().cpu()
        batch_m2 = batch_variance * batch_count

        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return

        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean.add_(delta * (batch_count / combined_count))
        self.m2.add_(
            batch_m2
            + delta.square() * (self.count * batch_count / combined_count)
        )
        self.count = combined_count

    def finalize(self, minimum_std: float) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.count == 0:
            raise RuntimeError("No values were accumulated")
        if minimum_std <= 0:
            raise ValueError("minimum_std must be positive")
        variance = self.m2 / self.count
        std = variance.clamp_min(0).sqrt().clamp_min(minimum_std)
        return self.mean.float(), std.float(), self.count


class LatentNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("mean and std must be same-shaped 1D tensors")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise FloatingPointError("Normalization statistics contain NaN or Inf")
        if not torch.all(std > 0):
            raise ValueError("Every standard deviation must be positive")
        self.mean = mean
        self.std = std

    @classmethod
    def from_file(
        cls, path: str, *, device: torch.device | str = "cpu"
    ) -> "LatentNormalizer":
        payload = torch.load(path, map_location=device, weights_only=True)
        return cls(payload["mean"], payload["std"])

    def to(self, device: torch.device | str) -> "LatentNormalizer":
        return LatentNormalizer(self.mean.to(device), self.std.to(device))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean
