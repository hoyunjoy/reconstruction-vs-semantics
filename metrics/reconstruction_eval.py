"""Reconstruction metrics for images represented in the [-1, 1] range."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


def to_zero_one(images: torch.Tensor) -> torch.Tensor:
    return images.add(1).div(2).clamp(0, 1)


@torch.no_grad()
def batch_psnr(
    targets_minus_one_to_one: torch.Tensor,
    reconstructions_minus_one_to_one: torch.Tensor,
) -> torch.Tensor:
    targets = to_zero_one(targets_minus_one_to_one)
    reconstructions = to_zero_one(reconstructions_minus_one_to_one)
    mse = (targets - reconstructions).square().flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


@torch.no_grad()
def batch_ssim(
    targets_minus_one_to_one: torch.Tensor,
    reconstructions_minus_one_to_one: torch.Tensor,
) -> list[float]:
    targets = to_zero_one(targets_minus_one_to_one).permute(0, 2, 3, 1).cpu().numpy()
    reconstructions = (
        to_zero_one(reconstructions_minus_one_to_one)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [
        float(
            structural_similarity(
                target,
                reconstruction,
                channel_axis=-1,
                data_range=1.0,
            )
        )
        for target, reconstruction in zip(targets, reconstructions)
    ]


class ReconstructionAccumulator:
    def __init__(self, compute_ssim: bool = False):
        self.compute_ssim = compute_ssim
        self.pixel_squared_error = 0.0
        self.pixel_count = 0
        self.psnr_sum = 0.0
        self.image_count = 0
        self.ssim_values: list[float] = []

    @torch.no_grad()
    def update(self, targets: torch.Tensor, reconstructions: torch.Tensor) -> None:
        if targets.shape != reconstructions.shape:
            raise ValueError("Reconstruction and target shapes differ")
        if not torch.isfinite(reconstructions).all():
            raise FloatingPointError("Reconstruction contains NaN or Inf")
        difference = reconstructions - targets
        self.pixel_squared_error += float(difference.square().sum().item())
        self.pixel_count += difference.numel()
        psnr_values = batch_psnr(targets, reconstructions)
        self.psnr_sum += float(psnr_values.sum().item())
        self.image_count += int(targets.shape[0])
        if self.compute_ssim:
            self.ssim_values.extend(batch_ssim(targets, reconstructions))

    def compute(self) -> dict[str, float | int]:
        if self.image_count == 0 or self.pixel_count == 0:
            raise RuntimeError("No reconstruction samples accumulated")
        result: dict[str, float | int] = {
            "mse_minus_one_to_one": self.pixel_squared_error / self.pixel_count,
            "psnr_db": self.psnr_sum / self.image_count,
            "image_count": self.image_count,
        }
        if self.compute_ssim:
            result["ssim"] = float(np.mean(self.ssim_values))
        return result
