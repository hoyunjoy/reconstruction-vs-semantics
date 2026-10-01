"""Trainable ViT-S/14 autoencoder for the controlled PushT comparison.

The encoder is instantiated from the exact DINOv2 ViT-S/14 implementation,
but with ``pretrained=False``.  No DINO weights are loaded.  The decoder is
the same 2,611,543-parameter convolutional decoder used by the matched
reconstruction probe in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 256, norm_groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return residual + x


class ReconstructionDecoder(nn.Module):
    """Convolutional decoder used for the controlled reconstruction model.

    With the defaults below this module has exactly 2,611,543 trainable
    parameters.  It accepts the common 256 x 384 token representation.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        hidden_channels: int = 256,
        grid_size: int = 16,
        patch_size: int = 14,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        norm_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.grid_size = int(grid_size)
        self.input_projection = nn.Conv2d(latent_dim, hidden_channels, kernel_size=1)
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_channels, norm_groups, dropout)
                for _ in range(num_res_blocks)
            ]
        )
        self.upsample = nn.ConvTranspose2d(
            hidden_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.output_projection = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected BNC tokens, received shape {tuple(tokens.shape)}")
        batch, num_tokens, channels = tokens.shape
        expected_tokens = self.grid_size * self.grid_size
        if num_tokens != expected_tokens or channels != self.latent_dim:
            raise ValueError(
                f"Expected (B, {expected_tokens}, {self.latent_dim}), "
                f"received {tuple(tokens.shape)}"
            )
        x = tokens.transpose(1, 2).reshape(
            batch, self.latent_dim, self.grid_size, self.grid_size
        )
        x = self.input_projection(x)
        x = self.residual_blocks(x)
        x = self.upsample(x)
        return torch.tanh(self.output_projection(x))


class DinoV2ScratchEncoder(nn.Module):
    """Exact DINOv2 backbone architecture with randomly initialized weights."""

    def __init__(
        self,
        name: str = "dinov2_vits14",
        image_size: int = 224,
        torch_hub_repo: Optional[str] = None,
    ):
        super().__init__()
        if name != "dinov2_vits14":
            raise ValueError("This controlled experiment requires dinov2_vits14")

        if torch_hub_repo:
            repo = str(Path(torch_hub_repo).expanduser().resolve())
            self.base_model = torch.hub.load(
                repo,
                name,
                source="local",
                pretrained=False,
            )
        else:
            self.base_model = torch.hub.load(
                "facebookresearch/dinov2",
                name,
                pretrained=False,
                trust_repo=True,
            )

        self.name = name
        self.image_size = int(image_size)
        self.patch_size = int(self.base_model.patch_size)
        self.emb_dim = int(self.base_model.num_features)
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size * self.grid_size

        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by the DINO patch size")
        if (self.patch_size, self.emb_dim, self.num_tokens) != (14, 384, 256):
            raise RuntimeError(
                "Unexpected ViT-S/14 geometry: "
                f"patch={self.patch_size}, dim={self.emb_dim}, tokens={self.num_tokens}"
            )

        # torch.hub constructed the backbone from scratch.  Make the intended
        # training state explicit so an accidental freeze is caught in review.
        self.base_model.train()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(True)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"Expected (B, 3, {self.image_size}, {self.image_size}), "
                f"received {tuple(images.shape)}"
            )
        features = self.base_model.forward_features(images)
        tokens = features["x_norm_patchtokens"]
        if tokens.shape[1:] != (self.num_tokens, self.emb_dim):
            raise RuntimeError(f"Unexpected patch-token shape {tuple(tokens.shape)}")
        return tokens


class ViTAutoencoder(nn.Module):
    """ViT-S/14 encoder trained from scratch with a convolutional decoder."""

    def __init__(
        self,
        image_size: int = 224,
        encoder_name: str = "dinov2_vits14",
        torch_hub_repo: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = DinoV2ScratchEncoder(
            name=encoder_name,
            image_size=image_size,
            torch_hub_repo=torch_hub_repo,
        )
        self.decoder = ReconstructionDecoder(
            latent_dim=self.encoder.emb_dim,
            grid_size=self.encoder.grid_size,
            patch_size=self.encoder.patch_size,
        )

    def encode(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def decode(self, tokens: Tensor) -> Tensor:
        return self.decoder(tokens)

    def forward(self, images: Tensor, return_latent: bool = False):
        tokens = self.encode(images)
        reconstruction = self.decode(tokens)
        if return_latent:
            return reconstruction, tokens
        return reconstruction

    def parameter_counts(self) -> dict[str, int]:
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
