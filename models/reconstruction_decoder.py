"""Identical decoder probe for frozen ViT-AE and DINOv2 tokens."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class DecoderResidualBlock(nn.Module):
    def __init__(self, channels: int, norm_groups: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return residual + hidden


class MatchedTokenDecoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 14,
        out_channels: int = 3,
        hidden_channels: int = 256,
        latent_dim: int = 384,
        num_res_blocks: int = 2,
        norm_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size**2
        self.latent_dim = latent_dim
        self.from_latent = nn.Conv2d(latent_dim, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *[
                DecoderResidualBlock(hidden_channels, norm_groups, dropout)
                for _ in range(num_res_blocks)
            ]
        )
        self.to_pixels = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels, out_channels, patch_size, stride=patch_size
            ),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.num_tokens, self.latent_dim)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"Expected (B, {expected}), got {tokens.shape}")
        spatial = rearrange(
            tokens, "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size
        )
        return self.to_pixels(self.blocks(self.from_latent(spatial)))
