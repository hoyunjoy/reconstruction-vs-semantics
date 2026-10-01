"""Shared action-conditioned dynamics for cached ViT-AE and DINOv2 tokens."""

from __future__ import annotations

import torch
from torch import nn


class FactorizedActionDynamics(nn.Module):
    """Temporal token mixing followed by spatial Transformer interaction."""

    def __init__(
        self,
        *,
        num_tokens: int = 256,
        latent_dim: int = 384,
        action_dim: int = 2,
        context_length: int = 3,
        temporal_layers: int = 1,
        spatial_depth: int = 4,
        spatial_heads: int = 6,
        spatial_mlp_dim: int = 1536,
        action_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if latent_dim % spatial_heads != 0:
            raise ValueError("latent_dim must be divisible by spatial_heads")
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.context_length = context_length

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, latent_dim),
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(1, context_length, 1, latent_dim)
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(1, num_tokens, latent_dim)
        )
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)

        self.temporal_model = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=spatial_heads,
            dim_feedforward=spatial_mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_model = nn.TransformerEncoder(
            encoder_layer,
            num_layers=spatial_depth,
            enable_nested_tensor=False,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self, latent_history: torch.Tensor, action_history: torch.Tensor
    ) -> torch.Tensor:
        expected_latent = (
            self.context_length,
            self.num_tokens,
            self.latent_dim,
        )
        expected_action = (self.context_length, self.action_dim)
        if latent_history.ndim != 4 or tuple(latent_history.shape[1:]) != expected_latent:
            raise ValueError(
                f"Expected latent shape (B, {expected_latent}), got "
                f"{tuple(latent_history.shape)}"
            )
        if action_history.ndim != 3 or tuple(action_history.shape[1:]) != expected_action:
            raise ValueError(
                f"Expected action shape (B, {expected_action}), got "
                f"{tuple(action_history.shape)}"
            )

        action_features = self.action_encoder(action_history).unsqueeze(2)
        conditioned = latent_history + action_features + self.temporal_position
        batch_size = conditioned.shape[0]
        temporal_input = conditioned.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_tokens,
            self.context_length,
            self.latent_dim,
        )
        temporal_output, _ = self.temporal_model(temporal_input)
        fused = temporal_output[:, -1].reshape(
            batch_size, self.num_tokens, self.latent_dim
        )
        spatial_output = self.spatial_model(fused + self.spatial_position)
        delta = self.delta_head(spatial_output)
        return latent_history[:, -1] + delta

    def rollout(
        self,
        initial_latents: torch.Tensor,
        action_sequence: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        expected_actions = self.context_length + horizon - 1
        if action_sequence.shape[1] < expected_actions:
            raise ValueError(
                f"Need at least {expected_actions} actions for horizon {horizon}"
            )
        history = initial_latents
        predictions = []
        for step in range(horizon):
            action_history = action_sequence[
                :, step : step + self.context_length
            ]
            prediction = self(history, action_history)
            predictions.append(prediction)
            history = torch.cat((history[:, 1:], prediction.unsqueeze(1)), dim=1)
        return torch.stack(predictions, dim=1)
