"""CEM utilities for normalized-action latent MPC."""

from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def cem_plan(
    *,
    dynamics: torch.nn.Module,
    latent_context: torch.Tensor,
    past_actions: torch.Tensor,
    goal_latent: torch.Tensor,
    horizon: int,
    num_samples: int,
    num_elites: int,
    iterations: int,
    initial_std: float,
    minimum_std: float,
    action_clip: float,
    smoothing: float,
    generator: torch.Generator,
    use_amp: bool,
    initial_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Optimize a future normalized-action sequence by terminal latent MSE."""
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    if not 1 <= num_elites <= num_samples:
        raise ValueError("num_elites must be between 1 and num_samples")
    device = latent_context.device
    action_dim = past_actions.shape[-1]
    mean = (
        torch.zeros(horizon, action_dim, device=device)
        if initial_mean is None
        else initial_mean.to(device).clone()
    )
    if tuple(mean.shape) != (horizon, action_dim):
        raise ValueError("initial_mean has the wrong shape")
    std = torch.full_like(mean, float(initial_std))
    best_cost = float("inf")
    best_action = mean.clone()
    repeated_context = latent_context.expand(num_samples, -1, -1, -1)
    repeated_goal = goal_latent.expand(num_samples, -1, -1)
    prefix = past_actions[-(latent_context.shape[1] - 1) :]
    prefix = prefix.unsqueeze(0).expand(num_samples, -1, -1)

    for _ in range(iterations):
        noise = torch.randn(
            num_samples, horizon, action_dim, device=device, generator=generator
        )
        samples = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
            -action_clip, action_clip
        )
        samples[0] = mean
        action_sequence = torch.cat((prefix, samples), dim=1)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            predictions = dynamics.rollout(
                repeated_context, action_sequence, horizon
            )
            costs = (predictions[:, -1] - repeated_goal).float().square().mean(
                dim=(1, 2)
            )
        elite_costs, elite_indices = torch.topk(
            costs, k=num_elites, largest=False, sorted=True
        )
        elites = samples[elite_indices]
        candidate_mean = elites.mean(dim=0)
        candidate_std = elites.std(dim=0, unbiased=False).clamp_min(minimum_std)
        mean = smoothing * mean + (1.0 - smoothing) * candidate_mean
        std = smoothing * std + (1.0 - smoothing) * candidate_std
        if float(elite_costs[0]) < best_cost:
            best_cost = float(elite_costs[0])
            best_action = samples[elite_indices[0]].clone()
    return best_action, best_cost


def circular_angle_error(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def goal_state_metrics(final_state: np.ndarray, goal_state: np.ndarray) -> dict:
    position_error = float(np.linalg.norm(final_state[:4] - goal_state[:4]))
    angle_error = circular_angle_error(float(final_state[4]), float(goal_state[4]))
    return {
        "position_l2_error": position_error,
        "agent_position_l2_error": float(
            np.linalg.norm(final_state[:2] - goal_state[:2])
        ),
        "block_position_l2_error": float(
            np.linalg.norm(final_state[2:4] - goal_state[2:4])
        ),
        "block_angle_error_degrees": angle_error * 180.0 / math.pi,
        "goal_state_success": bool(
            position_error < 20.0 and angle_error < math.pi / 9.0
        ),
    }
