import math

import numpy as np
import torch

from planning.latent_cem_mpc import cem_plan, circular_angle_error, goal_state_metrics


def test_circular_angle_error_wraps():
    assert circular_angle_error(0.01, 2 * math.pi - 0.01) < 0.021


def test_goal_state_metrics_success_and_failure():
    goal = np.array([100, 100, 200, 200, 0.01, 0, 0], dtype=np.float32)
    close = np.array([105, 104, 204, 203, 2 * math.pi - 0.01, 0, 0], dtype=np.float32)
    far = close.copy()
    far[2] += 30
    assert goal_state_metrics(close, goal)["goal_state_success"]
    assert not goal_state_metrics(far, goal)["goal_state_success"]


def test_cem_reduces_terminal_latent_cost():
    class AdditiveDynamics:
        def rollout(self, initial, actions, horizon):
            future = actions[:, -horizon:, :1].cumsum(dim=1)
            return future.unsqueeze(-1)

    context = torch.zeros(1, 3, 1, 1)
    past_actions = torch.zeros(2, 2)
    goal = torch.full((1, 1, 1), 2.0)
    plan, cost = cem_plan(
        dynamics=AdditiveDynamics(),
        latent_context=context,
        past_actions=past_actions,
        goal_latent=goal,
        horizon=2,
        num_samples=256,
        num_elites=16,
        iterations=5,
        initial_std=1.0,
        minimum_std=0.01,
        action_clip=3.0,
        smoothing=0.0,
        generator=torch.Generator().manual_seed(42),
        use_amp=False,
    )
    assert plan.shape == (2, 2)
    assert cost < 0.05
