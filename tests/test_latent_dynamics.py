import numpy as np
import pytest
import torch

from datasets.cached_dynamics import select_window_starts
from models.latent_dynamics import FactorizedActionDynamics


def test_window_selection_is_reproducible_and_epoch_dependent():
    arguments = {
        "sequence_length": 100,
        "window_length": 13,
        "window_count": 8,
        "strategy": "random",
        "seed": 42,
        "source_index": 15,
    }
    first = select_window_starts(epoch=0, **arguments)
    repeated = select_window_starts(epoch=0, **arguments)
    next_epoch = select_window_starts(epoch=1, **arguments)
    np.testing.assert_array_equal(first, repeated)
    assert len(first) == len(np.unique(first)) == 8
    assert not np.array_equal(first, next_epoch)


def test_factorized_dynamics_forward_rollout_and_backward():
    model = FactorizedActionDynamics(
        num_tokens=8,
        latent_dim=24,
        spatial_heads=6,
        spatial_depth=1,
        spatial_mlp_dim=48,
        action_hidden_dim=16,
        dropout=0.0,
    )
    latent_history = torch.randn(2, 3, 8, 24)
    action_sequence = torch.randn(2, 12, 2)

    one_step = model(latent_history, action_sequence[:, :3])
    rollout = model.rollout(latent_history, action_sequence, horizon=10)
    loss = one_step.square().mean() + rollout.square().mean()
    loss.backward()

    assert one_step.shape == (2, 8, 24)
    assert rollout.shape == (2, 10, 8, 24)
    assert torch.isfinite(one_step).all()
    assert torch.isfinite(rollout).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_rollout_rejects_too_few_actions():
    model = FactorizedActionDynamics(
        num_tokens=8,
        latent_dim=24,
        spatial_heads=6,
        spatial_depth=1,
        spatial_mlp_dim=48,
    )
    with pytest.raises(ValueError, match="actions"):
        model.rollout(
            torch.zeros(1, 3, 8, 24),
            torch.zeros(1, 5, 2),
            horizon=10,
        )
