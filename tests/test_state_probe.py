import math

import torch

from datasets.cached_state_probe import encode_pusht_state
from metrics.state_probe_metrics import (
    StateMetricAccumulator,
    circular_angle_error,
    decode_normalized_state_targets,
)
from models.state_probe import SpatialLinearStateProbe


def test_spatial_linear_probe_shape_and_backward():
    probe = SpatialLinearStateProbe(
        num_tokens=16,
        latent_dim=6,
        target_dim=8,
        grid_size=4,
        pool_sizes=(1, 2),
    )
    latents = torch.randn(3, 16, 6, requires_grad=True)
    output = probe(latents)
    assert output.shape == (3, 8)
    output.square().mean().backward()
    assert torch.isfinite(latents.grad).all()


def test_circular_encoding_and_decoding_handles_wraparound():
    state = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 2.0 * math.pi - 0.01, 5.0, 6.0]]
    )
    encoded = encode_pusht_state(state)
    mean = torch.zeros(8)
    std = torch.ones(8)
    decoded = decode_normalized_state_targets(encoded, mean, std)
    error = circular_angle_error(decoded[:, 4], state[:, 4])
    assert torch.allclose(decoded[:, :4], state[:, :4], atol=1.0e-6)
    assert torch.allclose(decoded[:, 5:], state[:, 5:], atol=1.0e-6)
    assert error.max().item() < 1.0e-6


def test_perfect_state_metrics_are_zero_and_successful():
    states = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0, 0.01, 1.0, 2.0],
            [11.0, 21.0, 31.0, 41.0, 2.0 * math.pi - 0.01, 3.0, 4.0],
        ]
    )
    mean = torch.zeros(8)
    std = torch.ones(8)
    prediction = encode_pusht_state(states)
    accumulator = StateMetricAccumulator(mean, std)
    accumulator.update(prediction, states)
    metrics = accumulator.compute()
    assert metrics["encoded_normalized_mse"] == 0.0
    assert metrics["agent_position_l2_rmse_px"] < 1.0e-6
    assert metrics["block_position_l2_rmse_px"] < 1.0e-6
    assert metrics["block_angle_mae_degrees"] < 1.0e-4
    assert metrics["pusht_threshold_accuracy"] == 1.0
