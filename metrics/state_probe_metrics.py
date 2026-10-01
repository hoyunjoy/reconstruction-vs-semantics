"""Physical and normalized metrics for PushT state decoding."""

from __future__ import annotations

import math

import torch

from datasets.cached_state_probe import encode_pusht_state


def decode_normalized_state_targets(
    normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Decode normalized 8D targets back to the physical 7D PushT state."""
    encoded = normalized.float() * std + mean
    angle = torch.atan2(encoded[..., 4], encoded[..., 5])
    angle = torch.remainder(angle, 2.0 * math.pi).unsqueeze(-1)
    return torch.cat((encoded[..., :4], angle, encoded[..., 6:]), dim=-1)


def circular_angle_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = prediction - target
    return torch.abs(
        torch.remainder(difference + math.pi, 2.0 * math.pi) - math.pi
    )


class StateMetricAccumulator:
    """Accumulate probe errors without retaining individual samples."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std
        self.encoded_squared_error = 0.0
        self.encoded_value_count = 0
        self.component_squared_error = torch.zeros(7, dtype=torch.float64)
        self.agent_position_squared_l2 = 0.0
        self.block_position_squared_l2 = 0.0
        self.velocity_squared_l2 = 0.0
        self.angle_absolute_error = 0.0
        self.threshold_successes = 0
        self.sample_count = 0

    def update(
        self, prediction_normalized: torch.Tensor, target_state: torch.Tensor
    ) -> None:
        target_state = target_state.float()
        target_encoded = encode_pusht_state(target_state)
        target_normalized = (target_encoded - self.mean) / self.std
        prediction_normalized = prediction_normalized.float()
        self.encoded_squared_error += float(
            (prediction_normalized - target_normalized).square().sum().item()
        )
        self.encoded_value_count += target_normalized.numel()

        prediction_state = decode_normalized_state_targets(
            prediction_normalized, self.mean, self.std
        )
        difference = prediction_state - target_state
        angle_error = circular_angle_error(
            prediction_state[..., 4], target_state[..., 4]
        )
        difference = difference.clone()
        difference[..., 4] = angle_error
        flat_difference = difference.reshape(-1, 7)
        flat_angle = angle_error.reshape(-1)
        count = flat_difference.shape[0]
        self.component_squared_error += flat_difference.double().square().sum(dim=0).cpu()
        self.agent_position_squared_l2 += float(
            flat_difference[:, :2].square().sum().item()
        )
        self.block_position_squared_l2 += float(
            flat_difference[:, 2:4].square().sum().item()
        )
        self.velocity_squared_l2 += float(
            flat_difference[:, 5:].square().sum().item()
        )
        self.angle_absolute_error += float(flat_angle.sum().item())
        position_error = flat_difference[:, :4].square().sum(dim=-1).sqrt()
        successes = (position_error < 20.0) & (flat_angle < math.pi / 9.0)
        self.threshold_successes += int(successes.sum().item())
        self.sample_count += count

    def compute(self) -> dict[str, float | int | list[float]]:
        if self.sample_count == 0:
            raise RuntimeError("No state-probe samples were accumulated")
        component_rmse = (
            self.component_squared_error / self.sample_count
        ).sqrt().tolist()
        return {
            "encoded_normalized_mse": (
                self.encoded_squared_error / self.encoded_value_count
            ),
            "agent_position_l2_rmse_px": (
                self.agent_position_squared_l2 / self.sample_count
            ) ** 0.5,
            "block_position_l2_rmse_px": (
                self.block_position_squared_l2 / self.sample_count
            ) ** 0.5,
            "block_angle_mae_degrees": (
                self.angle_absolute_error / self.sample_count * 180.0 / math.pi
            ),
            "agent_velocity_l2_rmse": (
                self.velocity_squared_l2 / self.sample_count
            ) ** 0.5,
            "component_rmse": component_rmse,
            "pusht_threshold_accuracy": (
                self.threshold_successes / self.sample_count
            ),
            "sample_count": self.sample_count,
        }
