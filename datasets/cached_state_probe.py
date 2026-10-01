"""Datasets and train-only target statistics for PushT state probes."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.cached_dynamics import (
    CachedDynamicsTrajectoryDataset,
    select_window_starts,
)


def encode_pusht_state(states: torch.Tensor) -> torch.Tensor:
    """Encode [agent xy, block xy, angle, velocity xy] with a circular angle."""
    if states.shape[-1] != 7:
        raise ValueError(f"Expected seven PushT state values, got {states.shape}")
    angle = states[..., 4:5]
    return torch.cat(
        (
            states[..., :4],
            torch.sin(angle),
            torch.cos(angle),
            states[..., 5:],
        ),
        dim=-1,
    )


def load_raw_states(
    states_path: str | Path, velocities_path: str | Path
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.load(
        states_path, map_location="cpu", weights_only=True, mmap=True
    )
    velocities = torch.load(
        velocities_path, map_location="cpu", weights_only=True, mmap=True
    )
    if states.shape[:-1] != velocities.shape[:-1]:
        raise ValueError("State and velocity tensor shapes do not align")
    if states.shape[-1] != 5 or velocities.shape[-1] != 2:
        raise ValueError(
            f"Expected state/velocity dimensions 5 and 2, got "
            f"{states.shape[-1]} and {velocities.shape[-1]}"
        )
    return states, velocities


def compute_train_state_statistics(
    *,
    manifest_path: str | Path,
    sequence_lengths_path: str | Path,
    states_path: str | Path,
    velocities_path: str | Path,
    minimum_std: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute encoded-state mean/std using valid frames from train only."""
    manifest = json.loads(Path(manifest_path).read_text())
    indices = [int(index) for index in manifest["splits"]["train"]]
    with Path(sequence_lengths_path).open("rb") as handle:
        sequence_lengths = pickle.load(handle)
    states, velocities = load_raw_states(states_path, velocities_path)

    total = torch.zeros(8, dtype=torch.float64)
    total_square = torch.zeros(8, dtype=torch.float64)
    count = 0
    for source_index in indices:
        length = int(sequence_lengths[source_index])
        raw = torch.cat(
            (
                states[source_index, :length].float(),
                velocities[source_index, :length].float(),
            ),
            dim=-1,
        )
        encoded = encode_pusht_state(raw).double()
        total += encoded.sum(dim=0)
        total_square += encoded.square().sum(dim=0)
        count += length
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(float(minimum_std))
    return mean.float(), std.float(), count


class CachedStateProbeDataset(Dataset):
    """Return reproducibly sampled cached latents and physical PushT states."""

    def __init__(
        self,
        *,
        cache_root: str | Path,
        manifest_path: str | Path,
        sequence_lengths_path: str | Path,
        states_path: str | Path,
        velocities_path: str | Path,
        split: str,
        frames_per_trajectory: int,
        strategy: str,
        seed: int,
    ) -> None:
        self.cache_root = Path(cache_root)
        manifest = json.loads(Path(manifest_path).read_text())
        self.indices = tuple(int(index) for index in manifest["splits"][split])
        self.split = split
        self.frames_per_trajectory = int(frames_per_trajectory)
        self.strategy = strategy
        self.seed = int(seed)
        self.epoch = 0
        with Path(sequence_lengths_path).open("rb") as handle:
            self.sequence_lengths = pickle.load(handle)
        self.states, self.velocities = load_raw_states(
            states_path, velocities_path
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, dataset_index: int) -> dict[str, torch.Tensor]:
        source_index = self.indices[dataset_index]
        sequence_length = int(self.sequence_lengths[source_index])
        frame_indices = select_window_starts(
            sequence_length=sequence_length,
            window_length=1,
            window_count=self.frames_per_trajectory,
            strategy=self.strategy,
            seed=self.seed,
            epoch=self.epoch,
            source_index=source_index,
        )
        cache_path = (
            self.cache_root / self.split / f"episode_{source_index:05d}.pt"
        )
        payload = torch.load(
            cache_path, map_location="cpu", weights_only=True, mmap=True
        )
        features = payload["features"]
        if payload.get("source_index") != source_index:
            raise RuntimeError(f"Source index mismatch in {cache_path}")
        if tuple(features.shape) != (sequence_length, 256, 384):
            raise RuntimeError(f"Feature shape mismatch in {cache_path}")
        frame_tensor = torch.from_numpy(frame_indices.copy())
        raw_states = torch.cat(
            (
                self.states[source_index, frame_tensor].float(),
                self.velocities[source_index, frame_tensor].float(),
            ),
            dim=-1,
        )
        return {
            "latents": features[frame_tensor],
            "states": raw_states,
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "frames": frame_tensor,
        }


class CachedStateRolloutDataset(CachedDynamicsTrajectoryDataset):
    """Dynamics windows augmented with matching physical-state windows."""

    def __init__(
        self,
        *,
        states_path: str | Path,
        velocities_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.raw_states, self.velocities = load_raw_states(
            states_path, velocities_path
        )

    def __getitem__(self, dataset_index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(dataset_index)
        source_index = int(item["source_index"].item())
        state_windows = []
        for start in item["starts"].tolist():
            stop = start + self.window_length
            state_windows.append(
                torch.cat(
                    (
                        self.raw_states[source_index, start:stop].float(),
                        self.velocities[source_index, start:stop].float(),
                    ),
                    dim=-1,
                )
            )
        item["states"] = torch.stack(state_windows)
        return item
