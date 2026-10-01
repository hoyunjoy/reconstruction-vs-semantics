"""Trajectory-window datasets backed by cached visual representations."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def select_window_starts(
    *,
    sequence_length: int,
    window_length: int,
    window_count: int,
    strategy: str,
    seed: int,
    epoch: int,
    source_index: int,
) -> np.ndarray:
    available = sequence_length - window_length + 1
    if available < 1:
        raise ValueError(
            f"Sequence length {sequence_length} is shorter than window "
            f"length {window_length}"
        )
    if window_count == -1:
        return np.arange(available, dtype=np.int64)
    if window_count < 1:
        raise ValueError("window_count must be positive or -1")
    if window_count > available:
        raise ValueError(
            f"Requested {window_count} windows, but only {available} are available"
        )
    if window_count == available:
        return np.arange(available, dtype=np.int64)
    if strategy == "uniform":
        return np.linspace(0, available - 1, window_count).round().astype(np.int64)
    if strategy == "random":
        seed_sequence = np.random.SeedSequence([seed, epoch, source_index])
        generator = np.random.Generator(np.random.PCG64(seed_sequence))
        return np.sort(
            generator.choice(available, size=window_count, replace=False)
        ).astype(np.int64)
    raise ValueError(f"Unknown window strategy: {strategy!r}")


class CachedDynamicsTrajectoryDataset(Dataset):
    """Load one trajectory cache and return several contiguous windows."""

    def __init__(
        self,
        *,
        cache_root: str | Path,
        manifest_path: str | Path,
        sequence_lengths_path: str | Path,
        actions_path: str | Path,
        split: str,
        context_length: int,
        prediction_horizon: int,
        windows_per_trajectory: int,
        strategy: str,
        seed: int,
        action_scale: float,
        action_mean: list[float],
        action_std: list[float],
    ):
        self.cache_root = Path(cache_root)
        manifest = json.loads(Path(manifest_path).read_text())
        self.indices = tuple(int(index) for index in manifest["splits"][split])
        self.split = split
        self.context_length = int(context_length)
        self.prediction_horizon = int(prediction_horizon)
        self.window_length = self.context_length + self.prediction_horizon
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.strategy = strategy
        self.seed = int(seed)
        self.epoch = 0

        with Path(sequence_lengths_path).open("rb") as handle:
            self.sequence_lengths = pickle.load(handle)
        all_actions = torch.load(
            actions_path, map_location="cpu", weights_only=True, mmap=True
        )
        action_mean_tensor = torch.tensor(action_mean, dtype=torch.float32)
        action_std_tensor = torch.tensor(action_std, dtype=torch.float32)
        if action_scale <= 0 or not torch.all(action_std_tensor > 0):
            raise ValueError("Action scale and standard deviations must be positive")
        selected_actions = all_actions[list(self.indices)].float().div(action_scale)
        self.actions = (selected_actions - action_mean_tensor) / action_std_tensor
        self.source_to_local = {
            source_index: local_index
            for local_index, source_index in enumerate(self.indices)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def source_index(self, dataset_index: int) -> int:
        return self.indices[dataset_index]

    def __getitem__(self, dataset_index: int) -> dict[str, torch.Tensor]:
        source_index = self.source_index(dataset_index)
        sequence_length = int(self.sequence_lengths[source_index])
        starts = select_window_starts(
            sequence_length=sequence_length,
            window_length=self.window_length,
            window_count=self.windows_per_trajectory,
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

        local_index = self.source_to_local[source_index]
        trajectory_actions = self.actions[local_index]
        latent_windows = torch.stack(
            [
                features[start : start + self.window_length]
                for start in starts.tolist()
            ]
        )
        action_windows = torch.stack(
            [
                trajectory_actions[
                    start : start + self.window_length - 1
                ]
                for start in starts.tolist()
            ]
        )
        return {
            "latents": latent_windows,
            "actions": action_windows,
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "starts": torch.from_numpy(starts.copy()),
        }
