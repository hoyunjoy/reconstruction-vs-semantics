"""Trajectory-index subsets for the released Push-T training split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from .pusht_dset import PushTDataset
from .traj_dset import TrajDataset


VALID_SPLITS = ("train", "validation", "test")


def load_split_indices(manifest_path: str | Path, split: str) -> list[int]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    if manifest["dataset"].get("source_split") != "train":
        raise ValueError("This experiment requires subsets of the public train split")
    indices = manifest["splits"][split]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate indices within manifest split {split}")
    return [int(index) for index in indices]


class PushTIndexSubset(TrajDataset):
    """Map subset positions to immutable indices in a base PushTDataset."""

    def __init__(self, dataset: PushTDataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        if len(self.indices) != len(set(self.indices)):
            raise ValueError("Push-T subset indices must be unique")
        if any(index < 0 or index >= len(dataset) for index in self.indices):
            raise IndexError("Push-T subset contains an out-of-range index")

    def __len__(self) -> int:
        return len(self.indices)

    def source_index(self, index: int) -> int:
        return self.indices[index]

    def get_seq_length(self, index: int) -> int:
        return self.dataset.get_seq_length(self.source_index(index))

    def get_frames(self, index: int, frames):
        return self.dataset.get_frames(self.source_index(index), frames)

    def __getitem__(self, index: int):
        return self.dataset[self.source_index(index)]

    def get_all_actions(self) -> torch.Tensor:
        actions = []
        for source_index in self.indices:
            sequence_length = self.dataset.get_seq_length(source_index)
            actions.append(self.dataset.actions[source_index, :sequence_length])
        return torch.cat(actions, dim=0)

    def __getattr__(self, name: str) -> Any:
        dataset = self.__dict__.get("dataset")
        if dataset is not None and hasattr(dataset, name):
            return getattr(dataset, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


def load_pusht_manifest_subset(
    *,
    data_path: str | Path,
    manifest_path: str | Path,
    split: str,
    transform: Optional[Callable] = None,
    normalize_action: bool = True,
    relative: bool = True,
    with_velocity: bool = True,
) -> PushTIndexSubset:
    """Load one manifest split without copying or renaming source trajectories."""

    base_dataset = PushTDataset(
        n_rollout=None,
        transform=transform,
        data_path=str(Path(data_path) / "train"),
        normalize_action=normalize_action,
        relative=relative,
        with_velocity=with_velocity,
    )
    indices = load_split_indices(manifest_path, split)
    return PushTIndexSubset(base_dataset, indices)
