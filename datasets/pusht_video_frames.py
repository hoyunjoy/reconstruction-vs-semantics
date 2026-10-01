"""Efficient manifest-indexed Push-T video frame sampling for representation learning."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional

import decord
import numpy as np
import torch
from decord import VideoReader
from einops import rearrange
from torch.utils.data import Dataset

from .pusht_subset import load_split_indices


decord.bridge.set_bridge("torch")


def select_frame_indices(
    *,
    sequence_length: int,
    frame_count: int,
    strategy: str,
    seed: int,
    epoch: int,
    source_index: int,
) -> np.ndarray:
    """Select sorted frame indices reproducibly without touching global RNG state."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if frame_count == -1 or frame_count >= sequence_length:
        return np.arange(sequence_length, dtype=np.int64)
    if frame_count < 1:
        raise ValueError("frame_count must be positive or -1 for all frames")
    if strategy == "uniform":
        return np.linspace(0, sequence_length - 1, frame_count).round().astype(np.int64)
    if strategy == "random":
        seed_sequence = np.random.SeedSequence([seed, epoch, source_index])
        generator = np.random.Generator(np.random.PCG64(seed_sequence))
        return np.sort(
            generator.choice(sequence_length, size=frame_count, replace=False)
        ).astype(np.int64)
    raise ValueError(f"Unknown frame selection strategy: {strategy!r}")


class PushTManifestVideoFrames(Dataset):
    """Return several frames from one manifest trajectory per dataset item.

    Unlike ``PushTDataset``, this dataset does not load state/action tensors,
    because reconstruction training only needs images. One ``VideoReader`` is
    opened per item and all selected frames are decoded together.
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        manifest_path: str | Path,
        split: str,
        frame_count: int,
        strategy: str,
        seed: int,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.train_dir = self.data_root / "train"
        self.video_dir = self.train_dir / "obses"
        self.indices = tuple(load_split_indices(manifest_path, split))
        self.split = split
        self.frame_count = int(frame_count)
        self.strategy = strategy
        self.seed = int(seed)
        self.transform = transform
        self.epoch = 0

        sequence_path = self.train_dir / "seq_lengths.pkl"
        with sequence_path.open("rb") as handle:
            self.sequence_lengths = pickle.load(handle)
        if not self.video_dir.is_dir():
            raise FileNotFoundError(self.video_dir)
        if any(index < 0 or index >= len(self.sequence_lengths) for index in self.indices):
            raise IndexError("Manifest contains an out-of-range trajectory index")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def source_index(self, dataset_index: int) -> int:
        return self.indices[dataset_index]

    def get_sequence_length(self, dataset_index: int) -> int:
        return int(self.sequence_lengths[self.source_index(dataset_index)])

    def __getitem__(self, dataset_index: int) -> dict[str, torch.Tensor]:
        source_index = self.source_index(dataset_index)
        sequence_length = self.get_sequence_length(dataset_index)
        frame_indices = select_frame_indices(
            sequence_length=sequence_length,
            frame_count=self.frame_count,
            strategy=self.strategy,
            seed=self.seed,
            epoch=self.epoch,
            source_index=source_index,
        )
        video_path = self.video_dir / f"episode_{source_index:03d}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        reader = VideoReader(str(video_path), num_threads=1)
        images = reader.get_batch(frame_indices.tolist()).float().div(255.0)
        images = rearrange(images, "t h w c -> t c h w")
        if self.transform is not None:
            images = self.transform(images)
        if not torch.isfinite(images).all():
            raise FloatingPointError(f"Non-finite pixels in {video_path}")
        return {
            "images": images,
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "frame_indices": torch.from_numpy(frame_indices.copy()),
            "sequence_length": torch.tensor(sequence_length, dtype=torch.long),
        }
