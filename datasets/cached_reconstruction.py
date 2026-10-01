"""Aligned PushT images and frozen cached tokens for decoder probes."""

from pathlib import Path

import torch

from datasets.pusht_video_frames import PushTManifestVideoFrames


class CachedReconstructionDataset(PushTManifestVideoFrames):
    def __init__(self, *, cache_root: str | Path, **kwargs):
        super().__init__(**kwargs)
        self.cache_root = Path(cache_root)

    def __getitem__(self, dataset_index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(dataset_index)
        source_index = int(item["source_index"])
        path = self.cache_root / self.split / f"episode_{source_index:05d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        features = payload["features"]
        sequence_length = int(item["sequence_length"])
        if payload.get("source_index") != source_index:
            raise RuntimeError(f"Source index mismatch in {path}")
        if tuple(features.shape) != (sequence_length, 256, 384):
            raise RuntimeError(f"Feature shape mismatch in {path}")
        item["latents"] = features[item["frame_indices"]]
        return item
