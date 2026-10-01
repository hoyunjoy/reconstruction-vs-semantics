#!/usr/bin/env python3
"""Compute train-only channel statistics for cached Push-T representations."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.latent_normalization import StreamingChannelMoments


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def reusable_stats(
    path: Path,
    *,
    representation: str,
    manifest_sha256: str,
    expected_count: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return bool(
        payload.get("representation") == representation
        and payload.get("manifest_sha256") == manifest_sha256
        and payload.get("sample_count") == expected_count
        and tuple(payload.get("feature_shape_per_frame", [])) == (256, 384)
        and isinstance(payload.get("mean"), torch.Tensor)
        and isinstance(payload.get("std"), torch.Tensor)
        and tuple(payload["mean"].shape) == (384,)
        and tuple(payload["std"].shape) == (384,)
        and torch.isfinite(payload["mean"]).all()
        and torch.isfinite(payload["std"]).all()
        and torch.all(payload["std"] > 0)
    )


def compute_representation_stats(
    *,
    name: str,
    cache_root: Path,
    output_path: Path,
    train_indices: list[int],
    sequence_lengths,
    manifest_path: Path,
    manifest_sha256: str,
    device: torch.device,
    minimum_std: float,
    log_every: int,
    resume: bool,
) -> dict[str, Any]:
    expected_frames = sum(int(sequence_lengths[index]) for index in train_indices)
    expected_count = expected_frames * 256
    if resume and reusable_stats(
        output_path,
        representation=name,
        manifest_sha256=manifest_sha256,
        expected_count=expected_count,
    ):
        payload = torch.load(output_path, map_location="cpu", weights_only=True)
        print(f"reused validated statistics: {output_path}")
        return payload["summary"]

    moments = StreamingChannelMoments(channel_dim=384)
    for position, source_index in enumerate(train_indices, start=1):
        sequence_length = int(sequence_lengths[source_index])
        path = cache_root / "train" / f"episode_{source_index:05d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        features = payload["features"]
        if payload.get("source_index") != source_index:
            raise RuntimeError(f"Source index mismatch in {path}")
        if payload.get("sequence_length") != sequence_length:
            raise RuntimeError(f"Sequence length mismatch in {path}")
        if tuple(features.shape) != (sequence_length, 256, 384):
            raise RuntimeError(f"Feature shape mismatch in {path}")
        moments.update(features.to(device, non_blocking=True))
        if position == 1 or position % log_every == 0:
            print(
                f"representation={name} trajectories={position}/{len(train_indices)} "
                f"tokens={moments.count}",
                flush=True,
            )

    mean, std, sample_count = moments.finalize(minimum_std)
    if sample_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} samples for {name}, got {sample_count}"
        )
    raw_std = (moments.m2 / moments.count).clamp_min(0).sqrt()
    stored_mean = mean.double()
    stored_std = std.double()
    normalized_mean = (moments.mean - stored_mean) / stored_std
    normalized_std = raw_std / stored_std
    summary = {
        "representation": name,
        "trajectory_count": len(train_indices),
        "frame_count": expected_frames,
        "token_count": sample_count,
        "mean_min": float(mean.min().item()),
        "mean_max": float(mean.max().item()),
        "std_min": float(std.min().item()),
        "std_max": float(std.max().item()),
        "max_abs_normalized_mean": float(normalized_mean.abs().max().item()),
        "normalized_std_min": float(normalized_std.min().item()),
        "normalized_std_max": float(normalized_std.max().item()),
        "channels_clamped_to_minimum_std": int(
            (raw_std < minimum_std).sum().item()
        ),
    }
    stats_payload = {
        "schema_version": 1,
        "representation": name,
        "source_split": "train",
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "feature_shape_per_frame": [256, 384],
        "sample_axes": ["frame", "token"],
        "channel_axis": -1,
        "sample_count": sample_count,
        "minimum_std": minimum_std,
        "mean": mean,
        "std": std,
        "summary": summary,
    }
    atomic_torch_save(stats_payload, output_path)
    return summary


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/latent_stats_vit_dino",
)
def main(cfg: DictConfig) -> None:
    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest_path = Path(cfg.dataset.manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_digest = sha256_bytes(manifest_bytes)
    train_indices = [int(index) for index in manifest["splits"]["train"]]
    sequence_path = Path(cfg.dataset.sequence_lengths_path)
    with sequence_path.open("rb") as handle:
        sequence_lengths = pickle.load(handle)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, representation_cfg in cfg.representations.items():
        output_path = output_dir / str(representation_cfg.stats_filename)
        summaries[str(name)] = compute_representation_stats(
            name=str(name),
            cache_root=Path(representation_cfg.cache_root),
            output_path=output_path,
            train_indices=train_indices,
            sequence_lengths=sequence_lengths,
            manifest_path=manifest_path,
            manifest_sha256=manifest_digest,
            device=device,
            minimum_std=float(cfg.minimum_std),
            log_every=int(cfg.log_every),
            resume=bool(cfg.resume),
        )

    metadata = {
        "status": "completed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "source_split": "train",
        "normalization": "per-channel population mean/std over frames and tokens",
        "representations": summaries,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("Train-only latent normalization statistics: PASSED")


if __name__ == "__main__":
    main()
