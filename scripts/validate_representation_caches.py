#!/usr/bin/env python3
"""Validate cache/manifest alignment and sample tensor integrity."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/pusht_1000_seed42.json"),
    )
    parser.add_argument(
        "--sequence-lengths",
        type=Path,
        default=Path("data/pusht_noise/train/seq_lengths.pkl"),
    )
    parser.add_argument(
        "--vit-ae-root",
        type=Path,
        default=Path("feature_cache/vit_ae_s14_seed42_v1"),
    )
    parser.add_argument(
        "--dino-root",
        type=Path,
        default=Path("feature_cache/dinov2_vits14_seed42"),
    )
    parser.add_argument("--samples-per-split", type=int, default=3)
    return parser.parse_args()


def expected_paths(root: Path, split: str, indices: list[int]) -> set[Path]:
    return {root / split / f"episode_{index:05d}.pt" for index in indices}


def validate_root_files(root: Path, manifest: dict) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    for split in SPLITS:
        expected = expected_paths(root, split, manifest["splits"][split])
        actual = set((root / split).glob("episode_*.pt"))
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise RuntimeError(
                f"{root.name}/{split}: missing={len(missing)}, extra={len(extra)}"
            )


def sample_positions(length: int, count: int) -> list[int]:
    count = min(count, length)
    return np.linspace(0, length - 1, count).round().astype(int).tolist()


def validate_payload(
    path: Path, *, source_index: int, sequence_length: int
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload["features"]
    if payload["source_index"] != source_index:
        raise RuntimeError(f"Source index mismatch in {path}")
    if payload["sequence_length"] != sequence_length:
        raise RuntimeError(f"Sequence length mismatch in {path}")
    if tuple(features.shape) != (sequence_length, 256, 384):
        raise RuntimeError(f"Feature shape mismatch in {path}: {features.shape}")
    if features.dtype != torch.float16:
        raise RuntimeError(f"Feature dtype mismatch in {path}: {features.dtype}")
    if not torch.isfinite(features).all():
        raise FloatingPointError(f"Non-finite features in {path}")
    return payload


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    with args.sequence_lengths.open("rb") as handle:
        sequence_lengths = pickle.load(handle)
    roots = [args.vit_ae_root, args.dino_root]

    for root in roots:
        validate_root_files(root, manifest)

    checked = 0
    for split in SPLITS:
        indices = manifest["splits"][split]
        for position in sample_positions(len(indices), args.samples_per_split):
            source_index = int(indices[position])
            sequence_length = int(sequence_lengths[source_index])
            payloads = [
                validate_payload(
                    root / split / f"episode_{source_index:05d}.pt",
                    source_index=source_index,
                    sequence_length=sequence_length,
                )
                for root in roots
            ]
            if len(payloads) == 2:
                if payloads[0]["split"] != payloads[1]["split"]:
                    raise RuntimeError("Cross-cache split mismatch")
            checked += 1

    result = {
        "status": "passed",
        "roots": [str(root) for root in roots],
        "trajectory_counts": {
            split: len(manifest["splits"][split]) for split in SPLITS
        },
        "sampled_trajectories_checked_per_root": checked,
        "feature_shape_per_frame": [256, 384],
        "storage_dtype": "float16",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Representation cache validation: PASSED")


if __name__ == "__main__":
    main()
