#!/usr/bin/env python3
"""Create and validate a deterministic trajectory-level Push-T split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


REQUIRED_TENSORS = (
    "states.pth",
    "abs_actions.pth",
    "rel_actions.pth",
    "velocities.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/pusht_noise"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/splits/pusht_1000_seed42.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--expected-candidate-count", type=int, default=18_685)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_metadata_counts(train_dir: Path) -> dict[str, int]:
    sequence_path = train_dir / "seq_lengths.pkl"
    if not sequence_path.is_file():
        raise FileNotFoundError(sequence_path)

    with sequence_path.open("rb") as handle:
        sequence_lengths = pickle.load(handle)
    counts = {"seq_lengths.pkl": len(sequence_lengths)}

    for filename in REQUIRED_TENSORS:
        path = train_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise TypeError(f"Expected a batched tensor in {path}")
        counts[filename] = int(tensor.shape[0])
        del tensor

    if len(set(counts.values())) != 1:
        raise ValueError(f"Push-T metadata counts disagree: {counts}")
    return counts


def build_manifest(
    *,
    candidate_count: int,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> dict[str, Any]:
    split_total = train_count + validation_count + test_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("All split counts must be positive")
    if split_total > candidate_count:
        raise ValueError(
            f"Requested {split_total} trajectories from only {candidate_count}"
        )

    generator = np.random.Generator(np.random.PCG64(seed))
    selected = generator.permutation(candidate_count)[:split_total].tolist()
    train_end = train_count
    validation_end = train_end + validation_count

    return {
        "schema_version": 1,
        "dataset": {
            "name": "pusht_noise",
            "source_split": "train",
            "candidate_count": candidate_count,
            "official_val_used": False,
        },
        "sampling": {
            "seed": seed,
            "algorithm": "numpy.random.Generator(numpy.random.PCG64).permutation",
            "numpy_version": np.__version__,
            "selected_count": split_total,
            "split_counts": {
                "train": train_count,
                "validation": validation_count,
                "test": test_count,
            },
        },
        "splits": {
            "train": selected[:train_end],
            "validation": selected[train_end:validation_end],
            "test": selected[validation_end:],
        },
    }


def validate_manifest(
    manifest: dict[str, Any], data_root: Path, metadata_counts: dict[str, int]
) -> None:
    train_dir = data_root / "train"
    candidate_count = manifest["dataset"]["candidate_count"]
    if any(count != candidate_count for count in metadata_counts.values()):
        raise ValueError(
            f"Manifest candidate count {candidate_count} does not match "
            f"metadata {metadata_counts}"
        )

    split_indices = manifest["splits"]
    expected_counts = manifest["sampling"]["split_counts"]
    actual_counts = {name: len(indices) for name, indices in split_indices.items()}
    if actual_counts != expected_counts:
        raise ValueError(
            f"Expected split counts {expected_counts}, got {actual_counts}"
        )

    flattened = [index for indices in split_indices.values() for index in indices]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Trajectory indices overlap across splits")
    if any(index < 0 or index >= candidate_count for index in flattened):
        raise IndexError("Manifest contains an out-of-range trajectory index")

    video_dir = train_dir / "obses"
    missing_videos = [
        index
        for index in flattened
        if not (video_dir / f"episode_{index:03d}.mp4").is_file()
    ]
    if missing_videos:
        preview = missing_videos[:10]
        raise FileNotFoundError(
            f"Missing {len(missing_videos)} selected videos; first: {preview}"
        )


def render_manifest(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    args = parse_args()
    train_dir = args.data_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Push-T train split not found: {train_dir}")

    metadata_counts = load_metadata_counts(train_dir)
    candidate_count = metadata_counts["seq_lengths.pkl"]
    if candidate_count != args.expected_candidate_count:
        raise ValueError(
            f"Expected {args.expected_candidate_count} candidates, found "
            f"{candidate_count}"
        )

    manifest_kwargs = {
        "candidate_count": candidate_count,
        "seed": args.seed,
        "train_count": args.train_count,
        "validation_count": args.validation_count,
        "test_count": args.test_count,
    }
    manifest = build_manifest(**manifest_kwargs)
    repeated_manifest = build_manifest(**manifest_kwargs)
    rendered = render_manifest(manifest)
    if rendered != render_manifest(repeated_manifest):
        raise RuntimeError("Manifest generation is not byte-deterministic")

    validate_manifest(manifest, args.data_root, metadata_counts)

    if args.output.exists() and not args.overwrite:
        existing = args.output.read_bytes()
        if existing != rendered:
            raise FileExistsError(
                f"Existing manifest differs: {args.output}; use --overwrite only "
                "after reviewing the change"
            )
        status = "verified existing"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary_path.write_bytes(rendered)
        temporary_path.replace(args.output)
        status = "wrote"

    digest = hashlib.sha256(rendered).hexdigest()
    print(f"{status}: {args.output}")
    split_counts = manifest["sampling"]["split_counts"]
    print(
        "split counts: "
        f"train={split_counts['train']} "
        f"validation={split_counts['validation']} "
        f"test={split_counts['test']}"
    )
    print(f"sha256: {digest}")
    print("Push-T split manifest validation: PASSED")


if __name__ == "__main__":
    main()
