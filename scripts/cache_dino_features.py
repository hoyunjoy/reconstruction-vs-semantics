#!/usr/bin/env python3
"""Cache frozen DINOv2 patch features for a Push-T dataset split.

This smoke-test utility intentionally leaves the original DINO-WM source files
untouched. Each trajectory is written atomically to its own file so an
interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.img_transforms import default_transform  # noqa: E402
from datasets.pusht_dset import PushTDataset  # noqa: E402
from models.dino import DinoV2Encoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DINO_WM_DATA_ROOT", "data"))
        / "pusht_noise",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("feature_cache/dinov2_smoke100"),
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this feature-cache smoke test")

    split_dir = args.data_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Push-T split not found: {split_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    dataset = PushTDataset(
        n_rollout=args.num_trajectories,
        transform=default_transform(args.image_size),
        data_path=str(split_dir),
        normalize_action=True,
        relative=True,
        with_velocity=True,
    )
    if len(dataset) != args.num_trajectories:
        raise RuntimeError(
            f"Requested {args.num_trajectories} trajectories, got {len(dataset)}"
        )

    encoder = DinoV2Encoder(
        name="dinov2_vits14",
        feature_key="x_norm_patchtokens",
    ).to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    expected_tokens = (args.image_size // encoder.patch_size) ** 2
    expected_dim = encoder.emb_dim
    total_frames = 0
    written = 0
    skipped = 0

    for trajectory_index in range(len(dataset)):
        output_path = args.output_dir / f"episode_{trajectory_index:05d}.pt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        sequence_length = int(dataset.get_seq_length(trajectory_index))
        obs, _, _, _ = dataset.get_frames(
            trajectory_index, range(sequence_length)
        )
        images = obs["visual"]
        feature_batches = []

        with torch.inference_mode():
            for start in range(0, sequence_length, args.batch_size):
                image_batch = images[start : start + args.batch_size].to(
                    device, non_blocking=True
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = encoder(image_batch)
                features = features.to(device="cpu", dtype=torch.float16)
                feature_batches.append(features)

        cached_features = torch.cat(feature_batches, dim=0).contiguous()
        expected_shape = (sequence_length, expected_tokens, expected_dim)
        if tuple(cached_features.shape) != expected_shape:
            raise RuntimeError(
                f"Episode {trajectory_index}: expected {expected_shape}, "
                f"got {tuple(cached_features.shape)}"
            )
        if not torch.isfinite(cached_features).all():
            raise RuntimeError(f"Episode {trajectory_index}: non-finite features")

        payload = {
            "episode_index": trajectory_index,
            "sequence_length": sequence_length,
            "features": cached_features,
        }
        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)

        total_frames += sequence_length
        written += 1
        if written == 1 or written % 10 == 0:
            print(
                f"cached={written:3d} skipped={skipped:3d} "
                f"episode={trajectory_index:5d} frames={total_frames:6d}",
                flush=True,
            )

    metadata = {
        "data_root": str(args.data_root),
        "split": args.split,
        "num_trajectories": args.num_trajectories,
        "written_this_run": written,
        "skipped_existing": skipped,
        "frames_written_this_run": total_frames,
        "encoder": "dinov2_vits14",
        "feature_key": "x_norm_patchtokens",
        "image_size": args.image_size,
        "feature_shape_per_frame": [expected_tokens, expected_dim],
        "storage_dtype": "float16",
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"cache directory: {args.output_dir}")
    print(f"metadata: {metadata_path}")
    print("DINOv2 100-trajectory cache smoke test: PASSED")


if __name__ == "__main__":
    main()
