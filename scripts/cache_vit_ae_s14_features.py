"""Cache patch tokens from the best scratch ViT-AE checkpoint.

The output intentionally matches the existing manifest cache layout:
one float16 ``(T, 256, 384)`` tensor per trajectory and split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.img_transforms import default_transform
from datasets.pusht_video_frames import PushTManifestVideoFrames
from models.vit_autoencoder import ViTAutoencoder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def valid_existing_cache(
    path: Path, *, source_index: int, sequence_length: int, checkpoint_sha256: str
) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    features = payload.get("features")
    return bool(
        payload.get("source_index") == source_index
        and payload.get("sequence_length") == sequence_length
        and payload.get("checkpoint_sha256") == checkpoint_sha256
        and isinstance(features, torch.Tensor)
        and tuple(features.shape) == (sequence_length, 256, 384)
        and features.dtype == torch.float16
        and torch.isfinite(features).all()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vit_ae_s14_seed42_v1/best.pt",
    )
    parser.add_argument("--data-root", default="data/pusht_noise")
    parser.add_argument(
        "--manifest",
        default="artifacts/splits/pusht_1000_seed42.json",
    )
    parser.add_argument(
        "--output-root",
        default="feature_cache/vit_ae_s14_seed42_v1",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation", "test"]
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ViT-AE feature caching requires CUDA")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_digest = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if Path(config["manifest"]).resolve() != Path(args.manifest).resolve():
        raise RuntimeError("Checkpoint and cache manifest paths do not match")

    model = ViTAutoencoder(
        image_size=int(config["image_size"]),
        torch_hub_repo=config.get("torch_hub_repo"),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False).cuda()

    output_root = Path(args.output_root)
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite existing cache: {output_root}. Use --resume."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    totals = {"written": 0, "skipped": 0, "frames": 0}
    split_summaries: dict[str, dict[str, int]] = {}

    for split in args.splits:
        dataset = PushTManifestVideoFrames(
            data_root=args.data_root,
            manifest_path=args.manifest,
            split=split,
            frame_count=-1,
            strategy="uniform",
            seed=int(args.seed),
            transform=default_transform(int(config["image_size"])),
        )
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        written = skipped = frame_total = 0

        for dataset_index in range(len(dataset)):
            source_index = dataset.source_index(dataset_index)
            sequence_length = dataset.get_sequence_length(dataset_index)
            output_path = split_dir / f"episode_{source_index:05d}.pt"
            if output_path.is_file() and args.resume:
                if valid_existing_cache(
                    output_path,
                    source_index=source_index,
                    sequence_length=sequence_length,
                    checkpoint_sha256=checkpoint_digest,
                ):
                    skipped += 1
                    frame_total += sequence_length
                    continue
                raise RuntimeError(f"Invalid existing cache file: {output_path}")

            images = dataset[dataset_index]["images"]
            feature_batches = []
            with torch.inference_mode():
                for start in range(0, sequence_length, int(args.batch_size)):
                    image_batch = images[start : start + int(args.batch_size)].cuda(
                        non_blocking=True
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=not args.no_amp,
                    ):
                        features = model.encode(image_batch)
                    feature_batches.append(features.cpu().to(torch.float16))

            features = torch.cat(feature_batches, dim=0).contiguous()
            if tuple(features.shape) != (sequence_length, 256, 384):
                raise RuntimeError(
                    f"Episode {source_index}: unexpected shape {tuple(features.shape)}"
                )
            if not torch.isfinite(features).all():
                raise FloatingPointError(
                    f"Episode {source_index}: cached features contain NaN/Inf"
                )

            payload = {
                "representation": "vit_ae_s14_reconstruction",
                "feature_key": "x_norm_patchtokens",
                "split": split,
                "source_index": source_index,
                "sequence_length": sequence_length,
                "features": features,
                "checkpoint_sha256": checkpoint_digest,
                "best_epoch_zero_based": int(checkpoint["epoch"]),
                "best_validation_mse": float(checkpoint["best_validation_mse"]),
            }
            temporary_path = output_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary_path)
            os.replace(temporary_path, output_path)
            written += 1
            frame_total += sequence_length
            if written == 1 or written % int(args.log_every) == 0:
                print(
                    f"split={split} written={written}/{len(dataset)} "
                    f"skipped={skipped} source={source_index} frames={frame_total}",
                    flush=True,
                )

        summary = {
            "trajectory_count": len(dataset),
            "written_this_run": written,
            "skipped_existing": skipped,
            "total_frames": frame_total,
        }
        split_summaries[split] = summary
        totals["written"] += written
        totals["skipped"] += skipped
        totals["frames"] += frame_total
        atomic_json(split_dir / "metadata.json", summary)

    metadata = {
        "status": "completed",
        "representation": "vit_ae_s14_reconstruction",
        "feature_key": "x_norm_patchtokens",
        "feature_shape_per_frame": [256, 384],
        "storage_dtype": "float16",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "best_epoch_zero_based": int(checkpoint["epoch"]),
        "best_validation_mse": float(checkpoint["best_validation_mse"]),
        "manifest_path": str(Path(args.manifest)),
        "splits": split_summaries,
        "totals": totals,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    atomic_json(output_root / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print("ViT-AE manifest feature cache: PASSED", flush=True)


if __name__ == "__main__":
    main()

