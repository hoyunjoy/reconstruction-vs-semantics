#!/usr/bin/env python3
"""Cache frozen DINOv2 patch tokens for the fixed Push-T manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.img_transforms import default_transform
from datasets.pusht_video_frames import PushTManifestVideoFrames


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash parameter names, metadata, and bytes before moving the model to CUDA."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def valid_existing_cache(
    path: Path,
    *,
    source_index: int,
    sequence_length: int,
    encoder_digest: str,
) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    features = payload.get("features")
    return bool(
        payload.get("representation") == "dinov2_vits14"
        and payload.get("source_index") == source_index
        and payload.get("sequence_length") == sequence_length
        and payload.get("encoder_state_sha256") == encoder_digest
        and isinstance(features, torch.Tensor)
        and tuple(features.shape) == (sequence_length, 256, 384)
        and features.dtype == torch.float16
        and torch.isfinite(features).all()
    )


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/cache_dino_manifest",
)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DINOv2 feature caching requires CUDA")
    encoder = instantiate(cfg.encoder)
    encoder_digest = state_dict_sha256(encoder)
    encoder.eval().requires_grad_(False).cuda()
    expected_tokens = (int(cfg.image_size) // int(encoder.patch_size)) ** 2
    expected_dim = int(encoder.emb_dim)
    if (expected_tokens, expected_dim) != (256, 384):
        raise RuntimeError(
            f"Expected DINO shape (256, 384), got "
            f"({expected_tokens}, {expected_dim})"
        )

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    totals = {"written": 0, "skipped": 0, "frames": 0}
    split_summaries = {}

    for split_value in cfg.splits:
        split = str(split_value)
        dataset = PushTManifestVideoFrames(
            data_root=cfg.dataset.data_root,
            manifest_path=cfg.dataset.manifest_path,
            split=split,
            frame_count=-1,
            strategy="uniform",
            seed=int(cfg.seed),
            transform=default_transform(int(cfg.image_size)),
        )
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        skipped = 0
        frame_total = 0

        for dataset_index in range(len(dataset)):
            source_index = dataset.source_index(dataset_index)
            sequence_length = dataset.get_sequence_length(dataset_index)
            output_path = split_dir / f"episode_{source_index:05d}.pt"
            if output_path.is_file() and bool(cfg.resume):
                if valid_existing_cache(
                    output_path,
                    source_index=source_index,
                    sequence_length=sequence_length,
                    encoder_digest=encoder_digest,
                ):
                    skipped += 1
                    frame_total += sequence_length
                    continue
                raise RuntimeError(f"Invalid existing cache file: {output_path}")

            sample = dataset[dataset_index]
            images = sample["images"]
            feature_batches = []
            with torch.inference_mode():
                for start in range(0, sequence_length, int(cfg.batch_size)):
                    image_batch = images[start : start + int(cfg.batch_size)].cuda(
                        non_blocking=True
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=bool(cfg.amp),
                    ):
                        features = encoder(image_batch)
                    feature_batches.append(features.cpu().to(torch.float16))
            features = torch.cat(feature_batches, dim=0).contiguous()
            expected_shape = (sequence_length, expected_tokens, expected_dim)
            if tuple(features.shape) != expected_shape:
                raise RuntimeError(
                    f"Episode {source_index}: expected {expected_shape}, "
                    f"got {tuple(features.shape)}"
                )
            if not torch.isfinite(features).all():
                raise FloatingPointError(
                    f"Episode {source_index}: DINO features contain NaN/Inf"
                )
            payload = {
                "representation": "dinov2_vits14",
                "feature_key": "x_norm_patchtokens",
                "split": split,
                "source_index": source_index,
                "sequence_length": sequence_length,
                "features": features,
                "encoder_state_sha256": encoder_digest,
            }
            temporary_path = output_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary_path)
            temporary_path.replace(output_path)
            written += 1
            frame_total += sequence_length
            if written == 1 or written % int(cfg.log_every) == 0:
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
        (split_dir / "metadata.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    metadata = {
        "status": "completed",
        "representation": "dinov2_vits14",
        "feature_key": "x_norm_patchtokens",
        "feature_shape_per_frame": [expected_tokens, expected_dim],
        "storage_dtype": "float16",
        "encoder_state_sha256": encoder_digest,
        "manifest_path": str(Path(cfg.dataset.manifest_path)),
        "splits": split_summaries,
        "totals": totals,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("DINOv2 manifest feature cache: PASSED")


if __name__ == "__main__":
    main()

