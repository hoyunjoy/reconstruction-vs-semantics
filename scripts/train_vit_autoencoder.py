"""Train the scratch ViT-S/14 reconstruction encoder on the fixed PushT split.

This is intentionally standalone: it does not modify existing checkpoints,
configs, caches, or DINOv2 features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.vit_autoencoder import ViTAutoencoder


def _extract_split_indices(manifest: dict[str, Any], split: str) -> list[int]:
    candidates: list[Any] = []
    if split in manifest:
        candidates.append(manifest[split])
    if isinstance(manifest.get("splits"), dict) and split in manifest["splits"]:
        candidates.append(manifest["splits"][split])

    for value in candidates:
        if isinstance(value, dict):
            for key in ("indices", "trajectory_indices", "episodes"):
                if key in value:
                    value = value[key]
                    break
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    for key in ("index", "trajectory_index", "episode", "id"):
                        if key in item:
                            item = item[key]
                            break
                result.append(int(item))
            return result
    raise ValueError(f"Could not find split '{split}' in manifest")


class PushTManifestFrames(Dataset):
    """One item is a deterministic set of frames from one trajectory."""

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        split: str,
        frames_per_trajectory: int,
        image_size: int,
        seed: int,
        random_sampling: bool,
    ):
        from decord import VideoReader, cpu

        self.VideoReader = VideoReader
        self.cpu = cpu
        self.data_root = Path(data_root)
        self.split = split
        self.frames_per_trajectory = int(frames_per_trajectory)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.random_sampling = bool(random_sampling)
        self.epoch = 0

        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.indices = _extract_split_indices(manifest, split)
        if not self.indices:
            raise ValueError(f"Split '{split}' is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def _sample_frame_indices(self, length: int, trajectory_index: int) -> np.ndarray:
        count = self.frames_per_trajectory
        if self.random_sampling:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, trajectory_index])
            )
            return np.sort(rng.choice(length, size=count, replace=length < count))
        if count == 1:
            return np.asarray([length // 2], dtype=np.int64)
        return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        trajectory_index = self.indices[item]
        video_root = self.data_root / "train" / "obses"
        candidates = [
            video_root / f"episode_{trajectory_index:03d}.mp4",
            video_root / f"episode_{trajectory_index:05d}.mp4",
            video_root / f"episode_{trajectory_index}.mp4",
        ]
        video_path = next((path for path in candidates if path.is_file()), None)
        if video_path is None:
            raise FileNotFoundError(
                "No PushT video matched: " + ", ".join(map(str, candidates))
            )

        reader = self.VideoReader(str(video_path), ctx=self.cpu(0), num_threads=1)
        frame_indices = self._sample_frame_indices(len(reader), trajectory_index)
        frames = reader.get_batch(frame_indices).asnumpy()
        images = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        images = images.mul_(2.0).sub_(1.0)
        return {
            "images": images,
            "trajectory_index": torch.tensor(trajectory_index, dtype=torch.long),
            "frame_indices": torch.from_numpy(frame_indices.copy()).long(),
        }


@dataclass(frozen=True)
class TrainingConfig:
    data_root: str
    manifest: str
    checkpoint_dir: str
    output_dir: str
    torch_hub_repo: str | None
    seed: int = 42
    image_size: int = 224
    train_frames_per_trajectory: int = 32
    validation_frames_per_trajectory: int = 16
    trajectories_per_batch: int = 2
    workers: int = 4
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_relative_delta: float = 1e-3
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    amp: bool = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _flatten_images(batch: dict[str, Tensor], device: torch.device) -> Tensor:
    images = batch["images"]
    return images.flatten(0, 1).to(device, non_blocking=True)


def run_epoch(
    model: ViTAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    gradient_clip_norm: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_values = 0

    for batch in loader:
        images = _flatten_images(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                reconstruction = model(images)
                loss = F.mse_loss(reconstruction, images)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        total_squared_error += float(loss.detach()) * images.numel()
        total_values += images.numel()

    return total_squared_error / total_values


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def train(config: TrainingConfig) -> None:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This full experiment requires a CUDA GPU")

    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir = Path(config.output_dir)
    if checkpoint_dir.exists() or output_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing experiment. Choose new checkpoint/output paths."
        )
    checkpoint_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    save_json(output_dir / "config.json", asdict(config))

    train_dataset = PushTManifestFrames(
        config.data_root,
        config.manifest,
        "train",
        config.train_frames_per_trajectory,
        config.image_size,
        config.seed,
        random_sampling=True,
    )
    validation_dataset = PushTManifestFrames(
        config.data_root,
        config.manifest,
        "validation",
        config.validation_frames_per_trajectory,
        config.image_size,
        config.seed,
        random_sampling=False,
    )
    generator = torch.Generator().manual_seed(config.seed)
    common_loader_args = dict(
        batch_size=config.trajectories_per_batch,
        num_workers=config.workers,
        pin_memory=True,
        # Workers must be recreated after set_epoch so each epoch samples new
        # training frames deterministically.
        persistent_workers=False,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_args,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_args,
    )

    model = ViTAutoencoder(
        image_size=config.image_size,
        torch_hub_repo=config.torch_hub_repo,
    ).to(device)
    counts = model.parameter_counts()
    if counts["decoder"] != 2_611_543:
        raise RuntimeError(f"Decoder changed unexpectedly: {counts}")
    save_json(output_dir / "parameter_counts.json", counts)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=config.minimum_learning_rate,
    )
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, float | int]] = []
    best_validation_mse = math.inf
    early_stopping_reference = math.inf
    best_epoch = -1
    epochs_without_material_improvement = 0

    for epoch in range(config.max_epochs):
        train_dataset.set_epoch(epoch)
        train_mse = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            use_amp,
            config.gradient_clip_norm,
        )
        validation_mse = run_epoch(
            model,
            validation_loader,
            device,
            None,
            scaler,
            use_amp,
            config.gradient_clip_norm,
        )
        validation_psnr = 10.0 * math.log10(4.0 / validation_mse)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        improved = validation_mse < best_validation_mse
        material_improvement = (
            not math.isfinite(early_stopping_reference)
            or validation_mse
            < early_stopping_reference
            * (1.0 - config.early_stopping_min_relative_delta)
        )
        if improved:
            best_validation_mse = validation_mse
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": asdict(config),
                    "parameter_counts": counts,
                    "best_validation_mse": best_validation_mse,
                },
                checkpoint_dir / "best.pt",
            )
        if material_improvement:
            early_stopping_reference = validation_mse
            epochs_without_material_improvement = 0
        else:
            epochs_without_material_improvement += 1

        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "validation_psnr_db": validation_psnr,
            "learning_rate": learning_rate,
        }
        history.append(row)
        save_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch + 1:03d}/{config.max_epochs} "
            f"train_mse={train_mse:.7f} val_mse={validation_mse:.7f} "
            f"val_psnr={validation_psnr:.3f} best={best_validation_mse:.7f}",
            flush=True,
        )

        if epochs_without_material_improvement >= config.early_stopping_patience:
            print(
                "Early stopping: validation reconstruction loss has plateaued.",
                flush=True,
            )
            break

    torch.save(
        {
            "epoch": history[-1]["epoch"],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(config),
            "parameter_counts": counts,
            "best_validation_mse": best_validation_mse,
            "best_epoch": best_epoch,
        },
        checkpoint_dir / "last.pt",
    )
    save_json(
        output_dir / "metrics.json",
        {
            "status": "completed",
            "best_epoch_zero_based": best_epoch,
            "best_validation_mse": best_validation_mse,
            "epochs_completed": len(history),
            "stopped_for_plateau": len(history) < config.max_epochs,
            "parameter_counts": counts,
        },
    )


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/pusht_noise")
    parser.add_argument(
        "--manifest",
        default="artifacts/splits/pusht_1000_seed42.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/vit_ae_s14_seed42_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/vit_ae_s14_seed42_v1",
    )
    parser.add_argument(
        "--torch-hub-repo",
        default=None,
        help="Optional local facebookresearch/dinov2 checkout or torch-hub cache path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-relative-delta", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    return TrainingConfig(
        data_root=args.data_root,
        manifest=args.manifest,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        torch_hub_repo=args.torch_hub_repo,
        seed=args.seed,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_relative_delta=args.early_stopping_min_relative_delta,
        workers=args.workers,
        amp=not args.no_amp,
    )


if __name__ == "__main__":
    train(parse_args())
