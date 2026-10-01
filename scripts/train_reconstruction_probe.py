#!/usr/bin/env python3
"""Train an identical decoder on frozen normalized ViT-AE or DINOv2 tokens."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cached_reconstruction import CachedReconstructionDataset
from datasets.img_transforms import default_transform
from metrics.latent_normalization import LatentNormalizer
from metrics.reconstruction_eval import ReconstructionAccumulator, to_zero_one
from metrics.lpipsPyTorch.modules.lpips import LPIPS


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    value = torch.initial_seed() % 2**32
    random.seed(value); np.random.seed(value)


def dataset(cfg, split):
    return CachedReconstructionDataset(
        cache_root=cfg.cached_representation.cache_root,
        data_root=cfg.dataset.data_root,
        manifest_path=cfg.dataset.manifest_path,
        split=split,
        frame_count=int(cfg.dataset.frames_per_trajectory[split]),
        strategy="random" if split == "train" else "uniform",
        seed=int(cfg.seed),
        transform=default_transform(int(cfg.reconstruction_decoder.image_size)),
    )


def loader(data, cfg, shuffle, epoch):
    return DataLoader(
        data, batch_size=int(cfg.dataset.trajectories_per_batch), shuffle=shuffle,
        num_workers=int(cfg.dataset.num_workers), pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(int(cfg.seed) + epoch),
        persistent_workers=False, drop_last=False,
    )


def flatten(batch, device):
    latents, images = batch["latents"], batch["images"]
    if latents.ndim != 4 or images.ndim != 5:
        raise ValueError(f"Unexpected shapes: {latents.shape}, {images.shape}")
    return (
        latents.flatten(0, 1).to(device, non_blocking=True),
        images.flatten(0, 1).to(device, non_blocking=True),
    )


def train_epoch(*, model, data_loader, normalizer, optimizer, scaler, device,
                amp, clip, max_batches):
    model.train(); error = 0.0; count = 0; images_seen = 0
    for batch_index, batch in enumerate(data_loader, 1):
        raw, images = flatten(batch, device)
        tokens = normalizer.normalize(raw.float())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            recon = model(tokens); loss = F.mse_loss(recon, images)
        if not torch.isfinite(loss) or not torch.isfinite(recon).all():
            raise FloatingPointError("Decoder training produced NaN/Inf")
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer); scaler.update()
        error += float((recon.detach().float() - images).square().sum())
        count += images.numel(); images_seen += images.shape[0]
        if max_batches is not None and batch_index >= max_batches: break
    return {"mse_minus_one_to_one": error / count, "image_count": images_seen}


@torch.no_grad()
def evaluate(*, model, data_loader, normalizer, device, amp, ssim, max_batches,
             collect_grid=False, compute_lpips=False):
    model.eval(); accumulator = ReconstructionAccumulator(compute_ssim=ssim)
    lpips_metric = LPIPS(net_type="vgg", version="0.1").to(device).eval() if compute_lpips else None
    lpips_sum = 0.0; lpips_count = 0
    grid = None
    for batch_index, batch in enumerate(data_loader, 1):
        raw, images = flatten(batch, device)
        tokens = normalizer.normalize(raw.float())
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            recon = model(tokens)
        reconstruction = recon.float()
        accumulator.update(images, reconstruction)
        if lpips_metric is not None:
            batch_size = images.shape[0]
            lpips_sum += float(lpips_metric(images, reconstruction)) * batch_size
            lpips_count += batch_size
        if collect_grid and grid is None:
            n = min(8, images.shape[0]); grid = (images[:n].cpu(), recon[:n].float().cpu())
        if max_batches is not None and batch_index >= max_batches: break
    metrics = accumulator.compute()
    if lpips_metric is not None:
        metrics["lpips_vgg"] = lpips_sum / lpips_count
    return metrics, grid


def atomic_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def payload(model, optimizer, scheduler, scaler, cfg, epoch, best, history,
            early_stopping_best, epochs_without_improvement):
    return {
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch, "best_validation_mse": best, "history": history,
        "early_stopping_best": early_stopping_best,
        "epochs_without_improvement": epochs_without_improvement,
        "representation": str(cfg.cached_representation.name),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


@hydra.main(version_base=None, config_path="../conf", config_name="experiment/reconstruction_probe")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    seed_all(int(cfg.seed)); device = torch.device("cuda")
    amp = bool(cfg.training.amp)
    max_train = None if cfg.training.max_batches_per_epoch is None else int(cfg.training.max_batches_per_epoch)
    max_eval = None if cfg.evaluation.max_batches is None else int(cfg.evaluation.max_batches)
    train_data, val_data, test_data = (dataset(cfg, split) for split in ("train", "validation", "test"))
    normalizer = LatentNormalizer.from_file(str(cfg.cached_representation.stats_path), device=device)
    model = instantiate(cfg.reconstruction_decoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.learning_rate), weight_decay=float(cfg.training.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.training.epochs), eta_min=float(cfg.training.min_lr))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint_dir, output_dir = Path(cfg.outputs.checkpoint_dir), Path(cfg.outputs.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = checkpoint_dir / cfg.outputs.best_checkpoint_name, checkpoint_dir / cfg.outputs.last_checkpoint_name
    start, best, history = 0, float("inf"), []
    early_stopping_patience = int(cfg.training.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))
    if early_stopping_patience < 0 or early_stopping_min_delta < 0:
        raise ValueError("Early-stopping patience and min_delta must be non-negative")
    early_stopping_best = float("inf")
    epochs_without_improvement = 0
    early_stopped = False
    if bool(cfg.training.resume) and last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["representation"] != str(cfg.cached_representation.name): raise RuntimeError("Representation mismatch")
        model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"]); scaler.load_state_dict(saved["scaler_state_dict"])
        start, best, history = int(saved["epoch"]) + 1, float(saved["best_validation_mse"]), saved["history"]
        early_stopping_best = float(saved.get("early_stopping_best", best))
        epochs_without_improvement = int(saved.get("epochs_without_improvement", 0))
        print(f"Resumed from epoch {start}: {last_path}")
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, int(cfg.training.epochs)):
        train_data.set_epoch(epoch)
        train_metrics = train_epoch(model=model, data_loader=loader(train_data, cfg, True, epoch), normalizer=normalizer,
            optimizer=optimizer, scaler=scaler, device=device, amp=amp, clip=float(cfg.training.gradient_clip_norm), max_batches=max_train)
        val_metrics, _ = evaluate(model=model, data_loader=loader(val_data, cfg, False, 0), normalizer=normalizer,
            device=device, amp=amp, ssim=False, max_batches=max_eval)
        scheduler.step(); validation_mse = float(val_metrics["mse_minus_one_to_one"])
        improved = validation_mse < best
        if improved: best = validation_mse
        if validation_mse < early_stopping_best - early_stopping_min_delta:
            early_stopping_best = validation_mse
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics, "learning_rate": optimizer.param_groups[0]["lr"]})
        saved = payload(model, optimizer, scheduler, scaler, cfg, epoch, best, history,
            early_stopping_best, epochs_without_improvement)
        atomic_save(saved, last_path)
        if improved: atomic_save(saved, best_path)
        (output_dir / cfg.outputs.history_name).write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        print(f"representation={cfg.cached_representation.name} epoch={epoch+1:03d}/{cfg.training.epochs} train_mse={train_metrics['mse_minus_one_to_one']:.7f} val_mse={val_metrics['mse_minus_one_to_one']:.7f} val_psnr={val_metrics['psnr_db']:.3f} best={best:.7f} patience={epochs_without_improvement}/{early_stopping_patience}", flush=True)
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            early_stopped = True
            print(f"Early stopping at epoch {epoch+1}; best validation MSE={best:.7f}", flush=True)
            break
    saved = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(saved["model_state_dict"])
    test_metrics, grid = evaluate(model=model, data_loader=loader(test_data, cfg, False, 0), normalizer=normalizer,
        device=device, amp=amp, ssim=True, max_batches=max_eval, collect_grid=True, compute_lpips=True)
    if grid is None: raise RuntimeError("No grid images")
    originals, reconstructions = grid
    paired = torch.stack((to_zero_one(originals), to_zero_one(reconstructions)), 1).flatten(0, 1)
    grid_path = output_dir / cfg.outputs.grid_name; save_image(make_grid(paired, nrow=2, padding=2), grid_path)
    metrics = {"status": "completed", "representation": str(cfg.cached_representation.name), "seed": int(cfg.seed),
        "decoder_parameter_count": sum(p.numel() for p in model.parameters()), "best_epoch": int(saved["epoch"]),
        "best_validation_mse": float(saved["best_validation_mse"]), "test": test_metrics,
        "trained_epochs": len(history), "early_stopped": early_stopped,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "gpu_peak_memory_gib": torch.cuda.max_memory_allocated(device)/1024**3,
        "best_checkpoint": str(best_path), "reconstruction_grid": str(grid_path),
        "latent_input": "train-only channel-normalized frozen cache"}
    (output_dir / cfg.outputs.metrics_name).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Matched frozen-latent reconstruction probe: PASSED")


if __name__ == "__main__": main()
