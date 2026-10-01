#!/usr/bin/env python3
"""Train one shared dynamics architecture on either cached representation."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.cached_dynamics import CachedDynamicsTrajectoryDataset
from metrics.latent_normalization import LatentNormalizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def build_dataset(
    cfg: DictConfig,
    *,
    split: str,
    prediction_horizon: int,
) -> CachedDynamicsTrajectoryDataset:
    strategy = "random" if split == "train" else "uniform"
    return CachedDynamicsTrajectoryDataset(
        cache_root=cfg.cached_representation.cache_root,
        manifest_path=cfg.dataset.manifest_path,
        sequence_lengths_path=cfg.dataset.sequence_lengths_path,
        actions_path=cfg.dataset.actions_path,
        split=split,
        context_length=int(cfg.dataset.context_length),
        prediction_horizon=prediction_horizon,
        windows_per_trajectory=int(cfg.dataset.windows_per_trajectory[split]),
        strategy=strategy,
        seed=int(cfg.seed),
        action_scale=float(cfg.dataset.action_scale),
        action_mean=list(cfg.dataset.action_mean),
        action_std=list(cfg.dataset.action_std),
    )


def build_loader(
    dataset: CachedDynamicsTrajectoryDataset,
    cfg: DictConfig,
    *,
    shuffle: bool,
    epoch: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg.dataset.trajectories_per_batch),
        shuffle=shuffle,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(int(cfg.seed) + epoch),
        drop_last=False,
    )


def flatten_windows(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = batch["latents"]
    actions = batch["actions"]
    if latents.ndim != 5 or actions.ndim != 4:
        raise ValueError(
            f"Expected BKLPC latents and BKLA actions, got "
            f"{latents.shape}, {actions.shape}"
        )
    latents = latents.flatten(0, 1).to(device, non_blocking=True)
    actions = actions.flatten(0, 1).to(device, non_blocking=True)
    return latents, actions


def train_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: LatentNormalizer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    context_length: int,
    use_amp: bool,
    gradient_clip_norm: float,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.train()
    squared_error = 0.0
    value_count = 0
    sequence_count = 0
    for batch_index, batch in enumerate(loader, start=1):
        raw_latents, actions = flatten_windows(batch, device)
        latents = normalizer.normalize(raw_latents.float())
        history = latents[:, :context_length]
        target = latents[:, context_length]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            prediction = model(history, actions[:, :context_length])
            loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
            raise FloatingPointError("Dynamics training produced NaN or Inf")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        squared_error += float(
            (prediction.detach().float() - target.float()).square().sum().item()
        )
        value_count += target.numel()
        sequence_count += target.shape[0]
        if max_batches is not None and batch_index >= max_batches:
            break
    return {
        "normalized_mse": squared_error / value_count,
        "sequence_count": sequence_count,
    }


@torch.no_grad()
def evaluate_one_step(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: LatentNormalizer,
    device: torch.device,
    context_length: int,
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    squared_error = 0.0
    copy_squared_error = 0.0
    value_count = 0
    sequence_count = 0
    for batch_index, batch in enumerate(loader, start=1):
        raw_latents, actions = flatten_windows(batch, device)
        latents = normalizer.normalize(raw_latents.float())
        history = latents[:, :context_length]
        target = latents[:, context_length]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            prediction = model(history, actions[:, :context_length])
        squared_error += float(
            (prediction.float() - target.float()).square().sum().item()
        )
        copy_squared_error += float(
            (history[:, -1].float() - target.float()).square().sum().item()
        )
        value_count += target.numel()
        sequence_count += target.shape[0]
        if max_batches is not None and batch_index >= max_batches:
            break
    return {
        "normalized_mse": squared_error / value_count,
        "copy_baseline_normalized_mse": copy_squared_error / value_count,
        "sequence_count": sequence_count,
    }


@torch.no_grad()
def evaluate_rollouts(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: LatentNormalizer,
    device: torch.device,
    context_length: int,
    horizons: list[int],
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    max_horizon = max(horizons)
    accumulators = {
        horizon: {
            "squared_error": 0.0,
            "copy_squared_error": 0.0,
            "cosine_sum": 0.0,
            "value_count": 0,
            "sequence_count": 0,
        }
        for horizon in horizons
    }
    for batch_index, batch in enumerate(loader, start=1):
        raw_latents, actions = flatten_windows(batch, device)
        latents = normalizer.normalize(raw_latents.float())
        initial = latents[:, :context_length]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            predictions = model.rollout(initial, actions, max_horizon)
        for horizon in horizons:
            prediction = predictions[:, horizon - 1]
            target = latents[:, context_length + horizon - 1]
            values = accumulators[horizon]
            values["squared_error"] += float(
                (prediction.float() - target.float()).square().sum().item()
            )
            values["copy_squared_error"] += float(
                (initial[:, -1].float() - target.float()).square().sum().item()
            )
            cosine = F.cosine_similarity(
                prediction.flatten(1).float(), target.flatten(1).float(), dim=1
            )
            values["cosine_sum"] += float(cosine.sum().item())
            values["value_count"] += target.numel()
            values["sequence_count"] += target.shape[0]
        if max_batches is not None and batch_index >= max_batches:
            break

    results = {}
    for horizon, values in accumulators.items():
        results[str(horizon)] = {
            "normalized_mse": values["squared_error"] / values["value_count"],
            "copy_baseline_normalized_mse": (
                values["copy_squared_error"] / values["value_count"]
            ),
            "cosine_similarity": values["cosine_sum"] / values["sequence_count"],
            "sequence_count": values["sequence_count"],
        }
    return results


def make_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    cfg: DictConfig,
    epoch: int,
    best_validation_mse: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_mse": best_validation_mse,
        "history": history,
        "representation": str(cfg.cached_representation.name),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/latent_dynamics",
)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Latent dynamics training requires CUDA")
    seed_everything(int(cfg.seed))
    device = torch.device("cuda")
    context_length = int(cfg.dataset.context_length)
    horizons = [int(value) for value in cfg.evaluation.horizons]
    max_train_batches = (
        None
        if cfg.training.max_batches_per_epoch is None
        else int(cfg.training.max_batches_per_epoch)
    )
    max_eval_batches = (
        None
        if cfg.evaluation.max_batches is None
        else int(cfg.evaluation.max_batches)
    )
    if 1 not in horizons:
        raise ValueError("Evaluation horizons must include 1")

    output_dir = Path(cfg.outputs.output_dir)
    checkpoint_dir = Path(cfg.outputs.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / cfg.outputs.last_checkpoint_name
    best_path = checkpoint_dir / cfg.outputs.best_checkpoint_name

    train_dataset = build_dataset(cfg, split="train", prediction_horizon=1)
    validation_dataset = build_dataset(
        cfg, split="validation", prediction_horizon=1
    )
    rollout_validation_dataset = build_dataset(
        cfg, split="validation", prediction_horizon=max(horizons)
    )
    rollout_test_dataset = build_dataset(
        cfg, split="test", prediction_horizon=max(horizons)
    )
    normalizer = LatentNormalizer.from_file(
        str(cfg.cached_representation.stats_path), device=device
    )
    model = instantiate(cfg.dynamics).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.epochs),
        eta_min=float(cfg.training.min_lr),
    )
    use_amp = bool(cfg.training.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best_validation_mse = float("inf")
    history: list[dict[str, Any]] = []
    early_stopping_patience = int(cfg.training.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))
    if early_stopping_patience < 0:
        raise ValueError("training.early_stopping_patience must be non-negative")
    if early_stopping_min_delta < 0:
        raise ValueError("training.early_stopping_min_delta must be non-negative")
    early_stopping_best_validation_mse = float("inf")
    epochs_without_improvement = 0

    if bool(cfg.training.resume) and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["representation"] != str(cfg.cached_representation.name):
            raise RuntimeError("Checkpoint representation does not match config")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_mse = float(checkpoint["best_validation_mse"])
        history = checkpoint["history"]
        early_stopping_best_validation_mse = float(
            checkpoint.get("early_stopping_best_validation_mse", best_validation_mse)
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        print(f"Resumed from epoch {start_epoch}: {last_path}")

    torch.cuda.reset_peak_memory_stats(device)
    epochs = int(cfg.training.epochs)
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        train_loader = build_loader(train_dataset, cfg, shuffle=True, epoch=epoch)
        validation_loader = build_loader(
            validation_dataset, cfg, shuffle=False, epoch=0
        )
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            normalizer=normalizer,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            context_length=context_length,
            use_amp=use_amp,
            gradient_clip_norm=float(cfg.training.gradient_clip_norm),
            max_batches=max_train_batches,
        )
        validation_metrics = evaluate_one_step(
            model=model,
            loader=validation_loader,
            normalizer=normalizer,
            device=device,
            context_length=context_length,
            use_amp=use_amp,
            max_batches=max_eval_batches,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        validation_mse = float(validation_metrics["normalized_mse"])
        improved = validation_mse < best_validation_mse
        if improved:
            best_validation_mse = validation_mse
        early_stopping_improved = (
            validation_mse
            < early_stopping_best_validation_mse - early_stopping_min_delta
        )
        if early_stopping_improved:
            early_stopping_best_validation_mse = validation_mse
            epochs_without_improvement = 0
        elif early_stopping_patience > 0:
            epochs_without_improvement += 1
        payload = make_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
            best_validation_mse=best_validation_mse,
            history=history,
        )
        payload["early_stopping_best_validation_mse"] = (
            early_stopping_best_validation_mse
        )
        payload["epochs_without_improvement"] = epochs_without_improvement
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, best_path)
        (output_dir / cfg.outputs.history_name).write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"representation={cfg.cached_representation.name} "
            f"epoch={epoch + 1:03d}/{epochs} "
            f"train_mse={train_metrics['normalized_mse']:.7f} "
            f"val_mse={validation_mse:.7f} "
            f"copy={validation_metrics['copy_baseline_normalized_mse']:.7f} "
            f"best={best_validation_mse:.7f}",
            flush=True,
        )
        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"no validation improvement greater than "
                f"{early_stopping_min_delta:g} for "
                f"{early_stopping_patience} epochs",
                flush=True,
            )
            break

    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    validation_rollouts = evaluate_rollouts(
        model=model,
        loader=build_loader(
            rollout_validation_dataset, cfg, shuffle=False, epoch=0
        ),
        normalizer=normalizer,
        device=device,
        context_length=context_length,
        horizons=horizons,
        use_amp=use_amp,
        max_batches=max_eval_batches,
    )
    test_rollouts = evaluate_rollouts(
        model=model,
        loader=build_loader(rollout_test_dataset, cfg, shuffle=False, epoch=0),
        normalizer=normalizer,
        device=device,
        context_length=context_length,
        horizons=horizons,
        use_amp=use_amp,
        max_batches=max_eval_batches,
    )
    metrics = {
        "status": "completed",
        "representation": str(cfg.cached_representation.name),
        "seed": int(cfg.seed),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(best_checkpoint["epoch"]),
        "trained_epochs": len(history),
        "early_stopped": len(history) < epochs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "best_validation_one_step_mse": float(
            best_checkpoint["best_validation_mse"]
        ),
        "validation_rollouts": validation_rollouts,
        "test_rollouts": test_rollouts,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "best_checkpoint": str(best_path),
    }
    metrics_path = output_dir / cfg.outputs.metrics_name
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("Latent dynamics training and rollout evaluation: PASSED")


if __name__ == "__main__":
    main()
