#!/usr/bin/env python3
"""Train a frozen-latent linear state probe and evaluate dynamics rollouts."""

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

from datasets.cached_state_probe import (
    CachedStateProbeDataset,
    CachedStateRolloutDataset,
    compute_train_state_statistics,
    encode_pusht_state,
)
from metrics.latent_normalization import LatentNormalizer
from metrics.state_probe_metrics import StateMetricAccumulator


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


def build_loader(
    dataset: torch.utils.data.Dataset,
    cfg: DictConfig,
    *,
    batch_size: int,
    shuffle: bool,
    epoch: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(int(cfg.seed) + epoch),
        drop_last=False,
    )


def build_probe_dataset(
    cfg: DictConfig, split: str
) -> CachedStateProbeDataset:
    return CachedStateProbeDataset(
        cache_root=cfg.cached_representation.cache_root,
        manifest_path=cfg.dataset.manifest_path,
        sequence_lengths_path=cfg.dataset.sequence_lengths_path,
        states_path=cfg.dataset.states_path,
        velocities_path=cfg.dataset.velocities_path,
        split=split,
        frames_per_trajectory=int(cfg.dataset.frames_per_trajectory[split]),
        strategy="random" if split == "train" else "uniform",
        seed=int(cfg.seed),
    )


def build_rollout_dataset(
    cfg: DictConfig, split: str, max_horizon: int
) -> CachedStateRolloutDataset:
    return CachedStateRolloutDataset(
        cache_root=cfg.cached_representation.cache_root,
        manifest_path=cfg.dataset.manifest_path,
        sequence_lengths_path=cfg.dataset.sequence_lengths_path,
        actions_path=cfg.dataset.actions_path,
        states_path=cfg.dataset.states_path,
        velocities_path=cfg.dataset.velocities_path,
        split=split,
        context_length=int(cfg.dataset.context_length),
        prediction_horizon=max_horizon,
        windows_per_trajectory=int(cfg.dataset.rollout_windows_per_trajectory[split]),
        strategy="uniform",
        seed=int(cfg.seed),
        action_scale=float(cfg.dataset.action_scale),
        action_mean=list(cfg.dataset.action_mean),
        action_std=list(cfg.dataset.action_std),
    )


def flatten_probe_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = batch["latents"]
    states = batch["states"]
    if latents.ndim != 4 or states.ndim != 3:
        raise ValueError(
            f"Expected BKPC latents and BKD states, got {latents.shape}, "
            f"{states.shape}"
        )
    return (
        latents.flatten(0, 1).to(device, non_blocking=True),
        states.flatten(0, 1).to(device, non_blocking=True),
    )


def train_epoch(
    *,
    probe: torch.nn.Module,
    loader: DataLoader,
    latent_normalizer: LatentNormalizer,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float,
    max_batches: int | None,
) -> dict[str, float | int]:
    probe.train()
    squared_error = 0.0
    value_count = 0
    sample_count = 0
    for batch_index, batch in enumerate(loader, start=1):
        raw_latents, states = flatten_probe_batch(batch, device)
        latents = latent_normalizer.normalize(raw_latents.float())
        targets = (encode_pusht_state(states) - target_mean) / target_std
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            predictions = probe(latents)
            loss = F.mse_loss(predictions, targets)
        if not torch.isfinite(loss) or not torch.isfinite(predictions).all():
            raise FloatingPointError("State-probe training produced NaN or Inf")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(probe.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        squared_error += float(
            (predictions.detach().float() - targets.float()).square().sum().item()
        )
        value_count += targets.numel()
        sample_count += targets.shape[0]
        if max_batches is not None and batch_index >= max_batches:
            break
    return {
        "encoded_normalized_mse": squared_error / value_count,
        "sample_count": sample_count,
    }


@torch.no_grad()
def evaluate_probe(
    *,
    probe: torch.nn.Module,
    loader: DataLoader,
    latent_normalizer: LatentNormalizer,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, float | int | list[float]]:
    probe.eval()
    accumulator = StateMetricAccumulator(target_mean, target_std)
    for batch_index, batch in enumerate(loader, start=1):
        raw_latents, states = flatten_probe_batch(batch, device)
        latents = latent_normalizer.normalize(raw_latents.float())
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            predictions = probe(latents)
        if not torch.isfinite(predictions).all():
            raise FloatingPointError("State-probe evaluation produced NaN or Inf")
        accumulator.update(predictions.float(), states)
        if max_batches is not None and batch_index >= max_batches:
            break
    return accumulator.compute()


def load_frozen_dynamics(
    checkpoint_path: str | Path,
    representation: str,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint["representation"] != representation:
        raise RuntimeError("Dynamics checkpoint representation does not match")
    dynamics_config = OmegaConf.create(checkpoint["config"]["dynamics"])
    dynamics = instantiate(dynamics_config).to(device)
    dynamics.load_state_dict(checkpoint["model_state_dict"])
    dynamics.eval()
    for parameter in dynamics.parameters():
        parameter.requires_grad_(False)
    return dynamics


def flatten_rollout_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = batch["latents"]
    actions = batch["actions"]
    states = batch["states"]
    if latents.ndim != 5 or actions.ndim != 4 or states.ndim != 4:
        raise ValueError(
            f"Unexpected rollout shapes: {latents.shape}, {actions.shape}, "
            f"{states.shape}"
        )
    return (
        latents.flatten(0, 1).to(device, non_blocking=True),
        actions.flatten(0, 1).to(device, non_blocking=True),
        states.flatten(0, 1).to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate_state_rollouts(
    *,
    probe: torch.nn.Module,
    dynamics: torch.nn.Module,
    loader: DataLoader,
    latent_normalizer: LatentNormalizer,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    context_length: int,
    horizons: list[int],
    device: torch.device,
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, dict[str, Any]]:
    probe.eval()
    dynamics.eval()
    max_horizon = max(horizons)
    modes = (
        "dynamics_probe",
        "oracle_latent_probe",
        "copy_latent_probe",
        "true_state_copy",
    )
    accumulators = {
        horizon: {
            mode: StateMetricAccumulator(target_mean, target_std)
            for mode in modes
        }
        for horizon in horizons
    }
    for batch_index, batch in enumerate(loader, start=1):
        raw_latents, actions, states = flatten_rollout_batch(batch, device)
        latents = latent_normalizer.normalize(raw_latents.float())
        initial_latents = latents[:, :context_length]
        initial_state = states[:, context_length - 1]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            predicted_latents = dynamics.rollout(
                initial_latents, actions, max_horizon
            )
            predicted_state_targets = probe(
                predicted_latents.flatten(0, 1)
            ).reshape(
                predicted_latents.shape[0], max_horizon, -1
            )
            copy_latent_targets = probe(initial_latents[:, -1])
        initial_state_targets = (
            encode_pusht_state(initial_state) - target_mean
        ) / target_std
        if not torch.isfinite(predicted_state_targets).all():
            raise FloatingPointError("State rollout produced NaN or Inf")

        for horizon in horizons:
            target_state = states[:, context_length + horizon - 1]
            oracle_latent = latents[:, context_length + horizon - 1]
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                oracle_target = probe(oracle_latent)
            values = accumulators[horizon]
            values["dynamics_probe"].update(
                predicted_state_targets[:, horizon - 1].float(), target_state
            )
            values["oracle_latent_probe"].update(
                oracle_target.float(), target_state
            )
            values["copy_latent_probe"].update(
                copy_latent_targets.float(), target_state
            )
            values["true_state_copy"].update(
                initial_state_targets.float(), target_state
            )
        if max_batches is not None and batch_index >= max_batches:
            break

    return {
        str(horizon): {
            mode: accumulators[horizon][mode].compute() for mode in modes
        }
        for horizon in horizons
    }


def make_checkpoint(
    *,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    cfg: DictConfig,
    epoch: int,
    best_validation_mse: float,
    history: list[dict[str, Any]],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    target_stat_count: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": probe.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_mse": best_validation_mse,
        "history": history,
        "representation": str(cfg.cached_representation.name),
        "target_mean": target_mean.detach().cpu(),
        "target_std": target_std.detach().cpu(),
        "target_stat_count": target_stat_count,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/state_probe",
)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("State-probe training requires CUDA")
    seed_everything(int(cfg.seed))
    device = torch.device("cuda")
    use_amp = bool(cfg.training.amp)
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

    output_dir = Path(cfg.outputs.output_dir)
    checkpoint_dir = Path(cfg.outputs.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / cfg.outputs.last_checkpoint_name
    best_path = checkpoint_dir / cfg.outputs.best_checkpoint_name

    train_dataset = build_probe_dataset(cfg, "train")
    validation_dataset = build_probe_dataset(cfg, "validation")
    test_dataset = build_probe_dataset(cfg, "test")
    target_mean_cpu, target_std_cpu, target_stat_count = (
        compute_train_state_statistics(
            manifest_path=cfg.dataset.manifest_path,
            sequence_lengths_path=cfg.dataset.sequence_lengths_path,
            states_path=cfg.dataset.states_path,
            velocities_path=cfg.dataset.velocities_path,
            minimum_std=float(cfg.target_normalization.minimum_std),
        )
    )
    target_mean = target_mean_cpu.to(device)
    target_std = target_std_cpu.to(device)
    latent_normalizer = LatentNormalizer.from_file(
        str(cfg.cached_representation.stats_path), device=device
    )
    probe = instantiate(cfg.state_probe).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.epochs),
        eta_min=float(cfg.training.min_lr),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best_validation_mse = float("inf")
    history: list[dict[str, Any]] = []
    early_stopping_patience = int(cfg.training.early_stopping_patience)
    early_stopping_min_delta = float(cfg.training.early_stopping_min_delta)
    epochs_without_improvement = 0

    if bool(cfg.training.resume) and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["representation"] != str(cfg.cached_representation.name):
            raise RuntimeError("Probe checkpoint representation does not match")
        if not torch.equal(checkpoint["target_mean"], target_mean_cpu) or not torch.equal(
            checkpoint["target_std"], target_std_cpu
        ):
            raise RuntimeError("Train-only state normalization statistics changed")
        probe.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_mse = float(checkpoint["best_validation_mse"])
        history = checkpoint["history"]
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"Resumed from epoch {start_epoch}: {last_path}")

    torch.cuda.reset_peak_memory_stats(device)
    epochs = int(cfg.training.epochs)
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        train_loader = build_loader(
            train_dataset,
            cfg,
            batch_size=int(cfg.dataset.trajectories_per_batch),
            shuffle=True,
            epoch=epoch,
        )
        validation_loader = build_loader(
            validation_dataset,
            cfg,
            batch_size=int(cfg.dataset.trajectories_per_batch),
            shuffle=False,
            epoch=0,
        )
        train_metrics = train_epoch(
            probe=probe,
            loader=train_loader,
            latent_normalizer=latent_normalizer,
            target_mean=target_mean,
            target_std=target_std,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            gradient_clip_norm=float(cfg.training.gradient_clip_norm),
            max_batches=max_train_batches,
        )
        validation_metrics = evaluate_probe(
            probe=probe,
            loader=validation_loader,
            latent_normalizer=latent_normalizer,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
            use_amp=use_amp,
            max_batches=max_eval_batches,
        )
        scheduler.step()
        validation_mse = float(validation_metrics["encoded_normalized_mse"])
        improved = validation_mse < (best_validation_mse - early_stopping_min_delta)
        if improved:
            best_validation_mse = validation_mse
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        payload = make_checkpoint(
            probe=probe,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
            best_validation_mse=best_validation_mse,
            history=history,
            target_mean=target_mean,
            target_std=target_std,
            target_stat_count=target_stat_count,
        )
        payload["epochs_without_improvement"] = epochs_without_improvement
        payload["early_stopping_patience"] = early_stopping_patience
        payload["early_stopping_min_delta"] = early_stopping_min_delta
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, best_path)
        (output_dir / cfg.outputs.history_name).write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"representation={cfg.cached_representation.name} "
            f"epoch={epoch + 1:03d}/{epochs} "
            f"train_mse={train_metrics['encoded_normalized_mse']:.7f} "
            f"val_mse={validation_mse:.7f} "
            f"agent_px={validation_metrics['agent_position_l2_rmse_px']:.3f} "
            f"block_px={validation_metrics['block_position_l2_rmse_px']:.3f} "
            f"angle_deg={validation_metrics['block_angle_mae_degrees']:.3f} "
            f"best={best_validation_mse:.7f}",
            flush=True,
        )
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1:03d}: "
                f"no validation improvement >= {early_stopping_min_delta:.2e} "
                f"for {epochs_without_improvement} epochs.",
                flush=True,
            )
            break

    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    probe.load_state_dict(best_checkpoint["model_state_dict"])
    validation_probe = evaluate_probe(
        probe=probe,
        loader=build_loader(
            validation_dataset,
            cfg,
            batch_size=int(cfg.dataset.trajectories_per_batch),
            shuffle=False,
            epoch=0,
        ),
        latent_normalizer=latent_normalizer,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        use_amp=use_amp,
        max_batches=max_eval_batches,
    )
    test_probe = evaluate_probe(
        probe=probe,
        loader=build_loader(
            test_dataset,
            cfg,
            batch_size=int(cfg.dataset.trajectories_per_batch),
            shuffle=False,
            epoch=0,
        ),
        latent_normalizer=latent_normalizer,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        use_amp=use_amp,
        max_batches=max_eval_batches,
    )

    horizons = [int(value) for value in cfg.evaluation.horizons]
    max_horizon = max(horizons)
    dynamics = load_frozen_dynamics(
        cfg.dynamics_checkpoint_path,
        str(cfg.cached_representation.name),
        device,
    )
    rollout_results = {}
    for split in ("validation", "test"):
        rollout_dataset = build_rollout_dataset(cfg, split, max_horizon)
        rollout_results[split] = evaluate_state_rollouts(
            probe=probe,
            dynamics=dynamics,
            loader=build_loader(
                rollout_dataset,
                cfg,
                batch_size=int(cfg.dataset.rollout_trajectories_per_batch),
                shuffle=False,
                epoch=0,
            ),
            latent_normalizer=latent_normalizer,
            target_mean=target_mean,
            target_std=target_std,
            context_length=int(cfg.dataset.context_length),
            horizons=horizons,
            device=device,
            use_amp=use_amp,
            max_batches=max_eval_batches,
        )

    metrics = {
        "status": "completed",
        "representation": str(cfg.cached_representation.name),
        "seed": int(cfg.seed),
        "probe_parameter_count": sum(
            parameter.numel() for parameter in probe.parameters()
        ),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_encoded_normalized_mse": float(
            best_checkpoint["best_validation_mse"]
        ),
        "target_encoding": [
            "agent_x",
            "agent_y",
            "block_x",
            "block_y",
            "sin_block_angle",
            "cos_block_angle",
            "agent_vx",
            "agent_vy",
        ],
        "target_stat_train_frame_count": target_stat_count,
        "validation_probe": validation_probe,
        "test_probe": test_probe,
        "state_rollouts": rollout_results,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "best_checkpoint": str(best_path),
        "dynamics_checkpoint": str(cfg.dynamics_checkpoint_path),
    }
    metrics_path = output_dir / cfg.outputs.metrics_name
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("State probe and multi-step state rollout evaluation: PASSED")


if __name__ == "__main__":
    main()
