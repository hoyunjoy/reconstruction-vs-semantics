#!/usr/bin/env python3
"""Collect the matched ViT-AE/DINO state-probe and rollout metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PATHS = {
    "vit_ae_s14": Path(
        "outputs/state_probe/vit_ae_s14_earlystop_seed42/metrics.json"
    ),
    "dinov2_vits14": Path(
        "outputs/state_probe/dinov2_vits14_earlystop_seed42/metrics.json"
    ),
}

SCALAR_KEYS = (
    "encoded_normalized_mse",
    "agent_position_l2_rmse_px",
    "block_position_l2_rmse_px",
    "block_angle_mae_degrees",
    "agent_velocity_l2_rmse",
    "pusht_threshold_accuracy",
    "sample_count",
)


def select_metrics(values: dict[str, Any]) -> dict[str, Any]:
    return {key: values[key] for key in SCALAR_KEYS if key in values}


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"Incomplete state-probe result: {path}")
    return payload


def compact_result(payload: dict[str, Any]) -> dict[str, Any]:
    rollouts = payload["state_rollouts"]["test"]
    return {
        "representation": payload["representation"],
        "seed": payload["seed"],
        "probe_parameter_count": payload["probe_parameter_count"],
        "best_epoch": payload["best_epoch"],
        "best_checkpoint": payload["best_checkpoint"],
        "dynamics_checkpoint": payload["dynamics_checkpoint"],
        "real_test_latent_probe": select_metrics(payload["test_probe"]),
        "test_rollouts": {
            horizon: {
                mode: select_metrics(values)
                for mode, values in modes.items()
            }
            for horizon, modes in rollouts.items()
            if int(horizon) in (1, 5, 10)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vit-metrics", type=Path, default=DEFAULT_PATHS["vit_ae_s14"]
    )
    parser.add_argument(
        "--dino-metrics", type=Path, default=DEFAULT_PATHS["dinov2_vits14"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/state_probe/"
            "vit_ae_vs_dino_earlystop_seed42_summary.json"
        ),
    )
    args = parser.parse_args()

    report = {
        "protocol": {
            "probe_training_input": "real cached frozen latents",
            "probe_target": [
                "agent_x",
                "agent_y",
                "block_x",
                "block_y",
                "sin_block_angle",
                "cos_block_angle",
                "agent_vx",
                "agent_vy",
            ],
            "rollout_horizons": [1, 5, 10],
            "rollout_modes": [
                "dynamics_probe",
                "oracle_latent_probe",
                "copy_latent_probe",
                "true_state_copy",
            ],
        },
        "models": {
            "vit_ae_s14": compact_result(load_metrics(args.vit_metrics)),
            "dinov2_vits14": compact_result(load_metrics(args.dino_metrics)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved comparison summary to {args.output}")


if __name__ == "__main__":
    main()
