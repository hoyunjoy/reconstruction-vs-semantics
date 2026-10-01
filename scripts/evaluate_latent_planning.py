#!/usr/bin/env python3
"""Evaluate frozen latent dynamics with matched CEM+MPC in PushT."""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.pusht.pusht_env import pymunk_to_shapely
from env.pusht.pusht_wrapper import PushTWrapper
from metrics.latent_normalization import LatentNormalizer
from models.dino import DinoV2Encoder
from models.vit_autoencoder import ViTAutoencoder
from planning.latent_cem_mpc import cem_plan, goal_state_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_or_verify_tasks(cfg: DictConfig) -> dict:
    path = Path(cfg.tasks.manifest_path)
    source_manifest = json.loads(Path(cfg.dataset.manifest_path).read_text())
    with Path(cfg.dataset.sequence_lengths_path).open("rb") as handle:
        lengths = pickle.load(handle)
    states = torch.load(
        cfg.dataset.states_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    required = int(cfg.dataset.context_length) + int(cfg.tasks.execution_steps)
    minimum_position = float(cfg.tasks.minimum_block_position_change_px)
    minimum_angle = np.deg2rad(float(cfg.tasks.minimum_block_angle_change_degrees))
    candidates = []
    for index in source_manifest["splits"]["test"]:
        source_index = int(index)
        maximum_start = int(lengths[source_index]) - required
        for start in range(maximum_start + 1):
            current = start + int(cfg.dataset.context_length) - 1
            goal = current + int(cfg.tasks.execution_steps)
            block_position_change = float(
                torch.linalg.vector_norm(
                    states[source_index, goal, 2:4]
                    - states[source_index, current, 2:4]
                )
            )
            raw_angle_change = float(
                states[source_index, goal, 4]
                - states[source_index, current, 4]
            )
            angle_change = abs(
                (raw_angle_change + np.pi) % (2.0 * np.pi) - np.pi
            )
            if (
                block_position_change >= minimum_position
                or angle_change >= minimum_angle
            ):
                candidates.append((source_index, start))
    if len(candidates) < int(cfg.tasks.count):
        raise RuntimeError(
            f"Only {len(candidates)} moving-block planning windows satisfy "
            f"the thresholds, fewer than requested {cfg.tasks.count}"
        )
    rng = np.random.default_rng(int(cfg.seed))
    chosen_indices = rng.choice(
        len(candidates), size=int(cfg.tasks.count), replace=False
    ).tolist()
    tasks = [
        {"source_index": candidates[index][0], "start": candidates[index][1]}
        for index in chosen_indices
    ]
    expected = {
        "seed": int(cfg.seed),
        "algorithm": "enumerate moving-block windows; numpy.default_rng(seed); choice without replacement",
        "source_split": "test",
        "context_length": int(cfg.dataset.context_length),
        "execution_steps": int(cfg.tasks.execution_steps),
        "minimum_block_position_change_px": minimum_position,
        "minimum_block_angle_change_degrees": float(
            cfg.tasks.minimum_block_angle_change_degrees
        ),
        "candidate_count": len(candidates),
        "tasks": tasks,
    }
    serialized = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text() != serialized:
            raise RuntimeError(f"Existing planning task manifest differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized)
    return expected


def load_dynamics(path: Path, representation: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["representation"] != representation:
        raise RuntimeError("Dynamics checkpoint representation mismatch")
    model = instantiate(OmegaConf.create(checkpoint["config"]["dynamics"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval().requires_grad_(False)


def load_encoder(cfg: DictConfig, device: torch.device):
    name = str(cfg.cached_representation.name)
    if name.startswith("dinov2_vits14"):
        model = DinoV2Encoder(
            name=str(cfg.encoder.dino_name),
            feature_key=str(cfg.encoder.dino_feature_key),
        )
        encode = model.forward
    elif name.startswith("vit_ae_s14"):
        checkpoint = torch.load(
            cfg.encoder.vit_ae_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        saved = checkpoint["config"]
        model = ViTAutoencoder(
            image_size=int(saved["image_size"]),
            torch_hub_repo=saved.get("torch_hub_repo"),
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        encode = model.encode
    else:
        raise ValueError(f"Unknown representation: {name}")
    model = model.to(device).eval().requires_grad_(False)
    return model, encode


@torch.no_grad()
def encode_image(image: np.ndarray, encode, normalizer, device, use_amp):
    tensor = torch.from_numpy(np.asarray(image)).to(device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    if tuple(tensor.shape[-2:]) != (224, 224):
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    tensor = (tensor - 0.5) / 0.5
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        latent = encode(tensor)
    return normalizer.normalize(latent.float())


def make_env(initial_state: np.ndarray) -> tuple[PushTWrapper, dict, np.ndarray]:
    env = PushTWrapper(with_velocity=True, with_target=True)
    env.shape = "T"
    env.reset_to_state = initial_state.copy()
    observation, state = env.reset()
    return env, observation, state


def block_overlap(env: PushTWrapper, goal_state: np.ndarray) -> float:
    goal_body = env._get_goal_pose_body(goal_state[2:5])
    goal_geometry = pymunk_to_shapely(goal_body, env.block.shapes)
    block_geometry = pymunk_to_shapely(env.block, env.block.shapes)
    return float(goal_geometry.intersection(block_geometry).area / goal_geometry.area)


def denormalize_action(action, mean, std):
    return action * std + mean


def run_fixed_actions(initial_state, goal_state, actions, action_mean, action_std):
    env, _, state = make_env(initial_state)
    overlaps = [block_overlap(env, goal_state)]
    for normalized_action in actions:
        physical = denormalize_action(normalized_action, action_mean, action_std)
        _, _, _, info = env.step(physical)
        state = info["state"]
        overlaps.append(block_overlap(env, goal_state))
    metrics = goal_state_metrics(np.asarray(state), goal_state)
    metrics.update(
        final_block_overlap=overlaps[-1],
        max_block_overlap=max(overlaps),
        block_overlap_success=bool(overlaps[-1] >= 0.95),
    )
    env.close()
    return metrics


@torch.no_grad()
def run_mpc(
    *, cfg, task_id, initial_state, goal_state, latent_context, past_actions,
    goal_latent, dynamics, encode, normalizer, device, action_mean_np, action_std_np,
):
    env, observation, state = make_env(initial_state)
    overlaps = [block_overlap(env, goal_state)]
    planned_actions = []
    warm_mean = None
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + task_id)
    total_steps = int(cfg.tasks.execution_steps)
    for step in range(total_steps):
        horizon = min(int(cfg.cem.horizon), total_steps - step)
        if warm_mean is not None:
            warm_mean = warm_mean[:horizon]
        plan, _ = cem_plan(
            dynamics=dynamics,
            latent_context=latent_context,
            past_actions=past_actions,
            goal_latent=goal_latent,
            horizon=horizon,
            num_samples=int(cfg.cem.num_samples),
            num_elites=int(cfg.cem.num_elites),
            iterations=int(cfg.cem.iterations),
            initial_std=float(cfg.cem.initial_std),
            minimum_std=float(cfg.cem.minimum_std),
            action_clip=float(cfg.cem.action_clip),
            smoothing=float(cfg.cem.smoothing),
            generator=generator,
            use_amp=bool(cfg.amp),
            initial_mean=warm_mean,
        )
        action = plan[0]
        physical = denormalize_action(action.cpu().numpy(), action_mean_np, action_std_np)
        observation, _, _, info = env.step(physical)
        state = info["state"]
        new_latent = encode_image(
            observation["visual"], encode, normalizer, device, bool(cfg.amp)
        )
        latent_context = torch.cat((latent_context[:, 1:], new_latent.unsqueeze(1)), dim=1)
        past_actions = torch.cat((past_actions, action.unsqueeze(0)), dim=0)[-2:]
        planned_actions.append(action.cpu().tolist())
        overlaps.append(block_overlap(env, goal_state))
        warm_mean = plan[1:].clone() if plan.shape[0] > 1 else None
    metrics = goal_state_metrics(np.asarray(state), goal_state)
    metrics.update(
        final_block_overlap=overlaps[-1],
        max_block_overlap=max(overlaps),
        block_overlap_success=bool(overlaps[-1] >= 0.95),
        planned_actions=planned_actions,
    )
    env.close()
    return metrics


def aggregate(results: list[dict]) -> dict:
    numeric = [
        "position_l2_error", "agent_position_l2_error", "block_position_l2_error",
        "block_angle_error_degrees", "final_block_overlap", "max_block_overlap",
    ]
    output = {f"mean_{key}": float(np.mean([r[key] for r in results])) for key in numeric}
    output["goal_state_success_rate"] = float(np.mean([r["goal_state_success"] for r in results]))
    output["block_overlap_success_rate"] = float(np.mean([r["block_overlap_success"] for r in results]))
    output["task_count"] = len(results)
    return output


@hydra.main(version_base=None, config_path="../conf", config_name="experiment/latent_planning")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Planning requires CUDA")
    seed_everything(int(cfg.seed))
    device = torch.device("cuda")
    tasks_manifest = create_or_verify_tasks(cfg)
    tasks = tasks_manifest["tasks"]
    if cfg.tasks.max_tasks is not None:
        tasks = tasks[: int(cfg.tasks.max_tasks)]
    with Path(cfg.dataset.sequence_lengths_path).open("rb") as handle:
        lengths = pickle.load(handle)
    states = torch.load(cfg.dataset.states_path, map_location="cpu", weights_only=True, mmap=True)
    velocities = torch.load(cfg.dataset.velocities_path, map_location="cpu", weights_only=True, mmap=True)
    raw_actions = torch.load(cfg.dataset.actions_path, map_location="cpu", weights_only=True, mmap=True).float().div(float(cfg.dataset.action_scale))
    action_mean_np = np.asarray(cfg.dataset.action_mean, dtype=np.float32)
    action_std_np = np.asarray(cfg.dataset.action_std, dtype=np.float32)
    action_mean = torch.tensor(action_mean_np, device=device)
    action_std = torch.tensor(action_std_np, device=device)
    dynamics = load_dynamics(Path(cfg.dynamics_checkpoint_path), str(cfg.cached_representation.name), device)
    encoder, encode = load_encoder(cfg, device)
    normalizer = LatentNormalizer.from_file(str(cfg.cached_representation.stats_path), device=device)
    results = {"cem_mpc": [], "zero_action": [], "dataset_action_replay": []}
    torch.cuda.reset_peak_memory_stats(device)

    for task_id, task in enumerate(tasks):
        source = int(task["source_index"])
        start = int(task["start"])
        current = start + int(cfg.dataset.context_length) - 1
        goal_index = current + int(cfg.tasks.execution_steps)
        if goal_index >= int(lengths[source]):
            raise RuntimeError("Planning task exceeds trajectory length")
        raw_state = torch.cat((states[source].float(), velocities[source].float()), dim=-1)
        initial_state = raw_state[current].numpy()
        goal_state = raw_state[goal_index].numpy()
        payload = torch.load(
            Path(cfg.cached_representation.cache_root) / "test" / f"episode_{source:05d}.pt",
            map_location="cpu", weights_only=True, mmap=True,
        )
        cached = payload["features"].float().to(device)
        latent_context = normalizer.normalize(cached[start : current + 1]).unsqueeze(0)
        goal_latent = normalizer.normalize(cached[goal_index]).unsqueeze(0)
        normalized_actions = (raw_actions[source].to(device) - action_mean) / action_std
        past_actions = normalized_actions[start:current]
        expert_actions = normalized_actions[current:goal_index].cpu().numpy()
        zero_physical = np.zeros((int(cfg.tasks.execution_steps), 2), dtype=np.float32)
        zero_normalized = (zero_physical - action_mean_np) / action_std_np
        results["zero_action"].append(run_fixed_actions(initial_state, goal_state, zero_normalized, action_mean_np, action_std_np))
        results["dataset_action_replay"].append(run_fixed_actions(initial_state, goal_state, expert_actions, action_mean_np, action_std_np))
        results["cem_mpc"].append(run_mpc(
            cfg=cfg, task_id=task_id, initial_state=initial_state, goal_state=goal_state,
            latent_context=latent_context, past_actions=past_actions, goal_latent=goal_latent,
            dynamics=dynamics, encode=encode, normalizer=normalizer, device=device,
            action_mean_np=action_mean_np, action_std_np=action_std_np,
        ))
        print(f"task {task_id + 1}/{len(tasks)} success={results['cem_mpc'][-1]['goal_state_success']} overlap={results['cem_mpc'][-1]['final_block_overlap']:.3f}", flush=True)

    output = {
        "status": "completed", "representation": str(cfg.cached_representation.name),
        "seed": int(cfg.seed), "task_manifest": str(cfg.tasks.manifest_path),
        "task_count": len(tasks), "summary": {key: aggregate(value) for key, value in results.items()},
        "per_task": results, "gpu_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    output_dir = Path(cfg.outputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / cfg.outputs.metrics_name
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"metrics: {path}")
    print("Latent CEM+MPC planning evaluation: PASSED")


if __name__ == "__main__":
    main()
