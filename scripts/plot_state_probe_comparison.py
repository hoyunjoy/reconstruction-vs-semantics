#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_ROOT = Path(os.environ.get("DINO_WM_OUTPUT_ROOT", "outputs"))
SUMMARY = Path(
    os.environ.get(
        "DINO_WM_PROBE_SUMMARY",
        OUTPUT_ROOT / "state_probe/vit_ae_vs_dino_earlystop_seed42_summary.json",
    )
)
OUT = Path(
    os.environ.get(
        "DINO_WM_FIGURE_ROOT", "metrics/state_probe_seed42_figures"
    )
)
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads(SUMMARY.read_text(encoding="utf-8"))

MODEL_KEYS = ["dinov2_vits14", "vit_ae_s14"]
MODEL_LABELS = {"dinov2_vits14": "DINO", "vit_ae_s14": "ViT-AE"}
COLORS = {"dinov2_vits14": "#0072B2", "vit_ae_s14": "#D55E00"}
HORIZONS = [1, 5, 10]
MODES = ["dynamics_probe", "oracle_latent_probe", "copy_latent_probe"]
MODE_LABELS = {"dynamics_probe": "Dynamics", "oracle_latent_probe": "Oracle latent", "copy_latent_probe": "Copy latent"}
MODE_STYLES = {"dynamics_probe": ("-", "o"), "oracle_latent_probe": ("--", "s"), "copy_latent_probe": (":", "^")}
PHYSICAL = [
    ("agent_position_l2_rmse_px", "Agent position RMSE (px)"),
    ("block_position_l2_rmse_px", "Block position RMSE (px)"),
    ("block_angle_mae_degrees", "Block angle MAE (degrees)"),
    ("agent_velocity_l2_rmse", "Agent velocity RMSE"),
]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# 1. Physical information linearly decodable from real cached latents.
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (metric, title) in zip(axes.flat, PHYSICAL):
    vals = [data["models"][m]["real_test_latent_probe"][metric] for m in MODEL_KEYS]
    bars = ax.bar([MODEL_LABELS[m] for m in MODEL_KEYS], vals, color=[COLORS[m] for m in MODEL_KEYS], width=0.62)
    ax.set_title(title)
    ax.set_ylabel("Error (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=3, fontsize=9)
fig.suptitle("Real-latent linear probe (seed 42, n=1600)")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "01_real_latent_probe")

# 2. Physical errors across rollout horizons, including baselines.
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
legend_handles, legend_labels = [], []
for ax, (metric, title) in zip(axes.flat, PHYSICAL):
    for model in MODEL_KEYS:
        for mode in MODES:
            ls, marker = MODE_STYLES[mode]
            vals = [data["models"][model]["test_rollouts"][str(h)][mode][metric] for h in HORIZONS]
            label = f"{MODEL_LABELS[model]} - {MODE_LABELS[mode]}"
            line, = ax.plot(HORIZONS, vals, linestyle=ls, marker=marker, color=COLORS[model], linewidth=2 if mode == "dynamics_probe" else 1.4, alpha=1.0 if mode == "dynamics_probe" else 0.72, label=label)
            if len(legend_handles) < 6:
                legend_handles.append(line); legend_labels.append(label)
    ax.set_title(title)
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Error (lower is better)")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.25)
fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Rollout probe: physical-state prediction")
fig.tight_layout(rect=(0, 0.09, 1, 0.95))
save(fig, "02_rollout_physical_errors")

# 3. Normalized latent MSE across horizons.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, model in zip(axes, MODEL_KEYS):
    for mode in MODES:
        ls, marker = MODE_STYLES[mode]
        vals = [data["models"][model]["test_rollouts"][str(h)][mode]["encoded_normalized_mse"] for h in HORIZONS]
        ax.plot(HORIZONS, vals, linestyle=ls, marker=marker, linewidth=2, color=COLORS[model], alpha=1.0 if mode == "dynamics_probe" else 0.6, label=MODE_LABELS[mode])
    ax.set_title(MODEL_LABELS[model])
    ax.set_xlabel("Rollout horizon")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Normalized latent MSE (lower is better)")
axes[1].legend(frameon=False)
fig.suptitle("Latent rollout error")
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "03_rollout_latent_mse")

# 4. Improvement of learned dynamics over the copy-latent baseline.
metric_short = ["Agent pos.", "Block pos.", "Block angle", "Agent velocity"]
all_gains = []
for model in MODEL_KEYS:
    matrix = []
    for metric, _ in PHYSICAL:
        row = []
        for h in HORIZONS:
            dyn = data["models"][model]["test_rollouts"][str(h)]["dynamics_probe"][metric]
            copy = data["models"][model]["test_rollouts"][str(h)]["copy_latent_probe"][metric]
            row.append(100.0 * (copy - dyn) / copy)
        matrix.append(row)
    all_gains.append(np.asarray(matrix))
limit = max(10.0, max(float(np.abs(x).max()) for x in all_gains))
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
for ax, model, matrix in zip(axes, MODEL_KEYS, all_gains):
    im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_title(MODEL_LABELS[model])
    ax.set_xticks(range(len(HORIZONS)), [str(h) for h in HORIZONS])
    ax.set_yticks(range(len(metric_short)), metric_short)
    ax.set_xlabel("Rollout horizon")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:+.1f}%", ha="center", va="center", color="white" if abs(matrix[i,j]) > 0.45 * limit else "black", fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.82, label="Error reduction vs copy baseline (%)")
fig.suptitle("Does learned dynamics beat copying the current latent?\nPositive values are better")
save(fig, "04_dynamics_vs_copy_gain")

# 5. Compact publication-style summary of dynamics-only rollouts.
summary_metrics = PHYSICAL + [("encoded_normalized_mse", "Normalized latent MSE"), ("pusht_threshold_accuracy", "PushT threshold accuracy")]
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
for ax, (metric, title) in zip(axes.flat, summary_metrics):
    for model in MODEL_KEYS:
        vals = [data["models"][model]["test_rollouts"][str(h)]["dynamics_probe"][metric] for h in HORIZONS]
        ax.plot(HORIZONS, vals, marker="o", linewidth=2.3, color=COLORS[model], label=MODEL_LABELS[model])
    ax.set_title(title)
    ax.set_xlabel("Rollout horizon")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.25)
    ax.set_ylabel("Higher is better" if metric == "pusht_threshold_accuracy" else "Lower is better")
axes[0, 0].legend(frameon=False)
fig.suptitle("Learned-dynamics rollout summary (seed 42, n=800 per horizon)")
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "05_dynamics_rollout_summary")

# 6. Probe validation curves (documents the stopping behavior).
fig, ax = plt.subplots(figsize=(8, 5))
for model, output_name in [("dinov2_vits14", "dinov2_vits14_earlystop_seed42"), ("vit_ae_s14", "vit_ae_s14_earlystop_seed42")]:
    history_path = OUTPUT_ROOT / "state_probe" / output_name / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [int(x["epoch"]) + 1 for x in history]
    vals = [x["validation"]["encoded_normalized_mse"] for x in history]
    ax.plot(epochs, vals, color=COLORS[model], linewidth=1.5, alpha=0.9, label=f"{MODEL_LABELS[model]} (best epoch {int(np.argmin(vals))+1})")
    best = int(np.argmin(vals))
    ax.scatter([epochs[best]], [vals[best]], color=COLORS[model], s=35, zorder=5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation normalized MSE")
ax.set_title("Linear-probe validation curves")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "06_probe_validation_curves")

# Machine-readable long-form table.
fields = ["representation", "source", "horizon", "mode", "metric", "value", "sample_count", "seed"]
rows = []
for model in MODEL_KEYS:
    m = data["models"][model]
    real = m["real_test_latent_probe"]
    for metric, _ in PHYSICAL:
        rows.append([MODEL_LABELS[model], "real_latent", 0, "probe", metric, real[metric], real["sample_count"], m["seed"]])
    rows.append([MODEL_LABELS[model], "real_latent", 0, "probe", "pusht_threshold_accuracy", real["pusht_threshold_accuracy"], real["sample_count"], m["seed"]])
    for h in HORIZONS:
        for mode, vals in m["test_rollouts"][str(h)].items():
            for metric in [x[0] for x in PHYSICAL] + ["encoded_normalized_mse", "pusht_threshold_accuracy"]:
                rows.append([MODEL_LABELS[model], "rollout", h, mode, metric, vals[metric], vals["sample_count"], m["seed"]])
with (OUT / "state_probe_seed42_long.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(fields); w.writerows(rows)

manifest = {
    "source": str(SUMMARY),
    "seed": 42,
    "note": "Single-seed descriptive figures; no error bars or inferential statistics.",
    "figures": [
        "01_real_latent_probe",
        "02_rollout_physical_errors",
        "03_rollout_latent_mse",
        "04_dynamics_vs_copy_gain",
        "05_dynamics_rollout_summary",
        "06_probe_validation_curves",
    ],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote figures and tables to {OUT}")
for p in sorted(OUT.iterdir()):
    print(p.name, p.stat().st_size)
