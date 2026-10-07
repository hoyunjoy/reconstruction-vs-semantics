# Does Reconstruction Ensure Physical Semantics?

This repository contains a controlled PushT study of whether a representation
that reconstructs pixels well also preserves the physical information needed
for latent dynamics and model-predictive control. It compares:

- **ViT-AE**: a ViT-S/14 encoder trained from scratch on PushT reconstruction,
  paired with the project's convolutional decoder; and
- **DINOv2-S/14**: a frozen, pretrained DINOv2 encoder using normalized patch
  tokens (`x_norm_patchtokens`).

Both representations have the same latent geometry: 256 patch tokens with 384
channels. They use the same train/validation/test trajectories, train-only
normalization protocol, action-conditioned dynamics architecture, linear-probe
capacity, CEM+MPC planner, and seed (42). The comparison does **not** isolate
the training objective alone: DINOv2 is externally pretrained, whereas ViT-AE
is trained from scratch on PushT.

This code is derived from the official
[DINO-WM repository](https://github.com/gaoyuezhou/dino_wm) and retains its MIT
license. Please cite DINO-WM as described in [Acknowledgements](#acknowledgements).

## Main result

ViT-AE reconstructs test images much more accurately, but DINOv2 retains more
linearly accessible physical state, produces more stable long-horizon latent
rollouts, and plans better under matched downstream conditions.

| Metric (seed 42) | DINOv2 | ViT-AE |
|---|---:|---:|
| Reconstruction PSNR (dB) | 38.024 | **45.360** |
| Reconstruction SSIM | 0.9891 | **0.9969** |
| Reconstruction LPIPS-VGG | 0.01880 | **0.00756** |
| Dynamics validation normalized MSE | **0.02057** | 0.03127 |
| Goal-pose success, 100 planning tasks | **92%** | 40% |
| Final block overlap >= 0.95 | 32% | 27% |

Goal-pose success requires the joint L2 error over agent and block positions
(`agent_x`, `agent_y`, `block_x`, `block_y`) to be below 20 pixels **and** the
circular block-angle error to be below 20 degrees. The overlap metric is
reported separately and requires final block-goal overlap of at least 0.95.
The 100-task paired planning statistics are stored in
[`planning_n100_stats.json`](planning_n100_stats.json).

## Repository layout

```text
artifacts/                  Fixed split and planning-task manifests
conf/                       Hydra configurations
datasets/                   PushT and cached-latent datasets
env/pusht/                  PushT simulator
metrics/                    Reconstruction, normalization, and probe metrics
models/                     ViT-AE, dynamics, reconstruction decoder, probes
planning/                   CEM and latent receding-horizon MPC
scripts/                    Training, caching, evaluation, and plotting CLIs
tests/                      Unit and shape tests
```

Datasets, feature caches, checkpoints, logs, and generated outputs are excluded
from Git by `.gitignore`.

## Installation

Python 3.9 and a CUDA-capable PyTorch installation were used for the reported
experiments.

```bash
git clone <repository-url>
cd reconstruction-vs-semantics-pusht

# Reproduce the full original environment:
conda env create -f environment.yaml
conda activate dino_wm

# Or install the smaller PushT experiment dependency set:
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your CUDA version requires a platform-specific PyTorch wheel, install that
wheel first and then install the remaining requirements. The first DINOv2 or
ViT-AE run downloads the official `facebookresearch/dinov2` Torch Hub code and,
for DINOv2, pretrained weights. An offline run can pass a local DINOv2 checkout
to `--torch-hub-repo` for ViT-AE training and use a populated Torch Hub cache.

## Paths and data preparation

Run commands from the repository root. Defaults are repository-relative. The
following optional environment variables relocate large artifacts without code
changes:

```bash
export DINO_WM_DATA_ROOT=/path/to/data
export DINO_WM_CACHE_ROOT=/path/to/feature_cache
export DINO_WM_CHECKPOINT_ROOT=/path/to/checkpoints
export DINO_WM_OUTPUT_ROOT=/path/to/outputs
```

Download the PushT data linked by the
[DINO-WM project](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28)
and arrange it as follows:

```text
$DINO_WM_DATA_ROOT/
└── pusht_noise/
    ├── train/
    │   ├── obses/
    │   ├── rel_actions.pth
    │   ├── seq_lengths.pkl
    │   ├── states.pth
    │   └── velocities.pth
    └── ...
```

The fixed seed-42 split manifest is committed at
`artifacts/splits/pusht_1000_seed42.json`. To regenerate and validate it:

```bash
python scripts/create_pusht_split.py \
  --data-root "$DINO_WM_DATA_ROOT/pusht_noise" \
  --output artifacts/splits/pusht_1000_seed42.json \
  --seed 42
```

## End-to-end reproduction

### 1. Train the reconstruction-based ViT-AE

The encoder uses the exact DINOv2 ViT-S/14 architecture with random weights;
no DINOv2 pretrained parameters are loaded. The best checkpoint is selected by
validation reconstruction MSE.

```bash
python scripts/train_vit_autoencoder.py \
  --data-root "$DINO_WM_DATA_ROOT/pusht_noise" \
  --manifest artifacts/splits/pusht_1000_seed42.json \
  --checkpoint-dir "$DINO_WM_CHECKPOINT_ROOT/vit_ae_s14_seed42_v1" \
  --output-dir "$DINO_WM_OUTPUT_ROOT/vit_ae_s14_seed42_v1" \
  --max-epochs 100 \
  --early-stopping-patience 10 \
  --early-stopping-min-relative-delta 0.001
```

### 2. Cache frozen latent representations

```bash
# ViT-AE: always cache from the best validation checkpoint.
python scripts/cache_vit_ae_features.py \
  --checkpoint "$DINO_WM_CHECKPOINT_ROOT/vit_ae_s14_seed42_v1/best.pt" \
  --data-root "$DINO_WM_DATA_ROOT/pusht_noise" \
  --manifest artifacts/splits/pusht_1000_seed42.json \
  --output-root "$DINO_WM_CACHE_ROOT/vit_ae_s14_seed42_v1" \
  --resume

# Frozen pretrained DINOv2-S/14.
python scripts/cache_dino_manifest_features.py
```

Each cache stores a `[time, 256, 384]` tensor per trajectory. Encoders remain
frozen in all downstream experiments.

### 3. Compute train-only normalization statistics

```bash
python scripts/compute_latent_stats.py \
  --config-name=experiment/latent_stats_vit_dino
```

Statistics are computed only from the 800 training trajectories. Cached raw
latents are not modified; normalization is applied when they are loaded.

### 4. Train matched action-conditioned dynamics

The model consumes three past latent frames and 2-D actions. Both
representations use the same 8,282,880-parameter architecture. The committed
configuration uses a 100-epoch cap and validation early stopping (patience 15).

```bash
python scripts/train_latent_dynamics.py \
  cached_representation=vit_ae_s14_earlystop

python scripts/train_latent_dynamics.py \
  cached_representation=dinov2_vits14_earlystop
```

### 5. Train matched reconstruction decoders and evaluate reconstruction

Both frozen encoders are evaluated with the same decoder architecture, data
split, loss, and image preprocessing.

```bash
python scripts/train_reconstruction_probe.py \
  cached_representation=vit_ae_s14_earlystop

python scripts/train_reconstruction_probe.py \
  cached_representation=dinov2_vits14_earlystop
```

The resulting `metrics.json` files report PSNR, SSIM, and LPIPS-VGG; the output
directory also contains a reconstruction grid.

### 6. Train linear probes and evaluate dynamics rollouts

The probe is trained on real frozen latents to predict agent position, block
position, block angle as sine/cosine, and agent velocity. The best frozen probe
then evaluates real latents and autoregressive dynamics rollouts at horizons 1,
5, and 10, with oracle-latent and copy baselines.

```bash
bash scripts/run_state_probe_comparison.sh
python scripts/plot_state_probe_comparison.py
```

To redirect the generated comparison figures:

```bash
DINO_WM_FIGURE_ROOT=/path/to/figures \
  python scripts/plot_state_probe_comparison.py
```

### 7. Run CEM + receding-horizon MPC planning

The planner samples 10-step action sequences, refits a Gaussian distribution
to elite samples, executes the first action, observes the new image, and
replans. The following commands use the same 100-task manifest and seed for
both representations:

```bash
python scripts/evaluate_latent_planning.py \
  cached_representation=vit_ae_s14_earlystop \
  tasks.manifest_path=artifacts/planning/pusht_test_h10_moving_seed42_n100.json \
  tasks.count=100 tasks.max_tasks=100 \
  outputs.output_dir="$DINO_WM_OUTPUT_ROOT/planning/vit_ae_s14_earlystop_seed42_n100"

python scripts/evaluate_latent_planning.py \
  cached_representation=dinov2_vits14_earlystop \
  tasks.manifest_path=artifacts/planning/pusht_test_h10_moving_seed42_n100.json \
  tasks.count=100 tasks.max_tasks=100 \
  outputs.output_dir="$DINO_WM_OUTPUT_ROOT/planning/dinov2_vits14_earlystop_seed42_n100"
```

## Validation

Fast checks that do not require the dataset or checkpoints:

```bash
python -m compileall -q datasets metrics models planning scripts
python -m pytest -q
python scripts/train_vit_autoencoder.py --help
python scripts/cache_vit_ae_features.py --help
python scripts/create_pusht_split.py --help
```

GPU/data-dependent training and evaluation require the external artifacts
described above.

## Scope and limitations

- All reported results use a single seed (42) and the PushT environment.
- Only one reconstruction-trained representation and one non-reconstruction
  pretrained representation are compared.
- External DINOv2 pretraining remains a confound, so the results are an
  operational comparison rather than a causal estimate of objective alone.
- Goal-pose and strict-overlap success depend on explicit thresholds; continuous
  position, angle, and overlap metrics should be considered alongside them.

## Acknowledgements

The base world-model and PushT code comes from DINO-WM:

```bibtex
@misc{zhou2024dinowmworldmodelspretrained,
  title={DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning},
  author={Gaoyue Zhou and Hengkai Pan and Yann LeCun and Lerrel Pinto},
  year={2024},
  eprint={2411.04983},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2411.04983}
}
```
