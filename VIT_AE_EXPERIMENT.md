# ViT-AE-S/14 controlled reconstruction experiment

This patch adds a scratch-trained reconstruction encoder whose backbone is the
exact DINOv2 ViT-S/14 implementation.  It deliberately does **not** load DINO
weights. Its decoder is the same 2,611,543-parameter convolutional decoder
used in the matched reconstruction comparison.

## Safety properties

- Existing DINOv2 checkpoints, outputs, and caches are not modified.
- The default run writes only to new `vit_ae_s14_seed42_v1` directories.
- Training refuses to start if either destination directory already exists.
- Encoder and decoder parameter counts are saved separately.
- The best checkpoint is selected only with validation reconstruction MSE.
- Early stopping requires ten epochs without a 0.1% relative validation-MSE
  improvement, with a 100-epoch safety cap.

## Files

- `models/vit_autoencoder.py`
- `scripts/train_vit_autoencoder.py`
- `tests/test_vit_autoencoder.py`

## Pre-flight checks

```bash
conda activate dino_wm
cd /path/to/reconstruction-vs-semantics-pusht

python -m pytest -q tests/test_vit_autoencoder.py
python -m py_compile models/vit_autoencoder.py scripts/train_vit_autoencoder.py
```

Check the architecture without starting training:

```bash
python - <<'PY'
from models.vit_autoencoder import ViTAutoencoder

model = ViTAutoencoder()
print(model.parameter_counts())
assert model.parameter_counts()["decoder"] == 2_611_543
assert 20_000_000 <= model.parameter_counts()["encoder"] <= 23_000_000
PY
```

## Full run

Do not start this command until the GPU, mount, manifest, and destination paths
have been verified.

```bash
conda activate dino_wm
cd /path/to/reconstruction-vs-semantics-pusht

nohup python scripts/train_vit_autoencoder.py \
  --max-epochs 100 \
  --early-stopping-patience 10 \
  --early-stopping-min-relative-delta 0.001 \
  > outputs/vit_ae_s14_train_v1.log 2>&1 &
```

Monitor without modifying the run:

```bash
tail -f outputs/vit_ae_s14_train_v1.log
```

After training, do not create a latent cache until the history confirms a real
validation plateau and `best.pt` loads successfully.
