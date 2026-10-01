# ViT-AE vs DINO state-probe experiment

This patch reuses `scripts/train_state_probe.py`, which already implements the
required protocol:

1. train a representation-specific spatial linear probe on real cached frozen
   latents;
2. freeze the best probe and the previously trained dynamics model;
3. generate autoregressive latent rollouts for horizons 1, 5, and 10;
4. decode the rollout latents into PushT physical state;
5. compare the decoded state with ground truth and with oracle/copy baselines.

The new cached-representation configs intentionally use the representation
names stored inside the early-stopped dynamics checkpoints. This satisfies the
checkpoint-consistency guard in `load_frozen_dynamics` while pointing to the
same DINO and ViT-AE caches and train-only normalization files used previously.

Run both experiments sequentially with:

```bash
cd /path/to/reconstruction-vs-semantics-pusht
bash scripts/run_state_probe_comparison.sh
```

Extra Hydra overrides are forwarded to both runs. For example, a non-resuming
one-batch smoke test can use separate output directories by invoking
`train_state_probe.py` directly.

Full outputs:

- `outputs/state_probe/vit_ae_s14_earlystop_seed42/metrics.json`
- `outputs/state_probe/dinov2_vits14_earlystop_seed42/metrics.json`
- `outputs/state_probe/vit_ae_vs_dino_earlystop_seed42_summary.json`

The reported rollout modes are:

- `dynamics_probe`: learned latent dynamics followed by the frozen probe;
- `oracle_latent_probe`: true future-image latent followed by the same probe;
- `copy_latent_probe`: current latent copied into the future;
- `true_state_copy`: current physical state copied into the future.
