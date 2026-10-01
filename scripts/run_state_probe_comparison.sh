#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_ROOT="${REPO_ROOT}/logs/state_probe_comparison_seed42"

cd "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}"

run_probe() {
  local representation="$1"
  shift
  echo "=== ${representation}: probe training and h=1,5,10 rollout evaluation ==="
  "${PYTHON}" scripts/train_state_probe.py \
    "cached_representation=${representation}" \
    "$@" 2>&1 | tee "${LOG_ROOT}/${representation}.log"
}

# Run sequentially so both experiments receive the full GPU and the same conditions.
run_probe vit_ae_s14_earlystop "$@"
run_probe dinov2_vits14_earlystop "$@"

"${PYTHON}" scripts/summarize_state_probe_comparison.py

echo "State-probe comparison completed."
echo "Combined summary: ${DINO_WM_OUTPUT_ROOT:-outputs}/state_probe/vit_ae_vs_dino_earlystop_seed42_summary.json"
