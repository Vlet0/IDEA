#!/usr/bin/env bash
# Runs the full suite of research experiments
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

CONFIG="${1:-configs/ssh_remote.yaml}"
if [[ ! -f "${CONFIG}" ]]; then
    CONFIG="configs/base_config.yaml"
fi

# Auto-detect Python interpreter
PYTHON_BIN="python"
if [[ -f "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
fi

echo "=== Running Full Research Experiments on SSH ==="
echo "Config: ${CONFIG}"
echo "Python: ${PYTHON_BIN}"
echo "1. PoC Latent vs CSI Baseline"
echo "2. Leave-one-room-out (Cross-Environment E01..E04)"
echo "3. Seed Stability (5 independent random seeds)"
echo "4. Hyperparameter Sweeps (Poison rate, Epsilon, Gate)"

"${PYTHON_BIN}" run_experiments.py --config "${CONFIG}" --suite all

echo "All experiments completed! Results logged in ./outputs/experiments/"
