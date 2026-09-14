#!/usr/bin/env bash
# Runs the PoC comparing Latent-Space Backdoor vs CSI-Space Baseline on E04
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

echo "=========================================================="
echo "  PoC Experiment Runner"
echo "  Config: ${CONFIG}"
echo "  Python: ${PYTHON_BIN}"
echo "=========================================================="

echo ""
echo "=========================================================="
echo "  Step 1: Training LATENT-SPACE Backdoor (Proposed Method)"
echo "=========================================================="
"${PYTHON_BIN}" train.py --config "${CONFIG}" --mode latent --exp_name poc_latent_ssh

echo ""
echo "=========================================================="
echo "  Step 2: Training CSI-SPACE Baseline (Matched Control)   "
echo "=========================================================="
"${PYTHON_BIN}" train.py --config "${CONFIG}" --mode csi --exp_name poc_csi_ssh

echo ""
echo "PoC Training and Evaluation Complete! Check ./outputs/ or ./outputs_ssh/ for figures and summary CSV."
