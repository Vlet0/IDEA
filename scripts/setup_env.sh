#!/usr/bin/env bash
# =============================================================================
# scripts/setup_env.sh
# Robust Environment setup for SSH GPU Server (RTX 3080 Ti / Linux)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo " Setting up Python Environment for WiFi-CSI Latent Backdoor"
echo " Target Hardware: NVIDIA GPU (e.g. RTX 3080 Ti, Ampere sm_86)"
echo " Workspace: ${SCRIPT_DIR}"
echo "======================================================================"

# 1. Determine Environment Strategy: Conda vs UV vs Venv
VENV_DIR="${SCRIPT_DIR}/.venv"
USE_UV=false

if command -v uv &>/dev/null; then
    USE_UV=true
fi

# Check if a Conda environment is currently active
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "[info] Active Conda environment detected: ${CONDA_PREFIX}"
    PYTHON_BIN="python"
    PIP_INSTALL="pip install"
else
    # Virtualenv workflow
    echo "[info] Setting up isolated virtual environment in ${VENV_DIR}..."

    # If .venv exists but python binary is missing, clean it
    if [[ -d "${VENV_DIR}" ]] && [[ ! -f "${VENV_DIR}/bin/python" ]]; then
        echo "[warning] Corrupted .venv detected. Removing..."
        rm -rf "${VENV_DIR}"
    fi

    # Create virtual environment if not present
    if [[ ! -d "${VENV_DIR}" ]]; then
        if [[ "${USE_UV}" == "true" ]]; then
            echo "[info] Creating virtual environment using 'uv'..."
            uv venv --seed "${VENV_DIR}" 2>/dev/null || uv venv "${VENV_DIR}"
        else
            echo "[info] Creating virtual environment using 'python3 -m venv'..."
            if ! python3 -m venv "${VENV_DIR}" 2>/dev/null; then
                echo ""
                echo "[ERROR] 'python3 -m venv' failed! This typically happens on Ubuntu/Debian when 'python3-venv' is missing."
                echo "Please run one of the following solutions on your SSH machine:"
                echo ""
                echo "Option A (Recommended if NO sudo - install standalone 'uv'):"
                echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
                echo "  source \$HOME/.local/bin/env 2>/dev/null || export PATH=\"\$HOME/.local/bin:\$PATH\""
                echo "  bash scripts/setup_env.sh"
                echo ""
                echo "Option B (Recommended if you have sudo):"
                echo "  sudo apt update && sudo apt install -y python3-venv python3-pip"
                echo "  bash scripts/setup_env.sh"
                echo ""
                echo "Option C (Conda environment):"
                echo "  conda create -n wificsi python=3.11 -y"
                echo "  conda activate wificsi"
                echo "  bash scripts/setup_env.sh"
                exit 1
            fi
        fi
    fi

    PYTHON_BIN="${VENV_DIR}/bin/python"

    if [[ "${USE_UV}" == "true" ]]; then
        PIP_INSTALL="uv pip install --python ${PYTHON_BIN}"
    elif [[ -f "${VENV_DIR}/bin/pip" ]]; then
        PIP_INSTALL="${VENV_DIR}/bin/pip install"
    else
        PIP_INSTALL="${PYTHON_BIN} -m pip install"
    fi
fi

echo "[info] Python binary: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"

# Upgrade pip if traditional pip exists
if [[ "${USE_UV}" != "true" ]]; then
    echo "[info] Upgrading pip..."
    ${PIP_INSTALL} --upgrade pip || true
fi

# 2. Install PyTorch with CUDA 12.1 for RTX 3080 Ti (Ampere Architecture)
echo ""
echo "=== Installing PyTorch with CUDA 12.1 support ==="
echo "Targeting RTX 3080 Ti (Compute Capability 8.6)..."
${PIP_INSTALL} torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
echo ""
echo "=== Installing project dependencies from requirements.txt ==="
${PIP_INSTALL} -r requirements.txt

# 4. Verification
echo ""
echo "======================================================================"
echo " Verification: Testing GPU & Core Libraries"
echo "======================================================================"
"${PYTHON_BIN}" - << 'EOF'
import sys
import torch
import scipy
import yaml
import matplotlib
import sklearn

print(f"Python Version    : {sys.version.split()[0]}")
print(f"PyTorch Version   : {torch.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version      : {torch.version.cuda}")
    print(f"Device Count      : {torch.cuda.device_count()}")
    print(f"Device Name (0)   : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Device Memory     : {props.total_memory / (1024**3):.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
else:
    print("[WARNING] CUDA is NOT available! PyTorch will run on CPU.")
EOF

echo ""
echo "======================================================================"
echo " Environment setup complete!"
echo " To activate this environment in your SSH session:"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "   conda activate $(basename "${CONDA_PREFIX}")"
else
    echo "   source .venv/bin/activate"
fi
echo "======================================================================"
