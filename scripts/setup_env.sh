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

# 1. Determine Environment Strategy: Conda vs Venv vs UV
VENV_DIR="${SCRIPT_DIR}/.venv"

# Check if a Conda environment is currently active
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "[info] Active Conda environment detected: ${CONDA_PREFIX}"
    PYTHON_BIN="python"
    PIP_BIN="pip"
else
    # Virtualenv workflow
    echo "[info] Setting up isolated virtual environment in ${VENV_DIR}..."

    # If .venv exists but is broken, remove it
    if [[ -d "${VENV_DIR}" ]] && [[ ! -f "${VENV_DIR}/bin/python" || ! -f "${VENV_DIR}/bin/pip" ]]; then
        echo "[warning] Incomplete or corrupted .venv detected. Removing..."
        rm -rf "${VENV_DIR}"
    fi

    # Try creating venv if it doesn't exist
    if [[ ! -d "${VENV_DIR}" ]]; then
        if command -v uv &>/dev/null; then
            echo "[info] Creating virtual environment using 'uv'..."
            uv venv "${VENV_DIR}"
        else
            echo "[info] Creating virtual environment using 'python3 -m venv'..."
            # Test if python3-venv works or hits Ubuntu PEP 668 / missing ensurepip
            if ! python3 -m venv "${VENV_DIR}" 2>/dev/null; then
                echo ""
                echo "[ERROR] 'python3 -m venv' failed! This typically happens on Ubuntu/Debian when 'python3-venv' is missing."
                echo "Please run one of the following solutions on your SSH machine:"
                echo ""
                echo "Option A (Recommended if you have sudo):"
                echo "  sudo apt update && sudo apt install -y python3-venv python3-pip"
                echo ""
                echo "Option B (Recommended if NO sudo - install standalone 'uv'):"
                echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
                echo "  source \$HOME/.local/bin/env 2>/dev/null || source \$HOME/.cargo/env 2>/dev/null || export PATH=\"\$HOME/.local/bin:\$PATH\""
                echo "  uv venv .venv"
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
    PIP_BIN="${VENV_DIR}/bin/pip"
fi

echo "[info] Python binary: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"
echo "[info] Upgrading pip..."
"${PIP_BIN}" install --upgrade pip

# 2. Install PyTorch with CUDA 12.1 for RTX 3080 Ti (Ampere Architecture)
echo ""
echo "=== Installing PyTorch with CUDA 12.1 support ==="
echo "PyTorch + CUDA 12.1 is fully optimized for RTX 3080 Ti (Compute Capability 8.6)..."
"${PIP_BIN}" install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
echo ""
echo "=== Installing project dependencies from requirements.txt ==="
"${PIP_BIN}" install -r requirements.txt

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
