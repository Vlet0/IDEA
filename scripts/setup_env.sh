#!/usr/bin/env bash
# Environment setup for SSH server
set -e

echo "=== Setting up Python environment for WiFi-CSI Latent Backdoor ==="
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "=== Verification ==="
python -c "import torch, scipy, yaml, matplotlib; print('PyTorch CUDA available:', torch.cuda.is_available())"
echo "Environment setup complete!"
