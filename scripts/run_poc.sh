#!/usr/bin/env bash
# Runs the PoC comparing Latent-Space Backdoor vs CSI-Space Baseline on E04
set -e

CONFIG="configs/base_config.yaml"

echo "=========================================================="
echo "  Step 1: Training LATENT-SPACE Backdoor (Proposed Method)"
echo "=========================================================="
python train.py --config ${CONFIG} --mode latent --exp_name poc_latent_ssh

echo ""
echo "=========================================================="
echo "  Step 2: Training CSI-SPACE Baseline (Matched Control)   "
echo "=========================================================="
python train.py --config ${CONFIG} --mode csi --exp_name poc_csi_ssh

echo ""
echo "PoC Training and Evaluation Complete! Check ./outputs/ for figures and summary CSV."
