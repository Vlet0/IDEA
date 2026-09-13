#!/usr/bin/env bash
# Runs the full suite of research experiments
set -e

CONFIG="configs/base_config.yaml"

echo "=== Running Full Research Experiments on SSH ==="
echo "1. PoC Latent vs CSI Baseline"
echo "2. Leave-one-room-out (Cross-Environment E01..E04)"
echo "3. Seed Stability (5 independent random seeds)"
echo "4. Hyperparameter Sweeps (Poison rate, Epsilon, Gate)"

python run_experiments.py --config ${CONFIG} --suite all

echo "All experiments completed! Results logged in ./outputs/experiments/"
