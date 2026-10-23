#!/usr/bin/env bash
# Canonical single-entrypoint experiment protocol.  Run from repository root.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/base_config.yaml}"

command -v "$PYTHON_BIN" >/dev/null || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 127; }

"$PYTHON_BIN" -c "import torch, yaml; print('torch', torch.__version__); yaml.safe_load(open('$CONFIG', encoding='utf-8'))"
OUTPUT_DIR="$("$PYTHON_BIN" -c "from utils.config import load_config, resolve_config; print(resolve_config(load_config('$CONFIG'))['training']['output_dir'])")"
"$PYTHON_BIN" verify_pipeline.py
"$PYTHON_BIN" run_experiments.py --config "$CONFIG" --suite main_benchmark
"$PYTHON_BIN" run_experiments.py --config "$CONFIG" --suite leave_one_out
"$PYTHON_BIN" run_experiments.py --config "$CONFIG" --suite seeds
"$PYTHON_BIN" run_experiments.py --config "$CONFIG" --suite sweeps
"$PYTHON_BIN" run_experiments.py --config "$CONFIG" --suite loss_ablation
"$PYTHON_BIN" scripts/build_paper_artifacts.py --outputs "$OUTPUT_DIR" --config "$CONFIG"

echo "Complete. Tables and figures: $OUTPUT_DIR/paper_artifacts"
