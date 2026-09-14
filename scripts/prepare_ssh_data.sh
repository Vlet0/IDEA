#!/usr/bin/env bash
# Preprocess and cache MM-Fi dataset on the SSH server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

RAW_DIR="${1:-/media/jackson/Data/wificsi/MMFI/Compress/}"
OUTPUT_CACHE="${2:-./data/mmfi_cache.npz}"

# Auto-detect Python interpreter
PYTHON_BIN="python"
if [[ -f "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
fi

echo "=== Extracting MM-Fi 4-activity PoC cache from ${RAW_DIR} ==="
echo "Using Python: ${PYTHON_BIN}"
mkdir -p "$(dirname "${OUTPUT_CACHE}")"

"${PYTHON_BIN}" -m datasets.prepare_cache \
    --raw_root "${RAW_DIR}" \
    --output "${OUTPUT_CACHE}" \
    --actions A01 A02 A03 A04

echo "Caching finished! Cache stored at ${OUTPUT_CACHE}"
