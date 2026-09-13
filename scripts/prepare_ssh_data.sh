#!/usr/bin/env bash
# Preprocess and cache MM-Fi dataset on the SSH server
set -e

RAW_DIR="/media/jackson/Data/wificsi/MMFI/Compress/"
OUTPUT_CACHE="./data/mmfi_cache.npz"

echo "=== Extracting MM-Fi 4-activity PoC cache from ${RAW_DIR} ==="
mkdir -p ./data/

python -m datasets.prepare_cache \
    --raw_root "${RAW_DIR}" \
    --output "${OUTPUT_CACHE}" \
    --actions A01 A02 A03 A04

echo "Caching finished! Cache stored at ${OUTPUT_CACHE}"
