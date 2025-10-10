#!/usr/bin/env bash
set -euo pipefail

# Minimal single-node DDP launcher for training_new.py
# Usage: bash run_ddp_training_new.sh [NUM_GPUS]
# Defaults to 8 GPUs if not provided. Uses config inside training_new.py.

GPUS=${1:-8}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) NODES="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --global-batch) GLOBAL_BATCH="$2"; shift 2;;
    --per-device-batch) PER_DEVICE_BATCH="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --data-dir) DATA_DIR="$2"; shift 2;;
    --val-dir) VAL_DIR="$2"; shift 2;;
    --save-dir) SAVE_DIR="$2"; shift 2;;
    --master-addr) MASTER_ADDR="$2"; shift 2;;
    --master-port) MASTER_PORT="$2"; shift 2;;
    --node-rank) NODE_RANK="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "Launching training_new.py with --standalone DDP on ${GPUS} GPU(s)"

torchrun --standalone --nproc_per_node=${GPUS} training_new.py


