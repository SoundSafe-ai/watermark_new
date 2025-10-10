#!/usr/bin/env bash
set -euo pipefail

# Minimal single-node DDP launcher for training_new.py
# Usage: bash run_ddp_training_new.sh [NUM_GPUS]
# Defaults to 8 GPUs if not provided. Uses config inside training_new.py.

GPUS=${1:-8}

# No repo-dir logic; run this from the repo root (watermark_new)

echo "Launching training_new.py with --standalone DDP on ${GPUS} GPU(s)"

torchrun --standalone --nproc_per_node=${GPUS} training_new.py


