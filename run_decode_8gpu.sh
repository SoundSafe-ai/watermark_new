#!/usr/bin/env bash
set -euo pipefail

# Single-node, 8-GPU DDP launcher for decode/payload training
# Usage:
#   bash scripts/run_decode_8gpu.sh
# Optional overrides:
#   export NPROC_PER_NODE=8
#   export MASTER_PORT=29501
#   export TORCHRUN_BIN=torchrun
#   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MASTER_PORT="${MASTER_PORT:-29501}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

exec "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  watermark_new/train_decode_payload.py


