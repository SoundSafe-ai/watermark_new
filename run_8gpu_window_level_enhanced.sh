#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_8gpu_window_level_enhanced.sh \
#     --nodes 1 --gpus 8 \
#     --global-batch 64 \
#     --data-dir /path/train --val-dir /path/val \
#     [--epochs 30 --save-dir window_level_enhanced_checkpoints]

NODES=1
GPUS=8
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GLOBAL_BATCH=8
EPOCHS=30
DATA_DIR="data/train"
VAL_DIR="data/val"
SAVE_DIR="window_level_checkpoints"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) NODES="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --global-batch) GLOBAL_BATCH="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --data-dir) DATA_DIR="$2"; shift 2;;
    --val-dir) VAL_DIR="$2"; shift 2;;
    --save-dir) SAVE_DIR="$2"; shift 2;;
    --master-addr) MASTER_ADDR="$2"; shift 2;;
    --master-port) MASTER_PORT="$2"; shift 2;;
    --node-rank) NODE_RANK="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

export MASTER_ADDR
export MASTER_PORT
export NODE_RANK
export WORLD_SIZE=$(( NODES * GPUS ))
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH}

# Optional: improve NCCL stability in some environments
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=2
export OMP_NUM_THREADS=4

echo "Launching training with: nodes=${NODES}, gpus=${GPUS}, world_size=${WORLD_SIZE}, global_batch=${GLOBAL_BATCH}" \
     " master=${MASTER_ADDR}:${MASTER_PORT} node_rank=${NODE_RANK}"

torchrun --nnodes ${NODES} \
         --nproc_per_node ${GPUS} \
         --node_rank ${NODE_RANK} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
  watermark_new/train_window_level.py \
    --data_dir ${DATA_DIR} \
    --val_dir ${VAL_DIR} \
    --epochs ${EPOCHS} \
    --save_dir ${SAVE_DIR}


