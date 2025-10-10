#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_ddp_training_new.sh \
#     --nodes 1 --gpus 8 \
#     --global-batch 48 \
#     --data-dir /path/train --val-dir /path/val \
#     --save-dir checkpoints_phase1 \
#     --epochs 20
#
# Multi-node example (run on each node with appropriate node rank):
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=0 \
#   bash run_ddp_training_new.sh --nodes 2 --gpus 8 --global-batch 64 \
#        --data-dir /path/train --val-dir /path/val
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=1 \
#   bash run_ddp_training_new.sh --nodes 2 --gpus 8 --global-batch 64 \
#        --data-dir /path/train --val-dir /path/val

NODES=1
GPUS=8
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GLOBAL_BATCH=48
EPOCHS=20
DATA_DIR="data/train"
VAL_DIR="data/val"
SAVE_DIR="checkpoints_phase1"
PER_DEVICE_BATCH=""

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
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

WORLD_SIZE=$(( NODES * GPUS ))

# Derive per-device batch if not provided
if [[ -z "${PER_DEVICE_BATCH}" ]]; then
  # Floor division; ensure at least 1
  PER_DEVICE_BATCH=$(( GLOBAL_BATCH / WORLD_SIZE ))
  if [[ ${PER_DEVICE_BATCH} -lt 1 ]]; then
    PER_DEVICE_BATCH=1
  fi
fi

export MASTER_ADDR
export MASTER_PORT
export NODE_RANK
export WORLD_SIZE

# Expose run-time config via env (the Python launcher reads these)
export DATA_DIR
export VAL_DIR
export SAVE_DIR
export EPOCHS
export PER_DEVICE_BATCH

# Optional: improve NCCL stability
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=2
export OMP_NUM_THREADS=4

echo "Launching training_new.py with: nodes=${NODES}, gpus=${GPUS}, world_size=${WORLD_SIZE}, global_batch=${GLOBAL_BATCH}, per_device_batch=${PER_DEVICE_BATCH}" \
     " master=${MASTER_ADDR}:${MASTER_PORT} node_rank=${NODE_RANK}" \
     " data_dir=${DATA_DIR} val_dir=${VAL_DIR} save_dir=${SAVE_DIR} epochs=${EPOCHS}"

# Use torchrun to spawn DDP processes. We invoke python -c so we can override TrainConfig at runtime.
torchrun --nnodes ${NODES} \
         --nproc_per_node ${GPUS} \
         --node_rank ${NODE_RANK} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
  python -c "import os; from training_new import TrainConfig, main; cfg = TrainConfig(\
    data_dir=os.environ.get('DATA_DIR','data/train'), \
    val_dir=os.environ.get('VAL_DIR','data/val'), \
    save_dir=os.environ.get('SAVE_DIR','checkpoints_phase1'), \
    epochs=int(os.environ.get('EPOCHS','20')), \
    batch_size=int(os.environ.get('PER_DEVICE_BATCH','6'))\
  ); main(cfg)"


