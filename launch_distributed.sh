#!/bin/bash
set -euo pipefail

# Override these through the environment when running on another cluster.
MASTER_ADDR=${MASTER_ADDR:-192.168.2.1}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

if [[ $# -lt 1 || ! $1 =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <node_rank> [train_gpt2.py options]"
    exit 1
fi
NODE_RANK=$1
shift
if (( NODE_RANK >= NNODES )); then
    echo "node_rank must be smaller than NNODES"
    exit 1
fi

export NNODES MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp2s0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp2s0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${PYTHON:-} ]]; then
    if [[ -x "$SCRIPT_DIR/gpt/bin/python" ]]; then
        PYTHON="$SCRIPT_DIR/gpt/bin/python"
    else
        PYTHON=python3
    fi
fi
cd "$SCRIPT_DIR"
echo "Starting TP node $NODE_RANK/$NNODES, $NPROC_PER_NODE GPU(s) per node"
echo "Master: $MASTER_ADDR:$MASTER_PORT; NCCL interface: $NCCL_SOCKET_IFNAME"

exec "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train_gpt2.py "$@"
