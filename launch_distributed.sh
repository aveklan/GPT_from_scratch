#!/bin/bash

MASTER_ADDR="192.168.2.100"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=1

if [ -z "$1" ]; then
    echo "Usage: $0 <node_rank>"
    echo "Example: $0 0  (for master node)"
    echo "Example: $0 1  (for worker node)"
    exit 1
fi

NODE_RANK=$1
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# Verify master address is reachable
echo "Checking if master address is reachable: $MASTER_ADDR"
if ! ping -c 1 "$MASTER_ADDR" &> /dev/null; then
    echo "WARNING: Cannot ping $MASTER_ADDR"
fi

export NNODES=$NNODES
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Force Gloo to use the correct interface (disable loopback binding)
export GLOO_SOCKET_IFNAME=enp2s0d1  # Change to your RDMA NIC name if different
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

echo "========================================"
echo "Starting node $NODE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
echo "========================================"

# Kill any existing processes on this port (master node only)
if [ "$NODE_RANK" -eq 0 ]; then
    echo "Cleaning up port $MASTER_PORT..."
    lsof -i :$MASTER_PORT | grep -v COMMAND | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    sleep 2
fi

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_gpt2.py