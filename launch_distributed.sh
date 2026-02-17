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
export PRINT_COLLECTIVE_TRAFFIC=${PRINT_COLLECTIVE_TRAFFIC:-1}

# Force Gloo to use the correct interface (disable loopback binding)
# export GLOO_SOCKET_IFNAME=enp2s0  # Change to your RDMA NIC name if different
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp2s0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp2s0}
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# Resolve HCA+port for the selected socket interface (e.g., rocep3s0:1 for enp2s0)
detected_hca_port=""
if command -v rdma >/dev/null 2>&1; then
    detected_hca_port=$(rdma link show | awk -v iface="$NCCL_SOCKET_IFNAME" '$0 ~ ("netdev " iface "$") {split($2,a,"/"); print a[1] ":" a[2]; exit}')
fi

# If unset, or set without a port, prefer/normalize to the detected HCA:port
if [ -z "${NCCL_IB_HCA:-}" ] || [[ "$NCCL_IB_HCA" != *:* ]]; then
    if [ -n "$detected_hca_port" ]; then
        export NCCL_IB_HCA="$detected_hca_port"
    elif [ -d /sys/class/infiniband ] && [ -n "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
        export NCCL_IB_HCA="$(ls /sys/class/infiniband | head -n 1):1"
    else
        echo "WARNING: No RDMA HCA found under /sys/class/infiniband"
    fi
fi

export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-1}  # ConnectX-3 RoCE IPv4-mapped GID
export NCCL_NET_GDR_READ=1

echo "========================================"
echo "Starting node $NODE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "NCCL_IB_HCA: $NCCL_IB_HCA"
echo "PRINT_COLLECTIVE_TRAFFIC: $PRINT_COLLECTIVE_TRAFFIC"
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