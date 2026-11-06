#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Launcher for deepseek_v3_layer_ep.py on all local GPUs. Extra args go to the script:
#   bash run_deepseek_v3_layer_ep.sh                        # bf16, small dims
#   bash run_deepseek_v3_layer_ep.sh --dsv3 --recipe mxfp8  # DeepSeek-V3 dims, MXFP8 experts
#   NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 bash run_deepseek_v3_layer_ep.sh --dsv3 --recipe mxfp8
#   NSYS=1 bash run_deepseek_v3_layer_ep.sh --dsv3          # nsys report in results/
# Multi-node: run torchrun yourself with --nnodes/--rdzv-endpoint; EP spans all ranks.

set -uo pipefail

DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
if [ "${NUM_GPUS}" -lt 2 ]; then
  echo "EP requires >= 2 GPUs (found ${NUM_GPUS}); SKIPPING."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: ${NCCL_EP_JIT_CACHE_DIR:="${TMPDIR:-/tmp}/nccl_ep_jit_cache_$(id -u)"}
export NCCL_EP_JIT_CACHE_DIR
mkdir -p "$NCCL_EP_JIT_CACHE_DIR"

PREFIX=()
if [ "${NSYS:-0}" = "1" ]; then
  mkdir -p "${SCRIPT_DIR}/results"
  PREFIX=(nsys profile -t cuda,nvtx,nccl -c cudaProfilerApi --capture-range-end=stop
          --cuda-graph-trace=node -o "${SCRIPT_DIR}/results/deepseek_v3_layer_ep_%h")
fi

"${PREFIX[@]}" torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  "${SCRIPT_DIR}/deepseek_v3_layer_ep.py" "$@"
