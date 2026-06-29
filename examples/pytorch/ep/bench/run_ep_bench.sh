#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Launcher for examples/pytorch/ep/bench/ep_bench.py.
# Examples:
#   bash run_ep_bench.sh                  # plain run, stdout only
#   bash run_ep_bench.sh --cuda-graph     # capture + replay each stage as a CUDA graph
#   bash run_ep_bench.sh --kineto         # Chrome trace + per-kernel CSV (rank 0)
#   bash run_ep_bench.sh --nsys           # nsys profile on rank 0 -> results/pyt_nsys.nsys-rep

set -uo pipefail

NSYS=0; KINETO=0; CGRAPH=0
for a in "$@"; do
  case "$a" in
    --nsys)        NSYS=1 ;;
    --kineto)      KINETO=1 ;;
    --cuda-graph)  CGRAPH=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
if [ "${NSYS}" -eq 1 ] && [ "${KINETO}" -eq 1 ]; then
  echo "--nsys and --kineto both attach CUPTI; pick one." >&2; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RESULTS="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS}"
export PYTHONPATH="${TE_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
if [ "${NUM_GPUS}" -lt 4 ]; then
  echo "EP bench requires >=4 GPUs (found ${NUM_GPUS}); SKIPPING."; exit 0
fi
if [ "${NUM_GPUS}" -gt 8 ]; then NUM_GPUS=8; fi

: "${TIMEOUT_S:=1800}"
: "${NCCL_EP_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/nccl_ep_jit_cache_$(id -u)}"
export NCCL_EP_JIT_CACHE_DIR
mkdir -p "${NCCL_EP_JIT_CACHE_DIR}"

EXTRA_ARGS=()
TAG="pyt"
[ "${CGRAPH}" -eq 1 ] && EXTRA_ARGS+=(--cuda-graph) && TAG="${TAG}_cg"
if [ "${KINETO}" -eq 1 ]; then
  EXTRA_ARGS+=(--kineto "${RESULTS}/kineto_${TAG}")
fi

EP_BENCH_EXTRA_FLAGS="${EP_BENCH_EXTRA_FLAGS:-}"
LAUNCH=(torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}"
        "${SCRIPT_DIR}/ep_bench.py" "${EXTRA_ARGS[@]}" ${EP_BENCH_EXTRA_FLAGS})

if [ "${NSYS}" -eq 1 ]; then
  NSYS_CMD=(nsys profile
            --output "${RESULTS}/pyt_${TAG}_nsys"
            --force-overwrite=true
            --trace=cuda,nvtx
            --gpu-metrics-devices=none
            --cuda-um-cpu-page-faults=false
            --cuda-um-gpu-page-faults=false)
  echo "[run_ep_bench] launching with nsys (results/${TAG}_nsys.nsys-rep)"
  timeout --foreground --signal=TERM "${TIMEOUT_S}" "${NSYS_CMD[@]}" "${LAUNCH[@]}"
  RC=$?
else
  timeout --foreground --signal=TERM "${TIMEOUT_S}" "${LAUNCH[@]}"
  RC=$?
fi
exit $RC
