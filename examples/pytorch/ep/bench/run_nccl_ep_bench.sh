#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Launcher for the native NCCL EP ``ep_bench`` (baseline for PyTorch comparison).
# Usage:
#   bash run_nccl_ep_bench.sh           # plain run, stdout only
#   bash run_nccl_ep_bench.sh --nsys    # nsys → results/nccl_ep_nsys.nsys-rep

set -uo pipefail

NSYS=0
for a in "$@"; do
  case "$a" in
    --nsys) NSYS=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RESULTS="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS}"

BIN="${TE_REPO_ROOT}/3rdparty/nccl/build/test/nccl_ep/ep_bench"
LIB="${TE_REPO_ROOT}/3rdparty/nccl/build/lib"
[ -x "${BIN}" ] || { echo "ep_bench not built at ${BIN}" >&2; exit 2; }

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NUM_GPUS}" -lt 4 ]; then
  echo "NCCL EP bench requires >=4 GPUs (found ${NUM_GPUS}); SKIPPING."; exit 0
fi
if [ "${NUM_GPUS}" -gt 8 ]; then NUM_GPUS=8; fi

if [ "${NSYS}" -eq 1 ]; then
  ITERS=10
else
  ITERS=50
fi
ARGS=(--algorithm ht --layout em --tokens 2048 --hidden 7168 --top-k 8
      --experts 256 --warmup 5 --iters "${ITERS}")
[ "${NSYS}" -eq 1 ] && ARGS+=(--profile)  # enables NVTX ranges + cudaProfilerStart/Stop

CMD=(/usr/local/mpi/bin/mpirun --allow-run-as-root --oversubscribe -np "${NUM_GPUS}"
     -x LD_LIBRARY_PATH="${LIB}:${LD_LIBRARY_PATH:-}"
     "${BIN}" "${ARGS[@]}")

if [ "${NSYS}" -eq 1 ]; then
  CMD=(nsys profile
       --output "${RESULTS}/nccl_ep_nsys"
       --force-overwrite=true
       --capture-range=cudaProfilerApi
       --capture-range-end=stop
       --trace=cuda,nvtx,osrt
       "${CMD[@]}")
fi

[ "${NSYS}" -eq 1 ] && SUFFIX="_nsys" || SUFFIX=""
LOG="${RESULTS}/stdout_nccl_ep${SUFFIX}.txt"
"${CMD[@]}" 2>&1 | tee "${LOG}"
echo "Done. Log: ${LOG}"
