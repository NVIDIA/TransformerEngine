#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -uo pipefail

DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
if [ "${NUM_GPUS}" -lt 4 ]; then
  echo "EP requires >= 4 GPUs (found ${NUM_GPUS}); SKIPPING."
  exit 0
fi
if [ "${NUM_GPUS}" -gt 8 ]; then NUM_GPUS=8; fi

: ${TE_PATH:=/opt/transformerengine}
: ${TEST_TIMEOUT_S:=120}

SCRIPT="${TE_PATH}/examples/pytorch/ep/ep_moe.py"
export PYTHONPATH="${TE_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# Stage JIT cubins on tmpfs for fast iteration.
: ${NCCL_EP_JIT_CACHE_DIR:="${TMPDIR:-/tmp}/nccl_ep_jit_cache_$(id -u)"}
export NCCL_EP_JIT_CACHE_DIR
mkdir -p "$NCCL_EP_JIT_CACHE_DIR"

echo "*** Executing ep_moe.py across ${NUM_GPUS} GPUs (timeout=${TEST_TIMEOUT_S}s) ***"
timeout --foreground --signal=KILL "${TEST_TIMEOUT_S}" \
  torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  "${SCRIPT}" --check 2>&1 | tee stdout_ep_moe.txt
RC=${PIPESTATUS[0]}

RET=0
if [ "${RC}" -ne 0 ]; then RET=1; fi
if grep -qE "(^|]:)FAILED|(^|]:)Traceback" stdout_ep_moe.txt; then RET=1; fi
rm -f stdout_ep_moe.txt
exit $RET
