#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Launcher for tests/pytorch/distributed/run_deepseek_ep.py. Auto-detects GPU count.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${DETECTED_GPUS}" -lt 2 ]; then
  echo "DeepSeek EP test requires >= 2 GPUs (found ${DETECTED_GPUS}); SKIPPING."
  exit 0
fi

# NCCL EP requires active NVLink P2P among ranks on the node.
if ! nvidia-smi nvlink --status 2>/dev/null | grep -qE 'Link [0-9]+:.*GB/s'; then
  echo "No NVLink between GPUs (PCIe-only fabric); NCCL EP is unsupported here. SKIPPING."
  exit 0
fi

NUM_RANKS="${NVTE_TEST_EP_NUM_RANKS:-${DETECTED_GPUS}}"
if [ "${NUM_RANKS}" -gt 8 ]; then NUM_RANKS=8; fi

TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-180}"

: ${NCCL_EP_JIT_CACHE_DIR:="${TMPDIR:-/tmp}/nccl_ep_jit_cache_$(id -u)"}
export NCCL_EP_JIT_CACHE_DIR
mkdir -p "$NCCL_EP_JIT_CACHE_DIR"

SCRIPT="${SCRIPT_DIR}/run_deepseek_ep.py"
LOG="stdout_deepseek_ep.txt"

echo "=== Running ${SCRIPT} on ${NUM_RANKS} GPUs (timeout=${TEST_TIMEOUT_S}s) ==="
setsid timeout --foreground --kill-after=10 --signal=TERM "${TEST_TIMEOUT_S}" \
  torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_RANKS}" \
  "${SCRIPT}" 2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
pkill -9 -f "tests/pytorch/distributed/run_deepseek_ep.py" 2>/dev/null || true

RET=0
if [ "${RC}" -ne 0 ]; then echo "torchrun exited with ${RC}"; RET=1; fi
if grep -qE "(^|]:)FAILED|(^|]:)Traceback" "${LOG}"; then RET=1; fi
if ! grep -qE "Ran [0-9]+ test|^OK$" "${LOG}"; then
  echo "ERROR: no test summary — likely hang or early crash"
  RET=1
fi
if [ -z "${KEEP_EP_LOGS:-}" ]; then rm -f "${LOG}"; fi

exit $RET
