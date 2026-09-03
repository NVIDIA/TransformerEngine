#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Launcher for tests/pytorch/distributed/run_models.py (model-specific layers). Auto-detects GPU count.

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

SCRIPT="${SCRIPT_DIR}/run_models.py"

echo "=== Running ${SCRIPT} on ${NUM_RANKS} GPUs (timeout=${TEST_TIMEOUT_S}s) ==="
setsid timeout --foreground --kill-after=10 --signal=TERM "${TEST_TIMEOUT_S}" \
  torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_RANKS}" "${SCRIPT}"
RC=$?
pkill -9 -f "tests/pytorch/distributed/run_models.py" 2>/dev/null || true
if [ "${RC}" -ne 0 ]; then echo "torchrun exited with ${RC}"; fi
exit $RC
