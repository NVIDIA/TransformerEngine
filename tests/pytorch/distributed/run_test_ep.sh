#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Launcher for tests/pytorch/distributed/run_ep.py. Auto-detects GPU count.
# Short timeout by default to surface hangs early.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${TE_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${DETECTED_GPUS}" -lt 4 ]; then
  echo "EP requires >= 4 GPUs (found ${DETECTED_GPUS}); SKIPPING."
  exit 0
fi

# NCCL EP requires active NVLink P2P among ranks on the node.
# On PCIe-only nodes (no NVLink) it falls back to the network
# transport and deadlocks, so skip cleanly there.
if ! nvidia-smi nvlink --status 2>/dev/null | grep -qE 'Link [0-9]+:.*GB/s'; then
  echo "No NVLink between GPUs (PCIe-only fabric); NCCL EP is unsupported here. SKIPPING."
  exit 0
fi

NUM_RANKS="${NVTE_TEST_EP_NUM_RANKS:-${DETECTED_GPUS}}"
if [ "${NUM_RANKS}" -gt 8 ]; then NUM_RANKS=8; fi

# Short timeout to detect hangs early.
TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-120}"

# Stage NCCL EP JIT cubins on tmpfs to keep iteration fast.
: ${NCCL_EP_JIT_CACHE_DIR:="${TMPDIR:-/tmp}/nccl_ep_jit_cache_$(id -u)"}
export NCCL_EP_JIT_CACHE_DIR
mkdir -p "$NCCL_EP_JIT_CACHE_DIR"

SCRIPT="${SCRIPT_DIR}/run_ep.py"

RET=0

# Run the suite once per IO mode. Modes can't be mixed in one process
# (ep_bootstrap is once-per-process), so zero-copy gets its own run; only the
# zero-copy-capable tests execute there (the rest self-skip).
run_pass() {
  local label="$1"
  local zc="$2"
  local log="stdout_ep_${label}.txt"
  echo "=== Running ${SCRIPT} [${label}] on ${NUM_RANKS} GPUs (timeout=${TEST_TIMEOUT_S}s) ==="
  # setsid + kill-after so SIGKILL takes down the whole process group, not just torchrun.
  NVTE_EP_ZERO_COPY="${zc}" setsid timeout --foreground --kill-after=10 --signal=TERM \
    "${TEST_TIMEOUT_S}" \
    torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_RANKS}" \
    "${SCRIPT}" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  pkill -9 -f "tests/pytorch/distributed/run_ep.py" 2>/dev/null || true

  if [ "${rc}" -ne 0 ]; then echo "[${label}] torchrun exited with ${rc}"; RET=1; fi
  # Match unittest failure markers and unhandled Python tracebacks; torchrun
  # prefixes per-rank stderr with "[rankN]:" so don't anchor at column 0.
  if grep -qE "(^|]:)FAILED|(^|]:)Traceback" "${log}"; then RET=1; fi
  if ! grep -qE "Ran [0-9]+ test|^OK$" "${log}"; then
    echo "[${label}] ERROR: no test summary — likely hang or early crash"
    RET=1
  fi
  if [ -z "${KEEP_EP_LOGS:-}" ]; then rm -f "${log}"; fi
}

run_pass "default" 0
run_pass "zero_copy" 1

exit $RET
