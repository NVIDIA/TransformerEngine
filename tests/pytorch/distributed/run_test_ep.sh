#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Launcher for tests/pytorch/distributed/run_ep.py. Auto-detects GPU count.
# Short timeout by default to surface hangs early.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
  local eager="${3:-0}"
  local overflow="${4:-0}"
  local mxfp8="${5:-0}"
  local fused="${6:-0}"
  local overlap="${7:-0}"
  local log="stdout_ep_${label}.txt"
  # The overlap passes compile CuTeDSL GEMMs on first use and need a longer timeout.
  local timeout_s="${TEST_TIMEOUT_S}"
  if [ "${overlap}" = "1" ] && [ "${timeout_s}" -lt 600 ]; then timeout_s=600; fi
  echo "=== Running ${SCRIPT} [${label}] on ${NUM_RANKS} GPUs (timeout=${timeout_s}s) ==="
  # setsid + kill-after so SIGKILL takes down the whole process group, not just torchrun.
  NVTE_EP_ZERO_COPY="${zc}" NVTE_EP_EAGER="${eager}" NVTE_EP_OVERFLOW="${overflow}" \
    NVTE_EP_MXFP8_PASS="${mxfp8}" NVTE_EP_FUSED_PREPARE_DISPATCH="${fused}" NVTE_EP_OVERLAP="${overlap}" \
    setsid timeout --foreground --kill-after=10 --signal=TERM "${timeout_s}" \
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
  # An overlap pass in which a rank never ran the fused ops covers nothing: tests that skip, or
  # that fall back to the unfused ops, do not report. Every rank reports from a numerical test.
  if [ "${overlap}" = "1" ]; then
    local executed
    executed=$(grep -oE "OVERLAP_EXECUTED rank=[0-9]+" "${log}" | sort -u | wc -l)
    # A pass skipped for lack of backward SM budgets in NCCL EP executed nothing, by design.
    if grep -q "NCCL EP has no backward SM budgets" "${log}"; then
      echo "[${label}] SKIPPED: NCCL EP has no backward SM budgets"
    elif [ "${executed}" -lt "${NUM_RANKS}" ]; then
      echo "[${label}] ERROR: the fused overlap ops ran on ${executed} of ${NUM_RANKS} ranks;" \
        "set NVTE_ENABLE_CUTEDSL_BACKEND=1 and NVTE_MOE_OVERLAP"
      RET=1
    fi
  fi
  if [ -z "${KEEP_EP_LOGS:-}" ]; then rm -f "${log}"; fi
}

# Opt-in via NVTE_TEST_EP_CLASSIC_PG=1: cover only the classic ProcessGroupNCCL
# comm-borrow branch. Recent torch defaults the "nccl" backend to ProcessGroupNCCL2,
# which the passes below exercise; forcing the classic backend covers the other
# get_nccl_comm_ptr branch. Torch builds without the classic backend (pytorch#205374)
# ignore TORCH_DIST_USE_NCCL2=0, so this pass would only repeat the NCCL2 coverage.
if [ "${NVTE_TEST_EP_CLASSIC_PG:-0}" = "1" ]; then
  # Exit 3 means the header is missing; any other failure is a real error, not a skip.
  python3 -c 'import torch, pathlib, sys; sys.exit(0 if (pathlib.Path(torch.__file__).parent / "include/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp").exists() else 3)'
  probe=$?
  if [ "$probe" -eq 3 ]; then
    echo "SKIP classic_pg: this torch build has no classic ProcessGroupNCCL"
    exit 0
  elif [ "$probe" -ne 0 ]; then
    echo "classic_pg: probing torch for the classic ProcessGroupNCCL failed (exit $probe)"
    exit 1
  fi
  TORCH_DIST_USE_NCCL2=0 run_pass "classic_pg" 0
  exit $RET
fi

run_pass "default" 0
run_pass "zero_copy" 1
run_pass "eager" 0 1
run_pass "overflow" 0 0 1
# The MXFP8 pass uses per-expert alignment 256 for grouped MLP fusion. The backend caches it
# process-wide, so its tests get their own passes (normal + zero-copy + eager IO). mxfp8 is the
# 5th arg; eager is the 3rd.
run_pass "mxfp8" 0 0 0 1
run_pass "mxfp8_zero_copy" 1 0 0 1
run_pass "mxfp8_eager" 0 1 0 1

# Fused count-mode dispatch under CUDA graph capture (NVTE_EP_FUSED_PREPARE_DISPATCH=1) is
# opt-in and needs its own passes so the fused-only tests aren't skipped: overflow and mxfp8
# variants need their own gating envs too, and all require non-eager (caller-provided recv).
run_pass "fused" 0 0 0 0 1
run_pass "fused_overflow" 0 0 1 0 1
run_pass "fused_mxfp8" 0 0 0 1 1

# Overlapped MoE ops (ep_gemm.h, te.ops FusedMoE): alignment 256, capacity and eager IO. Run only
# when the overlap is available (NVTE_ENABLE_CUTEDSL_BACKEND=1, NVTE_MOE_OVERLAP and SM 10.x).
PROBE_ERR=$(mktemp)
OVERLAP_AVAILABLE=$(python3 -c \
  "import transformer_engine.pytorch.ep as ep; print(int(ep._moe_overlap_available()))" 2>"${PROBE_ERR}")
if [ $? -ne 0 ]; then
  cat "${PROBE_ERR}"
  OVERLAP_AVAILABLE=0
  # A failed probe is not "unsupported": when the overlap was requested it is a failure.
  if [ -n "${NVTE_MOE_OVERLAP:-}" ]; then
    echo "ERROR: the overlap availability probe failed with NVTE_MOE_OVERLAP set"
    RET=1
  fi
fi
rm -f "${PROBE_ERR}"
if [ "${OVERLAP_AVAILABLE}" != "1" ]; then
  echo "Overlapped MoE ops unavailable (needs NVTE_ENABLE_CUTEDSL_BACKEND=1, NVTE_MOE_OVERLAP, SM 10.x); SKIPPING overlap passes."
else
  run_pass "overlap" 0 0 0 1 0 1
  run_pass "overlap_eager" 0 1 0 1 0 1
  run_pass "overlap_overflow" 0 0 1 1 0 1
  # Per-direction SM counts (dispatch fwd, combine fwd, dispatch bwd, combine bwd). The backward
  # split skips when Transformer Engine was built against an NCCL EP without backward SM budgets.
  NVTE_EP_COMM_SMS=12,20,0,0 run_pass "overlap_split" 0 0 0 1 0 1
  NVTE_EP_COMM_SMS=12,20,24,16 run_pass "overlap_split_bwd" 0 0 0 1 0 1
  # A hidden size that 1024 does not divide: the combine waits on whole rows.
  NVTE_EP_HIDDEN_DIM=1536 run_pass "overlap_h1536" 0 0 0 1 0 1
  # Expert alignments below 256 (MXFP8 cases skip at 64): an expert's last GEMM tile holds fewer
  # rows. With 288 tokens per rank each expert also fills a whole tile; only the fused op tests run.
  for align in 128 64; do
    NVTE_EP_OVERLAP_ALIGNMENT=${align} run_pass "overlap_a${align}" 0 0 0 1 0 1
    NVTE_EP_OVERLAP_ALIGNMENT=${align} run_pass "overlap_a${align}_eager" 0 1 0 1 0 1
    NVTE_EP_OVERLAP_ALIGNMENT=${align} NVTE_EP_TOKENS_PER_RANK=288 \
      NVTE_EP_TEST_FILTER=test_moe_overlap run_pass "overlap_a${align}_t288" 0 0 0 1 0 1
  done
fi

exit $RET
