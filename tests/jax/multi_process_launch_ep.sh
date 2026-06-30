# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

#!/bin/bash

SCRIPT_NAMES="${SCRIPT_NAMES:-test_multi_process_ep.py}"
TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-180}"


XLA_BASE_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_graph_min_graph_size=1"

export XLA_FLAGS="${XLA_BASE_FLAGS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${TE_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Editable installs don't embed rpath; libtransformer_engine.so needs
# libnccl_ep.so.0 from the TE editable location at dlopen time.
TE_LIB_PATH=$(pip3 show transformer-engine 2>/dev/null \
    | grep -E "Location:|Editable project location:" \
    | tail -n 1 | awk '{print $NF}')
if [ -n "$TE_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="${TE_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

NUM_RUNS=$(nvidia-smi -L | wc -l)

if [ "${NUM_RUNS}" -lt 4 ]; then
  echo "NCCL EP requires at least 4 GPUs (found ${NUM_RUNS}); SKIPPING."
  exit 0
fi

# NCCL EP requires active NVLink P2P among ranks on the node.
if ! nvidia-smi nvlink --status 2>/dev/null | grep -qE 'Link [0-9]+:.*GB/s'; then
  echo "NVLink not detected on this platform — EP test requires NVLink; SKIPPING."
  exit 0
fi

# Default test mesh is (2, 2); use exactly 4 ranks even on larger boxes.
NUM_RUNS="${NVTE_TEST_EP_NUM_RANKS:-4}"

OVERALL_RET=0

for SCRIPT_NAME in $SCRIPT_NAMES; do
  # Allow callers to pass either a bare test name (resolved against this
  # script's directory) or an absolute/relative path.
  if [ -f "$SCRIPT_NAME" ]; then
    SCRIPT_PATH="$SCRIPT_NAME"
  else
    SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
  fi
  echo "=== Running ${SCRIPT_PATH} ==="
  for ((i=1; i<NUM_RUNS; i++))
  do
      timeout --foreground --signal=KILL "${TEST_TIMEOUT_S}" \
          python "$SCRIPT_PATH" 127.0.0.1:12345 $i $NUM_RUNS > stdout_rank_${i}.txt 2>&1 &
  done

  timeout --foreground --signal=KILL "${TEST_TIMEOUT_S}" \
      python "$SCRIPT_PATH" 127.0.0.1:12345 0 $NUM_RUNS 2>&1 | tee stdout_multi_process.txt

  wait

  RET=0
  if grep -q "FAILED" stdout_multi_process.txt; then
    RET=1
  fi
  # Treat missing test summary on rank 0 as hang/crash rather than silent success.
  if ! grep -qE "Ran [0-9]+ test|^OK$|PASSED" stdout_multi_process.txt; then
    echo "ERROR: rank 0 produced no test summary for ${SCRIPT_NAME} — likely a hang or early crash."
    echo "       NCCL EP requires NVLS multicast; check NCCL_DEBUG=INFO output."
    RET=1
  fi
  if [ "$RET" -ne 0 ]; then
    for ((i=1; i<NUM_RUNS; i++)); do
      echo "--- rank $i log ---"
      cat stdout_rank_${i}.txt 2>/dev/null || echo "(no log)"
    done
  fi

  rm -f stdout_multi_process.txt stdout_rank_*.txt
  if [ "$RET" -ne 0 ]; then
    OVERALL_RET=1
  fi
done

exit "$OVERALL_RET"
