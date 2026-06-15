# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#!/bin/bash

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

if [ "${NUM_GPUS}" -lt 4 ]; then
  echo "NCCL EP requires at least 4 GPUs (found ${NUM_GPUS}); SKIPPING."
  exit 0
fi
# Default mesh is (2, 2); use exactly 4 ranks even on larger boxes.
NUM_GPUS="${NVTE_EP_NUM_RANKS:-4}"

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# NCCL EP requires NVLink P2P among ranks on the node.
echo "*** Checking NVLINK support ***"
NVLINK_OUTPUT=$(nvidia-smi nvlink --status 2>&1)
NVLINK_EXIT_CODE=$?
if [ $NVLINK_EXIT_CODE -ne 0 ] || [[ "$NVLINK_OUTPUT" == *"not supported"* ]] \
   || [[ "$NVLINK_OUTPUT" == *"No devices"* ]] || [ -z "$NVLINK_OUTPUT" ]; then
  echo "NVLINK is not supported on this platform — EP example requires NVLINK; SKIPPING"
  exit 0
fi
echo "NVLINK support detected"

SCRIPT="$TE_PATH/examples/jax/ep/ep_moe.py"
export PYTHONPATH="${TE_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
COORD="${COORD:-127.0.0.1:12345}"
TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-300}"

XLA_BASE_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_graph_min_graph_size=1"
export XLA_FLAGS="${XLA_BASE_FLAGS}"

# Stage NCCL EP JIT cubins on tmpfs to keep build/iteration fast.
: ${NCCL_EP_JIT_CACHE_DIR:="${TMPDIR:-/tmp}/nccl_ep_jit_cache_$(id -u)"}
export NCCL_EP_JIT_CACHE_DIR
mkdir -p "$NCCL_EP_JIT_CACHE_DIR"

echo
echo "*** Executing ep_moe.py across $NUM_GPUS GPUs ***"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

EXTRA_ARGS=${EXTRA_ARGS:-"--check"}

for ((i=1; i<NUM_GPUS; i++)); do
  timeout --foreground --signal=KILL "${TEST_TIMEOUT_S}" \
    python -u "$SCRIPT" \
      --coordinator-address "$COORD" --process-id "$i" --num-processes "$NUM_GPUS" \
      $EXTRA_ARGS > "stdout_rank_${i}.txt" 2>&1 &
  PIDS+=($!)
done
timeout --foreground --signal=KILL "${TEST_TIMEOUT_S}" \
  python -u "$SCRIPT" \
    --coordinator-address "$COORD" --process-id "0" --num-processes "$NUM_GPUS" \
    $EXTRA_ARGS 2>&1 | tee stdout_rank_0.txt
wait

HAS_FAILURE=0
if grep -qE "FAILED|Traceback|ERROR" stdout_rank_0.txt; then
  echo "... ep_moe FAILED"
  HAS_FAILURE=1
elif ! grep -qE "\[ep_moe\]" stdout_rank_0.txt; then
  echo "... ep_moe INVALID (rank 0 produced no summary line)"
  for ((i=1; i<NUM_GPUS; i++)); do
    echo "--- rank $i log ---"
    tail -30 "stdout_rank_${i}.txt" 2>/dev/null
  done
  HAS_FAILURE=1
else
  echo "... ep_moe PASSED"
fi
rm -f stdout_rank_*.txt
exit $HAS_FAILURE
