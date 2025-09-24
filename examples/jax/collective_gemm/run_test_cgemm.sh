# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

# Check if NVLINK is supported before running tests
echo "*** Checking NVLINK support***"
NVLINK_OUTPUT=$(nvidia-smi nvlink --status 2>&1)
NVLINK_EXIT_CODE=$?

# Check if command failed OR output indicates no NVLINK
if [ $NVLINK_EXIT_CODE -ne 0 ] || [[ "$NVLINK_OUTPUT" == *"not supported"* ]] || [[ "$NVLINK_OUTPUT" == *"No devices"* ]] || [ -z "$NVLINK_OUTPUT" ]; then
  echo "NVLINK is not supported on this platform"
  echo "Collective GEMM tests require NVLINK connectivity"
  echo "SKIPPING all tests"
  exit 0
else
  echo "NVLINK support detected"
fi

# Define the test files to run
TEST_FILES=(
"test_gemm.py"
"test_dense_grad.py"
"test_layernorm_mlp_grad.py"
)

echo
echo "*** Executing tests in examples/jax/collective_gemm/ ***"

HAS_FAILURE=0  # Global failure flag
PIDS=()  # Array to store all process PIDs

# Cleanup function to kill all processes
cleanup() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Killing process $pid"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  # Wait a bit and force kill if needed
  sleep 2
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing process $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

# Set up signal handlers to cleanup on exit
trap cleanup EXIT INT TERM

# Run each test file across all GPUs
for TEST_FILE in "${TEST_FILES[@]}"; do
  echo
  echo "=== Starting test file: $TEST_FILE ..."

  # Clear PIDs array for this test file
  PIDS=()

  for i in $(seq 0 $(($NUM_GPUS - 1))); do
    # Define output file for logs
    LOG_FILE="${TEST_FILE}_gpu_${i}.log"

    if [ $i -eq 0 ]; then
      # For process 0: show live output AND save to log file using tee
      echo "=== Live output from process 0 ==="
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs "$TE_PATH/examples/jax/collective_gemm/$TEST_FILE" \
        --num-processes=$NUM_GPUS \
        --process-id=$i 2>&1 | tee "$LOG_FILE" &
      PID=$!
      PIDS+=($PID)
    else
      # For other processes: redirect to log files only
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs "$TE_PATH/examples/jax/collective_gemm/$TEST_FILE" \
        --num-processes=$NUM_GPUS \
        --process-id=$i > "$LOG_FILE" 2>&1 &
      PID=$!
      PIDS+=($PID)
    fi
  done

  # Wait for all processes to finish
  wait

  # Check and print the log content from process 0 (now has log file thanks to tee)
  if grep -q "SKIPPED" "${TEST_FILE}_gpu_0.log"; then
    echo "... $TEST_FILE SKIPPED"
  elif grep -q "PASSED" "${TEST_FILE}_gpu_0.log"; then
    echo "... $TEST_FILE PASSED"
  else
    HAS_FAILURE=1
    echo "... $TEST_FILE FAILED"
  fi

  # Remove the log files after processing them
  wait
  rm ${TEST_FILE}_gpu_*.log
done

wait

# Final cleanup (trap will also call cleanup on exit)
cleanup

exit $HAS_FAILURE
