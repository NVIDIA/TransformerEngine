# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

# Define the test cases to run
TEST_CASES=(
"test_te_bf16"
"test_te_delayed_scaling_fp8"
"test_te_current_scaling_fp8"
"test_te_mxfp8"
"test_te_bf16_shardy"
"test_te_delayed_scaling_fp8_shardy"
"test_te_current_scaling_fp8_shardy"
)

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

echo
echo "*** Executing tests in examples/jax/encoder/test_multiprocessing_encoder.py ***"

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
# Run each test case across all GPUs
for TEST_CASE in "${TEST_CASES[@]}"; do
  echo
  echo "=== Starting test: $TEST_CASE ..."

  for i in $(seq 0 $(($NUM_GPUS - 1))); do
    # Define output file for logs
    LOG_FILE="${TEST_CASE}_gpu_${i}.log"

    # For process 0: show live output AND save to log file using tee
    if [ $i -eq 0 ]; then
      echo "=== Live output from process 0 ==="
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs --junitxml=$XML_LOG_DIR/multiprocessing_encoder_${TEST_CASE}.xml \
        "$TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py::TestEncoder::$TEST_CASE" \
        --num-process=$NUM_GPUS \
        --process-id=$i 2>&1 | tee "$LOG_FILE" &
      PID=$!
      PIDS+=($PID)
    else
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs "$TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py::TestEncoder::$TEST_CASE" \
        --num-process=$NUM_GPUS \
        --process-id=$i > "$LOG_FILE" 2>&1 &
      PID=$!
      PIDS+=($PID)
    fi
  done

  # Wait for the process to finish
  wait

  # Check and print the log content accordingly
  if grep -q "SKIPPED" "${TEST_CASE}_gpu_0.log"; then
    echo "... $TEST_CASE SKIPPED"
  elif grep -q "FAILED" "${TEST_CASE}_gpu_0.log"; then
    echo "... $TEST_CASE FAILED"
    HAS_FAILURE=1
  elif grep -q "PASSED" "${TEST_CASE}_gpu_0.log"; then
    echo "... $TEST_CASE PASSED"
  else
    echo "... $TEST_CASE INVALID"
    HAS_FAILURE=1
  fi

  # Remove the log file after processing it
  wait
  rm ${TEST_CASE}_gpu_*.log
done

wait

# Final cleanup (trap will also call cleanup on exit)
cleanup

exit $HAS_FAILURE
