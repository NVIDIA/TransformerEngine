# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

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

# Define individual test cases to run (file::class::method)
# DelayedScalingFP8 and CurrentScalingFP8 use the same GEMM so we don't need to test both cases all
# the time.
TEST_CASES=(
# test_gemm.py cases
"test_gemm.py::TestCollectiveGemmWithDP::test_te_bf16_all_gather_with_dp"
"test_gemm.py::TestCollectiveGemmWithDP::test_te_bf16_reduce_scatter_with_dp"
"test_gemm.py::TestCollectiveGemmWithDP::test_te_delayed_scaling_fp8_all_gather_with_dp"
"test_gemm.py::TestCollectiveGemmWithDP::test_te_delayed_scaling_fp8_reduce_scatter_with_dp"
"test_gemm.py::TestCollectiveGemmWithDP::test_te_mxfp8_all_gather_with_dp"
"test_gemm.py::TestCollectiveGemmWithDP::test_te_mxfp8_reduce_scatter_with_dp"
# # "test_gemm.py::TestCollectiveGemmWithDP::test_te_nvfp4_all_gather_with_dp"
# # "test_gemm.py::TestCollectiveGemmWithDP::test_te_nvfp4_reduce_scatter_with_dp"
#
# # test_dense_grad.py cases
"test_dense_grad.py::TestCollectiveDenseGradient::test_te_bf16_all_gather"
"test_dense_grad.py::TestCollectiveDenseGradient::test_te_bf16_reduce_scatter"
"test_dense_grad.py::TestCollectiveDenseGradient::test_te_current_scaling_fp8_all_gather"
"test_dense_grad.py::TestCollectiveDenseGradient::test_te_current_scaling_fp8_reduce_scatter"
"test_dense_grad.py::TestCollectiveDenseGradient::test_te_mxfp8_all_gather"
"test_dense_grad.py::TestCollectiveDenseGradient::test_te_mxfp8_reduce_scatter"
# "test_dense_grad.py::TestCollectiveDenseGradient::test_te_nvfp4_all_gather"
# "test_dense_grad.py::TestCollectiveDenseGradient::test_te_nvfp4_reduce_scatter"

# test_layernorm_mlp_grad.py cases
"test_layernorm_mlp_grad.py::TestCollectiveLayerNormMLPGradient::test_te_bf16_layernorm_mlp_grad"
"test_layernorm_mlp_grad.py::TestCollectiveLayerNormMLPGradient::test_te_delayed_scaling_fp8_layernorm_mlp_grad"
"test_layernorm_mlp_grad.py::TestCollectiveLayerNormMLPGradient::test_te_current_scaling_fp8_layernorm_mlp_grad"
"test_layernorm_mlp_grad.py::TestCollectiveLayerNormMLPGradient::test_te_mxfp8_layernorm_mlp_grad"
# "test_layernorm_mlp_grad.py::TestCollectiveLayerNormMLPGradient::test_te_nvfp4_layernorm_mlp_grad"
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

# Run each test case across all GPUs
for TEST_CASE in "${TEST_CASES[@]}"; do
  echo
  echo "=== Starting test: $TEST_CASE ..."

  # Extract just the test method name for log/xml file naming
  TEST_NAME=$(echo "$TEST_CASE" | awk -F'::' '{print $NF}')

  # Clear PIDs array for this test case
  PIDS=()

  for i in $(seq 0 $(($NUM_GPUS - 1))); do
    # Define output file for logs
    LOG_FILE="${TEST_NAME}_gpu_${i}.log"

    if [ $i -eq 0 ]; then
      # For process 0: show live output AND save to log file using tee
      echo "=== Live output from process 0 ==="
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs --junitxml=$XML_LOG_DIR/collective_gemm_${TEST_NAME}.xml \
        "$TE_PATH/examples/jax/collective_gemm/$TEST_CASE" \
        --num-processes=$NUM_GPUS \
        --process-id=$i 2>&1 | tee "$LOG_FILE" &
      PID=$!
      PIDS+=($PID)
    else
      # For other processes: redirect to log files only
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs "$TE_PATH/examples/jax/collective_gemm/$TEST_CASE" \
        --num-processes=$NUM_GPUS \
        --process-id=$i > "$LOG_FILE" 2>&1 &
      PID=$!
      PIDS+=($PID)
    fi
  done

  # Wait for all processes to finish
  wait

  # Check and print the log content from process 0
  if grep -q "SKIPPED" "${TEST_NAME}_gpu_0.log"; then
    echo "... $TEST_CASE SKIPPED"
  elif grep -q "FAILED" "${TEST_NAME}_gpu_0.log"; then
    echo "... $TEST_CASE FAILED"
    HAS_FAILURE=1
  elif grep -q "PASSED" "${TEST_NAME}_gpu_0.log"; then
    echo "... $TEST_CASE PASSED"
  else
    echo "... $TEST_CASE INVALID"
    HAS_FAILURE=1
  fi

  # Remove the log files after processing them
  wait
  rm ${TEST_NAME}_gpu_*.log
done

wait

# Final cleanup (trap will also call cleanup on exit)
cleanup

exit $HAS_FAILURE
