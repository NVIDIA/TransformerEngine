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

echo
echo "*** Executing tests in examples/jax/encoder/test_multiprocessing_encoder.py ***"

HAS_FAILURE=0  # Global failure flag

# Run each test case across all GPUs
for TEST_CASE in "${TEST_CASES[@]}"; do
  echo
  echo "=== Starting test: $TEST_CASE ..."

  for i in $(seq 0 $(($NUM_GPUS - 1))); do
    # Define output file for logs
    LOG_FILE="${TEST_CASE}_gpu_${i}.log"

    # Run pytest and redirect stdout and stderr to the log file
    pytest -c "$TE_PATH/tests/jax/pytest.ini" \
      -vs "$TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py::TestEncoder::$TEST_CASE" \
      --num-process=$NUM_GPUS \
      --process-id=$i > "$LOG_FILE" 2>&1 &
    done

  # Wait for the process to finish
  wait

  # Check and print the log content accordingly
  if grep -q "FAILED" "${TEST_CASE}_gpu_0.log"; then
    HAS_FAILURE=1
    echo "... $TEST_CASE FAILED"
    tail -n +7 "${TEST_CASE}_gpu_0.log"
  elif grep -q "SKIPPED" "${TEST_CASE}_gpu_0.log"; then
    echo "... $TEST_CASE SKIPPED"
  elif grep -q "PASSED" "${TEST_CASE}_gpu_0.log"; then
    echo "... $TEST_CASE PASSED"
  else
    echo "Invalid ${TEST_CASE}_gpu_0.log"
  fi

  # Remove the log file after processing it
  rm ${TEST_CASE}_gpu_*.log
done

exit $HAS_FAILURE
