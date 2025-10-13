# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

function error_exit() {
    echo "Error: $1"
    exit 1
}

function test_fail() {
    RET=1
    FAILED_CASES="$FAILED_CASES $1"
    echo "Error: sub-test failed: $1"
}

RET=0
FAILED_CASES=""

set -xe

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

export NVTE_JAX_UNITTEST_LEVEL="L1"
# Make tests have run-to-run deterministic to have stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_* || test_fail "test_distributed_*"

SCRIPT_NAME=test_multi_process_distributed_grouped_gemm.py bash $TE_PATH/tests/jax/multi_process_launch.sh || test_fail "distributed_grouped_gemm"

## MNIST example
pip3 install -r $TE_PATH/examples/jax/mnist/requirements.txt || error_exit "Failed to install mnist requirements"
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_mnist.xml $TE_PATH/examples/jax/mnist || test_fail "mnist"

## Encoder examples
pip3 install -r $TE_PATH/examples/jax/encoder/requirements.txt || error_exit "Failed to install requirements"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_single_gpu_encoder.xml $TE_PATH/examples/jax/encoder/test_single_gpu_encoder.py || test_fail "test_single_gpu_encoder.py"
# Test without custom calls
NVTE_JAX_CUSTOM_CALLS="false" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_single_gpu_encoder_without_custom_call.xml $TE_PATH/examples/jax/encoder/test_single_gpu_encoder.py || test_fail "test_single_gpu_encoder.py without custom calls"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_multigpu_encoder.xml $TE_PATH/examples/jax/encoder/test_multigpu_encoder.py || test_fail "test_multigpu_encoder.py"
wait
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_model_parallel_encoder.xml $TE_PATH/examples/jax/encoder/test_model_parallel_encoder.py || test_fail "test_model_parallel_encoder.py"
wait
TE_PATH=$TE_PATH bash $TE_PATH/examples/jax/encoder/run_test_multiprocessing_encoder.sh || test_fail "run_test_multiprocessing_encoder.sh"
wait

TE_PATH=$TE_PATH bash $TE_PATH/examples/jax/collective_gemm/run_test_cgemm.sh || test_fail "run_test_cgemm.sh"
wait

if [ $RET -ne 0 ]; then
    echo "Error: some sub-tests failed: $FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
