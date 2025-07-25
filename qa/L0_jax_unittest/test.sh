# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -x

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

pip3 install "nltk>=3.8.2" || error_exit "Failed to install nltk"
pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_jax_not_distributed.xml $TE_PATH/tests/jax -k 'not distributed' || test_fail "tests/jax/*not_distributed_*"

pip3 install -r $TE_PATH/examples/jax/mnist/requirements.txt || error_exit "Failed to install mnist requirements"
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_mnist.xml $TE_PATH/examples/jax/mnist || test_fail "mnist"

pip3 install -r $TE_PATH/examples/jax/encoder/requirements.txt || error_exit "Failed to install encoder requirements"
# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_single_gpu_encoder.xml $TE_PATH/examples/jax/encoder/test_single_gpu_encoder.py || test_fail "test_single_gpu_encoder.py"
# Test without custom calls
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
NVTE_JAX_CUSTOM_CALLS="false" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_single_gpu_encoder.xml $TE_PATH/examples/jax/encoder/test_single_gpu_encoder.py || test_fail "test_single_gpu_encoder.py without custom calls"

if [ $RET -ne 0 ]; then
    echo "Error: some sub-tests failed: $FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
