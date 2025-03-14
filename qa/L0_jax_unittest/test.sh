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

pip3 install "nltk>=3.8.2" || error_exit "Failed to install nltk"
pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"
: ${TE_PATH:=/opt/transformerengine}

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax -k 'not distributed' --ignore=$TE_PATH/tests/jax/test_praxis_layers.py || test_fail "test_praxis_layers.py"

# Test without custom calls
NVTE_CUSTOM_CALLS_RE="" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax/test_custom_call_compute.py || test_fail "test_custom_call_compute.py"

pip3 install -r $TE_PATH/examples/jax/mnist/requirements.txt || error_exit "Failed to install mnist requirements"
pip3 install -r $TE_PATH/examples/jax/encoder/requirements.txt || error_exit "Failed to install encoder requirements"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/mnist || test_fail "test_mnist.py"

# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder/test_single_gpu_encoder.py || test_fail "test_single_gpu_encoder.py"

if [ $RET -ne 0 ]; then
    echo "Error: some sub-tests failed: $FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
