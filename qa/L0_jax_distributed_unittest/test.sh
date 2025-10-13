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

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

export NVTE_JAX_UNITTEST_LEVEL="L0"
# Make tests have run-to-run deterministic to have stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_multigpu_encoder.xml $TE_PATH/tests/jax/test_distributed_sanity_e2e_train.py || test_fail "test_distributed_sanity_e2e_train.py"

if [ $RET -ne 0 ]; then
    echo "Error: some sub-tests failed: $FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
