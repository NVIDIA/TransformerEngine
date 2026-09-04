# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_torch_fsdp2_hybrid.xml $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py -k "hybrid" || test_fail "hybrid test_torch_fsdp2.py"
python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_hybrid_tp_sp.xml $TE_PATH/tests/pytorch/distributed/test_hybrid_tp_sp.py || test_fail "test_hybrid_tp_sp.py"

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
