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


pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml $TE_PATH/tests/pytorch/distributed/test_numerics.py || test_fail "test_numerics.py"
python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_fusible_ops.xml $TE_PATH/tests/pytorch/distributed/test_fusible_ops.py || test_fail "test_fusible_ops.py"
python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_torch_fsdp2.xml $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py || test_fail "test_torch_fsdp2.py"
python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_comm_gemm_overlap.xml $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py || test_fail "test_comm_gemm_overlap.py"
python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_fusible_ops_with_userbuffers.xml $TE_PATH/tests/pytorch/distributed/test_fusible_ops_with_userbuffers.py || test_fail "test_fusible_ops_with_userbuffers.py"
python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_fused_attn_with_cp.xml $TE_PATH/tests/pytorch/fused_attn/test_fused_attn_with_cp.py || test_fail "test_fused_attn_with_cp.py"
python3 -m pytest -v -s --junitxml=$XML_LOG_DIR/pytest_test_cast_master_weights_to_fp8.xml $TE_PATH/tests/pytorch/distributed/test_cast_master_weights_to_fp8.py || test_fail "test_cast_master_weights_to_fp8.py"


# debug tests


# Config with the dummy feature which prevents nvinspect from being disabled.
# Nvinspect will be disabled if no feature is active.
: ${NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE:=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml}
: ${NVTE_TEST_NVINSPECT_FEATURE_DIRS:=$TE_PATH/transformer_engine/debug/features}

pytest -v -s $TE_PATH/tests/pytorch/debug/test_distributed.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS || test_fail "debug test_distributed.py"
# standard numerics tests with initialized debug
NVTE_TEST_NVINSPECT_ENABLED=True NVTE_TEST_NVINSPECT_CONFIG_FILE=$NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE NVTE_TEST_NVINSPECT_FEATURE_DIRS=$NVTE_TEST_NVINSPECT_FEATURE_DIRS pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics.py || test_fail "debug test_numerics.py"

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
