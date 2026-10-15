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

set -x

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_recipe.xml $TE_PATH/tests/pytorch/test_recipe.py || test_fail "test_recipe.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_custom_recipe.xml $TE_PATH/tests/pytorch/test_custom_recipe.py || test_fail "test_custom_recipe.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_nvfp4.xml $TE_PATH/tests/pytorch/nvfp4 || test_fail "test_nvfp4"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mxfp8.xml $TE_PATH/tests/pytorch/mxfp8 || test_fail "test_mxfp8"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_weight_swizzle_in_layers.xml $TE_PATH/tests/pytorch/test_weight_swizzle_in_layers.py || test_fail "test_weight_swizzle_in_layers.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_quantized_tensor.xml $TE_PATH/tests/pytorch/test_quantized_tensor.py || test_fail "test_quantized_tensor.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_ptq_calibration_metadata_buffering.xml $TE_PATH/tests/pytorch/test_ptq_calibration_metadata_buffering.py || test_fail "test_ptq_calibration_metadata_buffering.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8blockwisetensor.xml $TE_PATH/tests/pytorch/test_float8blockwisetensor.py || test_fail "test_float8blockwisetensor.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py || test_fail "test_float8_blockwise_scaling_exact.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_gemm_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py || test_fail "test_float8_blockwise_gemm_exact.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_current_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_current_scaling_exact.py || test_fail "test_float8_current_scaling_exact.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/test_grouped_tensor.xml $TE_PATH/tests/pytorch/test_grouped_tensor.py || test_fail "test_grouped_tensor.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_hybrid_quantization.xml $TE_PATH/tests/pytorch/test_hybrid_quantization.py || test_fail "test_hybrid_quantization.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_identity_quantizer.xml $TE_PATH/tests/pytorch/test_identity_quantizer.py || test_fail "test_identity_quantizer.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_mxfp8_2d_quantize.xml $TE_PATH/tests/pytorch/test_mxfp8_2d_quantize.py || test_fail "test_mxfp8_2d_quantize.py"

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi

echo "All tests passed"
exit 0
