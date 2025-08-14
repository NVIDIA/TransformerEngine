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

set -x

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"
pip3 install onnxruntime==1.20.1 || error_exit "Failed to install onnxruntime"
pip3 install onnxruntime_extensions==0.13.0 || error_exit "Failed to install onnxruntime_extensions"

python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_sanity.xml $TE_PATH/tests/pytorch/test_sanity.py || test_fail "test_sanity.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_recipe.xml $TE_PATH/tests/pytorch/test_recipe.py || test_fail "test_recipe.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_deferred_init.xml $TE_PATH/tests/pytorch/test_deferred_init.py || test_fail "test_deferred_init.py"
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_numerics.xml $TE_PATH/tests/pytorch/test_numerics.py || test_fail "test_numerics.py"
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cuda_graphs.xml $TE_PATH/tests/pytorch/test_cuda_graphs.py || test_fail "test_cuda_graphs.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_jit.xml $TE_PATH/tests/pytorch/test_jit.py || test_fail "test_jit.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_rope.xml $TE_PATH/tests/pytorch/test_fused_rope.py || test_fail "test_fused_rope.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8tensor.xml $TE_PATH/tests/pytorch/test_float8tensor.py || test_fail "test_float8tensor.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8blockwisetensor.xml $TE_PATH/tests/pytorch/test_float8blockwisetensor.py || test_fail "test_float8blockwisetensor.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py || test_fail "test_float8_blockwise_scaling_exact.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_float8_blockwise_gemm_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py || test_fail "test_float8_blockwise_gemm_exact.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_gqa.xml $TE_PATH/tests/pytorch/test_gqa.py || test_fail "test_gqa.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_optimizer.xml $TE_PATH/tests/pytorch/test_fused_optimizer.py || test_fail "test_fused_optimizer.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_multi_tensor.xml $TE_PATH/tests/pytorch/test_multi_tensor.py || test_fail "test_multi_tensor.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_onnx_export.xml $TE_PATH/tests/pytorch/test_onnx_export.py || test_fail "test_onnx_export.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fusible_ops.xml $TE_PATH/tests/pytorch/test_fusible_ops.py || test_fail "test_fusible_ops.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_permutation.xml $TE_PATH/tests/pytorch/test_permutation.py || test_fail "test_permutation.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_parallel_cross_entropy.xml $TE_PATH/tests/pytorch/test_parallel_cross_entropy.py || test_fail "test_parallel_cross_entropy.py"
NVTE_FLASH_ATTN=0 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_cpu_offloading.xml $TE_PATH/tests/pytorch/test_cpu_offloading.py || test_fail "test_cpu_offloading.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_attention.xml $TE_PATH/tests/pytorch/attention/test_attention.py || test_fail "test_attention.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_kv_cache.xml $TE_PATH/tests/pytorch/attention/test_kv_cache.py || test_fail "test_kv_cache.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_hf_integration.xml $TE_PATH/tests/pytorch/test_hf_integration.py || test_fail "test_hf_integration.py"
NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=$TE_PATH/artifacts/tests/pytorch/test_checkpoint python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_checkpoint.xml $TE_PATH/tests/pytorch/test_checkpoint.py || test_fail "test_checkpoint.py"
python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/pytest_test_fused_router.xml $TE_PATH/tests/pytorch/test_fused_router.py || test_fail "test_fused_router.py"

if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
