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

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_sanity.py || test_fail "test_sanity.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_recipe.py || test_fail "test_recipe.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_deferred_init.py || test_fail "test_deferred_init.py"
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_numerics.py || test_fail "test_numerics.py"
NVTE_CUDNN_MXFP8_NORM=0 PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_cuda_graphs.py || test_fail "test_cuda_graphs.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_jit.py || test_fail "test_jit.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_fused_rope.py || test_fail "test_fused_rope.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_float8tensor.py || test_fail "test_float8tensor.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_gqa.py || test_fail "test_gqa.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_fused_optimizer.py || test_fail "test_fused_optimizer.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_multi_tensor.py || test_fail "test_multi_tensor.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_fusible_ops.py || test_fail "test_fusible_ops.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_onnx_export.py || test_fail "test_onnx_export.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_permutation.py || test_fail "test_permutation.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_parallel_cross_entropy.py || test_fail "test_parallel_cross_entropy.py"
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_cpu_offloading.py || test_fail "test_cpu_offloading.py"
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python3 -m pytest -o log_cli=true --log-cli-level=INFO -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py || test_fail "test_fused_attn.py"
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python3 -m pytest -o log_cli=true --log-cli-level=INFO -v -s $TE_PATH/tests/pytorch/fused_attn/test_paged_attn.py || test_fail "test_paged_attn.py"
if [ "$RET" -ne 0 ]; then
    echo "Error in the following test cases:$FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
