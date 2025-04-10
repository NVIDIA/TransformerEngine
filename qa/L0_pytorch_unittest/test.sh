# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -x

source $(dirname "$0")/../test_utils.sh
initialize_test_variables

: ${TE_PATH:=/opt/transformerengine}

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

python3 -m pytest -v -s --junitxml=/logs/pytest_test_sanity.xml $TE_PATH/tests/pytorch/test_sanity.py || test_fail "test_sanity.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_recipe.xml $TE_PATH/tests/pytorch/test_recipe.py || test_fail "test_recipe.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_deferred_init.xml $TE_PATH/tests/pytorch/test_deferred_init.py || test_fail "test_deferred_init.py"
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest -v -s --junitxml=/logs/pytest_test_numerics.xml $TE_PATH/tests/pytorch/test_numerics.py || test_fail "test_numerics.py"
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest -v -s --junitxml=/logs/pytest_test_cuda_graphs.xml $TE_PATH/tests/pytorch/test_cuda_graphs.py || test_fail "test_cuda_graphs.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_jit.xml $TE_PATH/tests/pytorch/test_jit.py || test_fail "test_jit.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_fused_rope.xml $TE_PATH/tests/pytorch/test_fused_rope.py || test_fail "test_fused_rope.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_float8tensor.xml $TE_PATH/tests/pytorch/test_float8tensor.py || test_fail "test_float8tensor.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_float8blockwisetensor.xml $TE_PATH/tests/pytorch/test_float8blockwisetensor.py || test_fail "test_float8blockwisetensor.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_float8_blockwise_scaling_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py || test_fail "test_float8_blockwise_scaling_exact.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_float8_blockwise_gemm_exact.xml $TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py || test_fail "test_float8_blockwise_gemm_exact.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_gqa.xml $TE_PATH/tests/pytorch/test_gqa.py || test_fail "test_gqa.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_fused_optimizer.xml $TE_PATH/tests/pytorch/test_fused_optimizer.py || test_fail "test_fused_optimizer.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_multi_tensor.xml $TE_PATH/tests/pytorch/test_multi_tensor.py || test_fail "test_multi_tensor.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_fusible_ops.xml $TE_PATH/tests/pytorch/test_fusible_ops.py || test_fail "test_fusible_ops.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_permutation.xml $TE_PATH/tests/pytorch/test_permutation.py || test_fail "test_permutation.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_parallel_cross_entropy.xml $TE_PATH/tests/pytorch/test_parallel_cross_entropy.py || test_fail "test_parallel_cross_entropy.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_cpu_offloading.xml $TE_PATH/tests/pytorch/test_cpu_offloading.py || test_fail "test_cpu_offloading.py"
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python3 -m pytest -o log_cli=true --log-cli-level=INFO -v -s --junitxml=/logs/pytest_test_fused_attn.xml $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py || test_fail "test_fused_attn.py"
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python3 -m pytest -o log_cli=true --log-cli-level=INFO -v -s --junitxml=/logs/pytest_test_kv_cache.xml $TE_PATH/tests/pytorch/fused_attn/test_kv_cache.py || test_fail "test_kv_cache.py"

check_test_results
