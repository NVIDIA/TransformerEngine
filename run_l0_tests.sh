#!/bin/bash
# Run all L0 tests, capture summary lines
set -x
TE_PATH=/workspace
RESULTS=/workspace/l0_results.txt
> "$RESULTS"

cleanup_gpu() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 2
}

run_test() {
    local name="$1"
    shift
    cleanup_gpu
    echo "=== $name ===" >> "$RESULTS"
    "$@" 2>&1 | tee /tmp/test_${name}.log | tail -5 >> "$RESULTS"
    echo "EXIT=$?" >> "$RESULTS"
    echo "" >> "$RESULTS"
}

run_test "test_sanity" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_sanity.py
run_test "test_recipe" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_recipe.py
run_test "test_deferred_init" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_deferred_init.py
run_test "test_numerics" env PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_numerics.py
run_test "test_cuda_graphs" env PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_cuda_graphs.py
run_test "test_jit" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_jit.py
run_test "test_fused_rope" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_fused_rope.py
run_test "test_nvfp4" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/nvfp4
run_test "test_mxfp8" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/mxfp8
run_test "test_quantized_tensor" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_quantized_tensor.py
run_test "test_float8blockwisetensor" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_float8blockwisetensor.py
run_test "test_float8_blockwise_scaling_exact" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py
run_test "test_float8_blockwise_gemm_exact" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py
run_test "test_grouped_tensor" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_grouped_tensor.py
run_test "test_gqa" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_gqa.py
run_test "test_fused_optimizer" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_fused_optimizer.py
run_test "test_multi_tensor" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_multi_tensor.py
run_test "test_fusible_ops" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_fusible_ops.py
run_test "test_permutation" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_permutation.py
run_test "test_parallel_cross_entropy" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_parallel_cross_entropy.py
run_test "test_cpu_offloading" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_cpu_offloading.py
run_test "test_cpu_offloading_v1" env NVTE_FLASH_ATTN=0 NVTE_CPU_OFFLOAD_V1=1 python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_cpu_offloading_v1.py
run_test "test_attention" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/attention/test_attention.py
run_test "test_attention_deterministic" env NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/attention/test_attention.py
run_test "test_kv_cache" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/attention/test_kv_cache.py
run_test "test_hf_integration" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_hf_integration.py
run_test "test_checkpoint" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_checkpoint.py
run_test "test_fused_router" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_fused_router.py
run_test "test_partial_cast" python3 -m pytest --tb=no -q $TE_PATH/tests/pytorch/test_partial_cast.py

echo "=== ALL DONE ===" >> "$RESULTS"
cat "$RESULTS"
