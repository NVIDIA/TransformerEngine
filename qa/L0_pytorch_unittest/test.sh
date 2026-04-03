# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

function error_exit() {
    echo "Error: $1"
    exit 1
}

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

set -x

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

# ── Parallel test infrastructure ────────────────────────────────────────────
# Detect GPUs and run tests in parallel waves (one test per GPU per wave).
# With 1 GPU, behavior is identical to sequential execution.

FAIL_DIR=$(mktemp -d)

function test_fail() {
    echo "$1" >> "$FAIL_DIR/failures"
    echo "Error: sub-test failed: $1"
}

# Detect available GPUs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPU_LIST[@]}
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    NUM_GPUS=${NUM_GPUS:-1}
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
    GPU_LIST=(0)
fi
echo "Detected $NUM_GPUS GPU(s): ${GPU_LIST[*]}"

GPU_COUNTER=0
WAVE_COUNT=0

function run_test() {
    local env_prefix="$1"
    local xml_name="$2"
    local test_path="$3"
    local fail_label="$4"
    local gpu_id=$((GPU_COUNTER % NUM_GPUS))
    GPU_COUNTER=$((GPU_COUNTER + 1))
    WAVE_COUNT=$((WAVE_COUNT + 1))

    if [ "$NUM_GPUS" -le 1 ]; then
        # Single GPU: run synchronously (identical to original behavior)
        eval "${env_prefix} python3 -u -m pytest --tb=auto --junitxml=$XML_LOG_DIR/${xml_name} ${test_path}" \
            || test_fail "$fail_label"
    else
        # Multi GPU: run in background on assigned GPU
        echo ">>> Starting: ${fail_label} on GPU ${GPU_LIST[$gpu_id]}"
        (
            eval "CUDA_VISIBLE_DEVICES=${GPU_LIST[$gpu_id]} ${env_prefix} python3 -u -m pytest --tb=auto --junitxml=$XML_LOG_DIR/${xml_name} ${test_path}" \
                > "$XML_LOG_DIR/${xml_name%.xml}.log" 2>&1 \
                || test_fail "$fail_label"
            echo ">>> Finished: ${fail_label} on GPU ${GPU_LIST[$gpu_id]}"
        ) &
    fi

    # When we've filled all GPUs, wait for the wave to complete
    if [ "$WAVE_COUNT" -ge "$NUM_GPUS" ] && [ "$NUM_GPUS" -gt 1 ]; then
        wait
        WAVE_COUNT=0
    fi
}

# ── Checkpoint pre-step (must run before test_checkpoint.py) ────────────────

export NVTE_TEST_CHECKPOINT_ARTIFACT_PATH=$TE_PATH/artifacts/tests/pytorch/test_checkpoint
if [ ! -d "$NVTE_TEST_CHECKPOINT_ARTIFACT_PATH" ]; then
    python3 $TE_PATH/tests/pytorch/test_checkpoint.py --save-checkpoint all \
        || error_exit "Failed to generate checkpoint files"
fi

# ── Tests ───────────────────────────────────────────────────────────────────
# Each run_test call: env_prefix, xml_name, test_path, fail_label
# Tests are dispatched in waves of NUM_GPUS, one per GPU.

# DEBUG: inject a deliberate failure to test error capture (remove before merging)
run_test "" "pytest_debug_forced_fail.xml" "-c 'import pytest; pytest.fail(\"DELIBERATE FAILURE: testing parallel error capture\")'" "debug_forced_fail"

# DEBUG: inject a RuntimeError into test_sanity's GPU slot (remove before merging)
run_test "" "pytest_test_sanity.xml" "-c 'import torch; raise RuntimeError(f\"DELIBERATE ERROR: simulating OOM on GPU {torch.cuda.current_device()}\")'" "test_sanity.py_INJECTED"
run_test "" "pytest_test_recipe.xml" "$TE_PATH/tests/pytorch/test_recipe.py" "test_recipe.py"
run_test "" "pytest_test_deferred_init.xml" "$TE_PATH/tests/pytorch/test_deferred_init.py" "test_deferred_init.py"
run_test "PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0" "pytest_test_numerics.xml" "$TE_PATH/tests/pytorch/test_numerics.py" "test_numerics.py"
run_test "PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0" "pytest_test_cuda_graphs.xml" "$TE_PATH/tests/pytorch/test_cuda_graphs.py" "test_cuda_graphs.py"
run_test "" "pytest_test_jit.xml" "$TE_PATH/tests/pytorch/test_jit.py" "test_jit.py"
run_test "" "pytest_test_fused_rope.xml" "$TE_PATH/tests/pytorch/test_fused_rope.py" "test_fused_rope.py"
run_test "" "pytest_test_nvfp4.xml" "$TE_PATH/tests/pytorch/nvfp4" "test_nvfp4"
run_test "" "pytest_test_mxfp8.xml" "$TE_PATH/tests/pytorch/mxfp8" "test_mxfp8"
run_test "" "pytest_test_quantized_tensor.xml" "$TE_PATH/tests/pytorch/test_quantized_tensor.py" "test_quantized_tensor.py"
run_test "" "pytest_test_float8blockwisetensor.xml" "$TE_PATH/tests/pytorch/test_float8blockwisetensor.py" "test_float8blockwisetensor.py"
run_test "" "pytest_test_float8_blockwise_scaling_exact.xml" "$TE_PATH/tests/pytorch/test_float8_blockwise_scaling_exact.py" "test_float8_blockwise_scaling_exact.py"
run_test "" "pytest_test_float8_blockwise_gemm_exact.xml" "$TE_PATH/tests/pytorch/test_float8_blockwise_gemm_exact.py" "test_float8_blockwise_gemm_exact.py"
run_test "" "test_grouped_tensor.xml" "$TE_PATH/tests/pytorch/test_grouped_tensor.py" "test_grouped_tensor.py"
run_test "" "pytest_test_gqa.xml" "$TE_PATH/tests/pytorch/test_gqa.py" "test_gqa.py"
run_test "" "pytest_test_fused_optimizer.xml" "$TE_PATH/tests/pytorch/test_fused_optimizer.py" "test_fused_optimizer.py"
run_test "" "pytest_test_multi_tensor.xml" "$TE_PATH/tests/pytorch/test_multi_tensor.py" "test_multi_tensor.py"
run_test "" "pytest_test_fusible_ops.xml" "$TE_PATH/tests/pytorch/test_fusible_ops.py" "test_fusible_ops.py"
run_test "" "pytest_test_permutation.xml" "$TE_PATH/tests/pytorch/test_permutation.py" "test_permutation.py"
run_test "" "pytest_test_parallel_cross_entropy.xml" "$TE_PATH/tests/pytorch/test_parallel_cross_entropy.py" "test_parallel_cross_entropy.py"
run_test "" "pytest_test_cpu_offloading.xml" "$TE_PATH/tests/pytorch/test_cpu_offloading.py" "test_cpu_offloading.py"
run_test "NVTE_FLASH_ATTN=0 NVTE_CPU_OFFLOAD_V1=1" "pytest_test_cpu_offloading_v1.xml" "$TE_PATH/tests/pytorch/test_cpu_offloading_v1.py" "test_cpu_offloading_v1.py"
run_test "" "pytest_test_attention.xml" "$TE_PATH/tests/pytorch/attention/test_attention.py" "test_attention.py"
run_test "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0" "pytest_test_attention_deterministic.xml" "$TE_PATH/tests/pytorch/attention/test_attention.py" "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 test_attention.py"
run_test "" "pytest_test_kv_cache.xml" "$TE_PATH/tests/pytorch/attention/test_kv_cache.py" "test_kv_cache.py"
run_test "" "pytest_test_hf_integration.xml" "$TE_PATH/tests/pytorch/test_hf_integration.py" "test_hf_integration.py"
run_test "" "pytest_test_checkpoint.xml" "$TE_PATH/tests/pytorch/test_checkpoint.py" "test_checkpoint.py"
run_test "" "pytest_test_fused_router.xml" "$TE_PATH/tests/pytorch/test_fused_router.py" "test_fused_router.py"
run_test "" "pytest_test_partial_cast.xml" "$TE_PATH/tests/pytorch/test_partial_cast.py" "test_partial_cast.py"

# ── Wait for remaining background jobs ──────────────────────────────────────

if [ "$NUM_GPUS" -gt 1 ]; then
    wait
fi

# ── Replay per-test logs into trace ──────────────────────────────────────────

if [ "$NUM_GPUS" -gt 1 ]; then
    echo ""
    echo "=== Per-test output (replayed from parallel execution) ==="
    for logfile in "$XML_LOG_DIR"/*.log; do
        if [ -f "$logfile" ]; then
            echo ""
            echo "────────────────────────────────────────────────────────"
            echo ">>> $(basename "$logfile" .log)"
            echo "────────────────────────────────────────────────────────"
            cat "$logfile"
        fi
    done
    echo ""
    echo "=== End of per-test output ==="
fi

# ── Report results ──────────────────────────────────────────────────────────

if [ -s "$FAIL_DIR/failures" ]; then
    FAILED_CASES=$(cat "$FAIL_DIR/failures" | tr '\n' ' ')
    echo "Error in the following test cases: $FAILED_CASES"
    rm -rf "$FAIL_DIR"
    exit 1
fi
rm -rf "$FAIL_DIR"
echo "All tests passed"
exit 0
