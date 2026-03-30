#!/bin/bash
set -x
TE_PATH=/workspace
RESULTS=/workspace/l1_results.txt
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
    "$@" 2>&1 | tee /tmp/l1_test_${name}.log | tail -5 >> "$RESULTS"
    echo "EXIT=$?" >> "$RESULTS"
    echo "" >> "$RESULTS"
}

run_test "dist_test_sanity" python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_sanity.py
run_test "dist_test_numerics" python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics.py
run_test "dist_test_numerics_exact" python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics_exact.py
run_test "dist_test_fusible_ops" python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops.py
run_test "dist_test_torch_fsdp2" python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py
run_test "dist_test_comm_gemm_overlap" python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py
run_test "dist_test_fusible_ops_with_userbuffers" python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops_with_userbuffers.py
run_test "dist_test_attention_with_cp" python3 -m pytest -v -s $TE_PATH/tests/pytorch/attention/test_attention_with_cp.py
run_test "dist_test_cp_utils" python3 -m pytest -v -s $TE_PATH/tests/pytorch/attention/test_cp_utils.py
run_test "dist_test_cast_master_weights_to_fp8" python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_cast_master_weights_to_fp8.py

echo "=== ALL DONE ===" >> "$RESULTS"
cat "$RESULTS"
