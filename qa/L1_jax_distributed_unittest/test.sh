# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

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

export NVTE_JAX_UNITTEST_LEVEL="L1"

# Use --xla_gpu_enable_triton_gemm=false to ensure the reference JAX implementation we are using is accurate.
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_triton_gemm=false"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_dist_dense.xml $TE_PATH/tests/jax/test_distributed_dense.py || test_fail "test_distributed_dense.py"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_helper.xml $TE_PATH/tests/jax/test_distributed_helper.py || test_fail "test_distributed_helper.py"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_dist_layernorm.xml $TE_PATH/tests/jax/test_distributed_layernorm.py || test_fail "test_distributed_layernorm.py"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_dist_mlp.xml $TE_PATH/tests/jax/test_distributed_layernorm_mlp.py || test_fail "test_distributed_layernorm_mlp.py"

# XLA_FLAGS to WAR for test_distributed_softmax issue with NCCL
# TODO(Kshitij): remove when NCCL issue is fixed
XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_nccl_comm_splitting=false" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_dist_softmax.xml $TE_PATH/tests/jax/test_distributed_softmax.py || test_fail "test_distributed_softmax.py"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_dist_fused_attn.xml $TE_PATH/tests/jax/test_distributed_fused_attn.py || test_fail "test_distributed_fused_attn.py"

# TODO(Phuong): add this test back after it is verified
# SCRIPT_NAME=$TE_PATH/tests/jax/test_multi_process_distributed_grouped_gemm.py bash $TE_PATH/tests/jax/multi_process_launch.sh || test_fail "test_multi_process_distributed_grouped_gemm.py"

if [ $RET -ne 0 ]; then
    echo "Error: some sub-tests failed: $FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
