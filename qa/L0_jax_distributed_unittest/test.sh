# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

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

export NVTE_JAX_TEST_TIMING=1

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip3 install -r $TE_PATH/examples/jax/encoder/requirements.txt || error_exit "Failed to install requirements"

# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_multigpu_encoder.xml $TE_PATH/examples/jax/encoder/test_multigpu_encoder.py || test_fail "test_multigpu_encoder.py"
wait
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_model_parallel_encoder.xml $TE_PATH/examples/jax/encoder/test_model_parallel_encoder.py || test_fail "test_model_parallel_encoder.py"
wait
TE_PATH=$TE_PATH bash $TE_PATH/examples/jax/encoder/run_test_multiprocessing_encoder.sh || test_fail "run_test_multiprocessing_encoder.sh"
wait

TE_PATH=$TE_PATH bash $TE_PATH/examples/jax/collective_gemm/run_test_cgemm.sh || test_fail "run_test_cgemm.sh"
wait

# MoE custom_vjp distributed suite. Runs one Python process per GPU
# via tests/jax/run_multiprocess_moe_vjp.sh (mirrors the pattern in
# examples/jax/encoder/run_test_multiprocessing_encoder.sh). Requires
# >=4 visible GPUs.
TE_PATH=$TE_PATH bash $TE_PATH/tests/jax/run_multiprocess_moe_vjp.sh \
    || test_fail "test_multiprocess_moe_vjp.py"
# Exercise the multi-GPU tutorial in docs/examples/jax (needs >= 4 GPUs;
# auto-skips otherwise).
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_docs_examples_jax_distributed.xml -k multi_gpu $TE_PATH/docs/examples/jax/ || test_fail "docs/examples/jax (multi-GPU)"
wait

if [ $RET -ne 0 ]; then
    echo "Error: some sub-tests failed: $FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
