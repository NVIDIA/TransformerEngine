# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

set -xe

export NVTE_JAX_TEST_TIMING=1

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# Use --xla_gpu_enable_triton_gemm=false to ensure the reference JAX implementation we are using is accurate.
common_xla_flags="$XLA_FLAGS --xla_gpu_enable_triton_gemm=false"

distributed_tests=()
for test_file in $TE_PATH/tests/jax/test_distributed_*.py; do
    if [ "$test_file" != "$TE_PATH/tests/jax/test_distributed_softmax.py" ]; then
        distributed_tests+=("$test_file")
    fi
done
XLA_FLAGS="$common_xla_flags" NVTE_JAX_UNITTEST_LEVEL="L2" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml "${distributed_tests[@]}"

# Work around the XLA 26.08 generic async all-reduce deadlock (openxla/xla#46938).
XLA_FLAGS="$common_xla_flags --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_disable_async_collectives=ALLREDUCE" NVTE_JAX_UNITTEST_LEVEL="L2" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_dist_softmax.xml $TE_PATH/tests/jax/test_distributed_softmax.py

# NCCL EP multi-process suite. The launcher skips when fewer than 4 GPUs or no NVLink is detected.
TE_PATH=$TE_PATH bash $TE_PATH/tests/jax/multi_process_launch_ep.sh
