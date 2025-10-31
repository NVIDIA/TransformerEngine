# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# Use --xla_gpu_enable_triton_gemm=false to ensure the reference JAX implementation we are using is accurate.
XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_triton_gemm=false" NVTE_JAX_UNITTEST_LEVEL="L2" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_*
