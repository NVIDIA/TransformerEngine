# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

NVTE_JAX_UNITTEST_LEVEL="L1" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_*
SCRIPT_NAME=$TE_PATH/tests/jax/test_multi_process_distributed_grouped_gemm.py bash $TE_PATH/tests/jax/multi_process_launch.sh
