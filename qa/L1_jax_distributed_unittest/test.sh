# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_layernorm*
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_softmax.py
# Run partial distributed fused attn tests in L1
# TestReorderCausalLoadBalancing: Run only one (non symmetric) BSHD/SBHD data shape combination
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_fused_attn.py -k "TestReorderCausalLoadBalancing and 3-32-8-64"
# TestDistributedSelfAttn: Run only one (larger) BSHD type data shape combination
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_fused_attn.py -k "TestDistributedSelfAttn and 32-1024-16-128"
# TestDistributedCrossAttn: Run only one (larger) BSHD type data shape combination
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_fused_attn.py -k "TestDistributedCrossAttn and data_shape1"
# TestDistributedContextParallelSelfAttn: Run only non cp1 combinations
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest.xml $TE_PATH/tests/jax/test_distributed_fused_attn.py -k "TestDistributedContextParallelSelfAttn and not cp1"