# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}

# Skip ring attention tests since they need fixed environment vars
pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax/test_distributed_* -k 'not test_context_parallel_ring_attn'

# Test ring attention with and without scan loop
NVTE_FUSED_RING_ATTENTION_USE_SCAN=0 pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax/test_distributed_fused_attn.py -k test_context_parallel_ring_attn
NVTE_FUSED_RING_ATTENTION_USE_SCAN=1 XLA_FLAGS="--xla_experimental_ignore_channel_id" \
    pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax/test_distributed_fused_attn.py -k test_context_parallel_ring_attn