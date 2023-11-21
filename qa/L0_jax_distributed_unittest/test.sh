# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
pytest -Wignore -v $TE_PATH/tests/jax/test_distributed_fused_attn.py
pytest -Wignore -v $TE_PATH/tests/jax/test_distributed_layernorm.py
pytest -Wignore -v $TE_PATH/tests/jax/test_distributed_softmax.py

# Make encoder tests to have run-to-run deterministic to have the stable CI results
pip install -r $TE_PATH/examples/jax/encoder/requirements.txt
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
pytest -Wignore -v $TE_PATH/tests/jax/test_distributed_encoder.py
