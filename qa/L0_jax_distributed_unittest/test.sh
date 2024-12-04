# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

pip install -r $TE_PATH/examples/jax/encoder/requirements.txt

pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/mnist

# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder --ignore=$TE_PATH/examples/jax/encoder/test_single_gpu_encoder.py
