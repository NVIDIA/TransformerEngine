# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
pytest -Wignore -v $TE_PATH/tests/jax -k 'not distributed'

pip install -r $TE_PATH/examples/jax/mnist/requirements.txt
pip install -r $TE_PATH/examples/jax/encoder/requirements.txt

pytest -Wignore -v $TE_PATH/examples/jax/mnist

# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
pytest -Wignore -v $TE_PATH/examples/jax/encoder --ignore=$TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py
pytest -Wignore -v $TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py
