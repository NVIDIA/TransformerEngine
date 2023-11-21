# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
pip install -r $TE_PATH/examples/jax/mnist/requirements.txt
pytest -Wignore -v $TE_PATH/tests/jax \
    --ignore="$TE_PATH/tests/jax/test_encoder.py" \
    --ignore-glob="$TE_PATH/tests/jax/test_distributed_*.py"

# Make encoder tests to have run-to-run deterministic to have the stable CI results
pip install -r $TE_PATH/examples/jax/encoder/requirements.txt
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
pytest -Wignore -v $TE_PATH/tests/jax/test_encoder.py
