# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}

pip install -r $TE_PATH/examples/jax/encoder/requirements.txt

# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder/test_multigpu_encoder.py
pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder/test_model_parallel_encoder.py
. $TE_PATH/examples/jax/encoder/run_test_multiprocessing_encoder.sh
