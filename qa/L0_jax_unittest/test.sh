# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

pip3 install "nltk>=3.8.2"
pip3 install pytest==8.2.1
: ${TE_PATH:=/opt/transformerengine}

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax -k 'not distributed' --ignore=$TE_PATH/tests/jax/test_praxis_layers.py

# Test without custom calls
NVTE_CUSTOM_CALLS_RE="" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax/test_custom_call_compute.py

pip3 install -r $TE_PATH/examples/jax/mnist/requirements.txt
pip3 install -r $TE_PATH/examples/jax/encoder/requirements.txt

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/mnist

# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder/test_single_gpu_encoder.py
