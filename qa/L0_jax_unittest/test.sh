# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

pip install "nltk>=3.8.2"
pip install pytest==8.2.1
: ${TE_PATH:=/opt/transformerengine}

pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax -k 'not distributed'

# Test without custom calls
NVTE_CUSTOM_CALLS_RE="" pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax/test_custom_call_compute.py

pip install -r $TE_PATH/examples/jax/mnist/requirements.txt
pip install -r $TE_PATH/examples/jax/encoder/requirements.txt

pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/mnist

pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder --ignore=$TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py
pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py
