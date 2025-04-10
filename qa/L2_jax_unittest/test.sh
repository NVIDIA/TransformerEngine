# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -x

source $(dirname "$0")/../test_utils.sh
initialize_test_variables

pip3 install "nltk>=3.8.2" || error_exit "Failed to install nltk"
pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"
: ${TE_PATH:=/opt/transformerengine}

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=/logs/pytest_jax_not_distributed.xml $TE_PATH/tests/jax -k 'not distributed' --ignore=$TE_PATH/tests/jax/test_praxis_layers.py || test_fail "tests/jax/*not_distributed_*"

# Test without custom calls
NVTE_JAX_UNITTEST_LEVEL="L2" NVTE_CUSTOM_CALLS_RE="" python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=/logs/pytest_test_custom_call_compute.xml $TE_PATH/tests/jax/test_custom_call_compute.py || test_fail "test_custom_call_compute.py"

pip3 install -r $TE_PATH/examples/jax/mnist/requirements.txt || error_exit "Failed to install mnist requirements"
pip3 install -r $TE_PATH/examples/jax/encoder/requirements.txt || error_exit "Failed to install encoder requirements"

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=/logs/pytest_mnist.xml $TE_PATH/examples/jax/mnist || test_fail "mnist"

# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=/logs/pytest_test_single_gpu_encoder.xml $TE_PATH/examples/jax/encoder/test_single_gpu_encoder.py || test_fail "test_single_gpu_encoder.py"

check_test_results
