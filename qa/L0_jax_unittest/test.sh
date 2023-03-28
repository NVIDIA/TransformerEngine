# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
pytest -Wignore -v $TE_PATH/tests/jax

pip install -r $TE_PATH/examples/jax/mnist/requirements.txt
pip install -r $TE_PATH/examples/jax/encoder/requirements.txt
pytest -Wignore -v $TE_PATH/examples/jax
