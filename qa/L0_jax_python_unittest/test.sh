# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}

pip install flax@git+https://github.com/google/flax#egg=flax
pytest -Wignore -v $TE_PATH/tests/jax
