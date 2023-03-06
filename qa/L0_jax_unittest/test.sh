# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
pytest -Wignore -v $TE_PATH/tests/jax
pytest -Wignore -v $TE_PATH/examples/jax
