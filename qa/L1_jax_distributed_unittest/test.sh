# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
pytest -c $TE_PATH/tests/jax/pytest.ini -Wignore -v $TE_PATH/tests/jax/test_distributed_*

