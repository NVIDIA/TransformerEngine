# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}

python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/tests/jax/test_distributed_*
