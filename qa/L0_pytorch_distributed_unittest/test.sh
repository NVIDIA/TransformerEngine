# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

pip install pytest==7.2.0

: ${TE_PATH:=/opt/transformerengine}
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics.py
