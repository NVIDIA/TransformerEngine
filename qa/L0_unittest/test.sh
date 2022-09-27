# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==6.2.5
pytest -v -s $TE_PATH/tests/test_transformerengine.py
