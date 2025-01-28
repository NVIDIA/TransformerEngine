# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${THUNDER_PATH:=/opt/pytorch/lightning-thunder}

pip install pytest==8.1.1 pytest-benchmark==5.1.0
python3 -m pytest -v -s ${THUNDER_PATH}/thunder/tests/test_transformer_engine_executor.py
