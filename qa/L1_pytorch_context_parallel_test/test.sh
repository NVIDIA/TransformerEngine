# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==6.2.5 onnxruntime==1.13.1
pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn_with_cp.py
