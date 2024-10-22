# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==8.2.1
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics.py
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops.py
pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn_with_cp.py
