# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==6.2.5 onnxruntime==1.13.1
pytest -v -s $TE_PATH/tests/test_transformerengine.py $TE_PATH/tests/test_fp8.py
NVTE_FLASH_ATTN=0 pytest -v -s $TE_PATH/tests/test_onnx_export.py
