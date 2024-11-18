# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==8.2.1 onnxruntime==1.19.2

# Build custom ONNX Runtime operators
export CUSTOM_ORT_OPS_PATH=$TE_PATH/tests/pytorch/custom_ort_ops
bash $CUSTOM_ORT_OPS_PATH/build.sh

# Run tests
NVTE_TORCH_COMPILE=0 pytest -v -s $TE_PATH/tests/pytorch/test_onnx_export.py
