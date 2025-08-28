# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


pip3 install onnxruntime==1.20.1
pip3 install onnxruntime_extensions==0.13.0

: ${TE_PATH:=/opt/transformerengine}

python3 -m pytest --tb=auto  $TE_PATH/tests/pytorch/test_onnx_export.py
