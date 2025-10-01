# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


pip3 install onnxruntime
pip3 install onnxruntime_extensions

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/test_onnx_export.xml $TE_PATH/tests/pytorch/test_onnx_export.py
