# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# NVTE_UnfusedDPA_Emulate_FP8=1 enables FP8 attention emulation when no native backend is available
NVTE_UnfusedDPA_Emulate_FP8=1 python3 -m pytest --tb=auto --junitxml=$XML_LOG_DIR/test_onnx_export.xml $TE_PATH/tests/pytorch/test_onnx_export.py
