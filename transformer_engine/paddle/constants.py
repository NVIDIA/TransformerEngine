# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Constants"""
import paddle
import transformer_engine_paddle as tex
"""
Map from paddle dtype to TE dtype
"""
TE_DType = {
    paddle.uint8: tex.DType.kByte,
    paddle.int32: tex.DType.kInt32,
    paddle.float32: tex.DType.kFloat32,
    paddle.float16: tex.DType.kFloat16,
    paddle.bfloat16: tex.DType.kBFloat16,
}
