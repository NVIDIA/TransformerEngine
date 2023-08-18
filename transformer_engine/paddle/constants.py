# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Constants"""

from enum import Enum

import paddle

import transformer_engine_paddle as tex


class FP8FwdTensors(Enum):
    """Used as named indices on the `scale`, `scale_inv`,
    and `amax` tensors in the `FP8TensorMeta` class."""
    GEMM1_INPUT = 0
    GEMM1_WEIGHT = 1
    GEMM1_OUTPUT = 2
    GEMM2_INPUT = 3
    GEMM2_WEIGHT = 4
    GEMM2_OUTPUT = 5


class FP8BwdTensors(Enum):
    """Used as named indices on the `scale`, `scale_inv`,
    and `amax` tensors in the `FP8TensorMeta` class."""
    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
    GRAD_OUTPUT2 = 2
    GRAD_INPUT2 = 3


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

AttnMaskTypes = ("causal", "padding", "no_mask")

AttnTypes = ("self", "cross")

LayerTypes = ("encoder", "decoder")
