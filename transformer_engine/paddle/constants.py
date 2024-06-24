# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Constants"""

from enum import Enum

import paddle

from transformer_engine import transformer_engine_paddle as tex


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

GemmParallelModes = ("row", "column", None)

dist_group_type = paddle.distributed.collective.Group

RecomputeFunctionNames = ("unpack", "backward")

AttnBiasType = {
    "no_bias": tex.NVTE_Bias_Type.NVTE_NO_BIAS,
    "pre_scale_bias": tex.NVTE_Bias_Type.NVTE_PRE_SCALE_BIAS,
    "post_scale_bias": tex.NVTE_Bias_Type.NVTE_POST_SCALE_BIAS,
}

AttnMaskType = {
    "no_mask": tex.NVTE_Mask_Type.NVTE_NO_MASK,
    "padding": tex.NVTE_Mask_Type.NVTE_PADDING_MASK,
    "causal": tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
}

FusedAttnBackend = {
    "F16_max512_seqlen": tex.NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen,
    "F16_arbitrary_seqlen": tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
    "No_Backend": tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend,
}
