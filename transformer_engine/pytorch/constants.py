# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Enums for e2e transformer"""
import torch
import torch.distributed
import transformer_engine_extensions as tex


"""
This is a map: torch.dtype -> int
Used for passing dtypes into cuda
extension. Has one to one mapping
with enum in transformer_engine.h
"""
TE_DType = {
    torch.uint8: tex.DType.kByte,
    torch.int32: tex.DType.kInt32,
    torch.float32: tex.DType.kFloat32,
    torch.half: tex.DType.kFloat16,
    torch.bfloat16: tex.DType.kBFloat16,
}

AttnMaskTypes = ("causal", "padding", "no_mask")

AttnTypes = ("self", "cross")

AttnBiasTypes = ("pre_scale_bias", "post_scale_bias", "no_bias")

LayerTypes = ("encoder", "decoder")

GemmParallelModes = ("row", "column", None)

dist_group_type = torch.distributed.ProcessGroup
