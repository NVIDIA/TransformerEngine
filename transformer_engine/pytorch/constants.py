# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Enums for e2e transformer"""
import torch
import torch.distributed
import transformer_engine_torch as tex


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

AttnMaskTypes = ("causal", "padding", "padding_causal", "arbitrary", "no_mask")

AttnTypes = ("self", "cross")

AttnBiasTypes = ("pre_scale_bias", "post_scale_bias", "no_bias", "alibi")

QKVLayouts = (
    "sb3hd",
    "sbh3d",
    "sbhd_sb2hd",
    "sbhd_sbh2d",
    "sbhd_sbhd_sbhd",
    "bs3hd",
    "bsh3d",
    "bshd_bs2hd",
    "bshd_bsh2d",
    "bshd_bshd_bshd",
    "t3hd",
    "th3d",
    "thd_t2hd",
    "thd_th2d",
    "thd_thd_thd",
)

LayerTypes = ("encoder", "decoder")

GemmParallelModes = ("row", "column", None)

dist_group_type = torch.distributed.ProcessGroup
