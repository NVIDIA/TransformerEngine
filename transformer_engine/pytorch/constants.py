# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    torch.float8_e4m3fn: tex.DType.kFloat8E4M3,
    torch.float8_e5m2: tex.DType.kFloat8E5M2,
    torch.int32: tex.DType.kInt32,
    torch.float32: tex.DType.kFloat32,
    torch.half: tex.DType.kFloat16,
    torch.bfloat16: tex.DType.kBFloat16,
}

"""
This is a map: int -> torch.dtype
Used for resolving cuda extension types to torch.
Has one to one mapping with enum in
transformer_engine.h
"""
TE_DType_To_Torch = {
    tex.DType.kByte: torch.uint8,
    tex.DType.kFloat8E4M3: torch.float8_e4m3fn,
    tex.DType.kFloat8E5M2: torch.float8_e5m2,
    tex.DType.kInt32: torch.int32,
    tex.DType.kFloat32: torch.float32,
    tex.DType.kFloat16: torch.half,
    tex.DType.kBFloat16: torch.bfloat16,
}

AttnMaskTypes = (
    "no_mask",
    "padding",
    "causal",
    "padding_causal",
    "causal_bottom_right",
    "padding_causal_bottom_right",
    "arbitrary",
)

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
    "sbhd_bshd_bshd",
    "bshd_sbhd_sbhd",
    "thd_bshd_bshd",
    "thd_sbhd_sbhd",
    "paged_kv_bshd_bshd_bshd",
    "paged_kv_bshd_sbhd_sbhd",
    "paged_kv_sbhd_bshd_bshd",
    "paged_kv_sbhd_sbhd_sbhd",
    "paged_kv_thd_bshd_bshd",
    "paged_kv_thd_sbhd_sbhd",
)

LayerTypes = ("encoder", "decoder")

GemmParallelModes = ("row", "column", None)

dist_group_type = torch.distributed.ProcessGroup

MXFP8_BLOCK_SCALING_SIZE = 32
