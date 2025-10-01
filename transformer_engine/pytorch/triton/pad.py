# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVFP4 padding kernels

TODO(ksivamani): Documentation

"""

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=1),
    ],
    key=["out_dim0", "out_dim1"],
)
@triton.jit
def zero_pad_kernel(
    inp_ptr,
    out_ptr,
    in_dim0: tl.constexpr,
    in_dim1: tl.constexpr,
    out_dim0: tl.constexpr,
    out_dim1: tl.constexpr,
    in_s0,
    in_s1,
    out_s0,
    out_s1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Pads a tensor assuming it's a columnwise scaling inverse."""

    # tile over OUTPUT coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output cols
    om = offs_m[:, None]
    on = offs_n[None, :]

    # edge masking for output
    out_mask = (om < out_dim0) & (on < out_dim1)

    # valid input region is simply top-left (no offsets)
    in_mask = (om < in_dim0) & (on < in_dim1)

    # load valid input, else zero (masked load touches memory only where True)
    x = tl.load(inp_ptr + om * in_s0 + on * in_s1, mask=in_mask, other=0)

    # store to output (only within bounds of the output tile)
    tl.store(out_ptr + om * out_s0 + on * out_s1, x, mask=out_mask)


def pad_columnwise_scale_inv(inp: torch.Tensor) -> torch.Tensor:
    """Pads a tensor assuming it's a columnwise scaling inverse."""

    assert inp.ndim == 2
    dim0, dim1 = inp.shape

    pad_x = (128 - dim0 % 128) % 128
    pad_y = (4 - dim1 % 4) % 4
    out_x = dim0 + pad_x
    out_y = dim1 + pad_y
    out = torch.empty((out_x, out_y), device=inp.device, dtype=inp.dtype)

    in_s0, in_s1 = inp.stride()
    out_s0, out_s1 = out.stride()

    BLOCK_M, BLOCK_N = 128, 128
    grid = (triton.cdiv(out_x, BLOCK_M), triton.cdiv(out_y, BLOCK_N))

    zero_pad_kernel[grid](
        inp,
        out,
        dim0,
        dim1,
        out_x,
        out_y,
        in_s0,
        in_s1,
        out_s0,
        out_s1,
    )
    return out
