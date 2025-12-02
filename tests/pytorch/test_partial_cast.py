# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine_torch import multi_tensor_compute_scale_inv_e8m0
from transformer_engine.pytorch import is_mxfp8_available
from transformer_engine.pytorch.optimizers.multi_tensor_apply import multi_tensor_applier


mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)


def compute_partial_amax_reference(inp, amax_rowwise, amax_colwise, h, w, start_offset):
    n = inp.view(-1).size(0)
    if n == h * w:
        full = inp.view(-1)
    else:
        full = torch.zeros(h * w, dtype=inp.dtype, device=inp.device)
        full[start_offset : start_offset + n].copy_(inp)
    full = torch.abs(full)
    _amax_rowwise, _ = torch.max(full.view(h, w // 32, 32), dim=2)
    amax_rowwise[:h, : (w // 32)].copy_(_amax_rowwise)
    _amax_colwise, _ = torch.max(full.view(h // 32, 32, w), dim=1)
    amax_colwise[: (h // 32), :w].copy_(_amax_colwise)


def partial_cast_reference(
    inp, rowwise_out, colwise_out, rowwise_inv_scale, colwise_inv_scale, h, w, start_offset
):
    rowwise_scale = ((254 - rowwise_inv_scale.int()) * 2**23).view(torch.float32)
    colwise_scale = ((254 - colwise_inv_scale.int()) * 2**23).view(torch.float32)
    n = inp.view(-1).size(0)
    if n == h * w:
        full = inp
    else:
        full = torch.empty(h * w, dtype=inp.dtype, device=inp.device)
        full[start_offset : start_offset + n].copy_(inp)
    full = full.float()
    rowwise_scale = rowwise_scale[:h, : (w // 32)].contiguous().float()
    colwise_scale = colwise_scale[: (h // 32), :w].contiguous().float()
    scaled = (full.view(-1, 32) * rowwise_scale.view(-1, 1)).view(-1)
    rowwise_out.copy_(
        scaled[start_offset : start_offset + n].to(torch.float8_e4m3fn).view(rowwise_out.dtype)
    )
    scaled = (full.view(h // 32, 32, w) * colwise_scale.view(h // 32, 1, w)).view(-1)
    colwise_out.copy_(
        scaled[start_offset : start_offset + n].to(torch.float8_e4m3fn).view(colwise_out.dtype)
    )


def run_one_case(n, h, w, start_offset):
    inp = torch.randn(n, dtype=torch.bfloat16, device="cuda")

    rowwise_padding = [128, 4]
    colwise_padding = [4, 128]

    def _pad(x, padding):
        return (x + padding - 1) // padding * padding

    rowwise_shape = [_pad(h, rowwise_padding[0]), _pad(w // 32, rowwise_padding[1])]
    colwise_shape = [_pad(h // 32, colwise_padding[0]), _pad(w, colwise_padding[1])]

    # Partial amax cuda kernel
    amax_rowwise = torch.zeros(*rowwise_shape, dtype=inp.dtype, device=inp.device)
    amax_colwise = torch.zeros(*colwise_shape, dtype=inp.dtype, device=inp.device)
    tex.mxfp8_scaling_compute_partial_amax(inp, amax_rowwise, amax_colwise, h, w, start_offset)

    # Partial amax pytorch reference
    amax_rowwise_ref = torch.zeros(*rowwise_shape, dtype=inp.dtype, device=inp.device)
    amax_colwise_ref = torch.zeros(*colwise_shape, dtype=inp.dtype, device=inp.device)
    compute_partial_amax_reference(inp, amax_rowwise_ref, amax_colwise_ref, h, w, start_offset)

    # Check partial amax
    torch.testing.assert_close(amax_rowwise, amax_rowwise_ref, atol=0, rtol=0)
    torch.testing.assert_close(amax_colwise, amax_colwise_ref, atol=0, rtol=0)

    # Calculate scales and scale_invs
    scale_inv_rowwise = torch.empty_like(amax_rowwise).to(torch.uint8)
    scale_inv_colwise = torch.empty_like(amax_colwise).to(torch.uint8)
    multi_tensor_applier(
        multi_tensor_compute_scale_inv_e8m0,
        None,
        [
            [amax_rowwise, amax_colwise],
            [scale_inv_rowwise, scale_inv_colwise],
        ],
    )

    # Partial cast cuda kernel
    output_rowwise = torch.empty_like(inp).to(torch.uint8)
    output_colwise = torch.empty_like(inp).to(torch.uint8)
    tex.mxfp8_scaling_partial_cast(
        inp,
        output_rowwise,
        output_colwise,
        scale_inv_rowwise,
        scale_inv_colwise,
        h,
        w,
        start_offset,
    )

    # Partial cast pytorch reference
    output_rowwise_ref = torch.empty_like(inp).to(torch.uint8)
    output_colwise_ref = torch.empty_like(inp).to(torch.uint8)
    partial_cast_reference(
        inp,
        output_rowwise_ref,
        output_colwise_ref,
        scale_inv_rowwise,
        scale_inv_colwise,
        h,
        w,
        start_offset,
    )

    # Check partial cast results
    torch.testing.assert_close(output_rowwise, output_rowwise_ref, atol=0, rtol=0)
    torch.testing.assert_close(output_colwise, output_colwise_ref, atol=0, rtol=0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_mxfp8_scaling_partial_cast():
    torch.cuda.manual_seed(1234)

    run_one_case(3, 32, 64, 31)
    run_one_case(64 * 64 - 2, 64, 64, 1)
    run_one_case(16384 * 6144, 16384, 6144, 0)
    run_one_case(32768, 256, 128, 0)
    run_one_case(131072, 768, 256, 0)
    run_one_case(65536, 768, 256, 131072)
    run_one_case(98304, 128, 768, 0)
