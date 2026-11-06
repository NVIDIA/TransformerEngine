# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    run_one_case(0, 64, 64, 0)
    run_one_case(3, 32, 64, 31)
    run_one_case(64 * 64 - 2, 64, 64, 1)
    run_one_case(16384 * 6144, 16384, 6144, 0)
    run_one_case(32768, 256, 128, 0)
    run_one_case(131072, 768, 256, 0)
    run_one_case(65536, 768, 256, 131072)
    run_one_case(98304, 128, 768, 0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("offset,length", [(0, 0), (31, 3), (1, 4094)])
@pytest.mark.parametrize("auxiliary_scales", ["empty", "present"])
def test_mxfp8_partial_cast_rowwise_output(offset, length, auxiliary_scales):
    """Empty columnwise output selects row-only work, regardless of auxiliary scales."""
    inp = torch.randn(length, device="cuda", dtype=torch.bfloat16)
    scales = torch.full((128, 4), 127, dtype=torch.uint8, device="cuda")
    output = torch.empty(length, dtype=torch.uint8, device="cuda")
    backing = torch.empty(8, dtype=torch.uint8, device="cuda")
    omitted_output = backing[:0]
    if auxiliary_scales == "empty":
        columnwise_scales = backing[:0].view(0, 0)
    else:
        # Deliberately invalid scale metadata must not enable columnwise casting.
        columnwise_scales = torch.ones(3, dtype=torch.float32, device="cuda")
    tex.mxfp8_scaling_partial_cast(
        inp, output, omitted_output, scales, columnwise_scales, 64, 64, offset
    )
    expected = inp.to(torch.float8_e4m3fn).view(torch.uint8)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_mxfp8_partial_cast_requires_scales_for_columnwise_output():
    inp = torch.ones(3, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(3, dtype=torch.uint8, device="cuda")
    columnwise = torch.empty_like(output)
    scales = torch.full((128, 4), 127, dtype=torch.uint8, device="cuda")
    missing_scales = torch.empty((0, 0), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="Columnwise output requires columnwise scales"):
        tex.mxfp8_scaling_partial_cast(inp, output, columnwise, scales, missing_scales, 64, 64, 31)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("h,w,start_offset,n", [(132, 130, 126, 395), (128, 128, 0, 16384)])
def test_block_scaling_partial_cast(input_dtype, output_dtype, h, w, start_offset, n):
    """The adapter supplies typed output and encoding scales to the common API."""
    inp = torch.linspace(-3, 3, n, device="cuda", dtype=input_dtype)
    # Exercise the adapter's handling of the strided scale matrices supported by its API.
    scale = torch.arange(1, 5, device="cuda", dtype=torch.float32).reshape(2, 2).T
    # Block FP8 stores can cover an overlapping tile outside the local shard.
    storage = torch.empty(h * w, device="cuda", dtype=torch.uint8)
    output = storage[start_offset : start_offset + n]
    dtype = tex.DType.kFloat8E4M3 if output_dtype == torch.float8_e4m3fn else tex.DType.kFloat8E5M2
    tex.fp8_block_scaling_partial_cast(inp, output, scale, h, w, start_offset, 128, dtype)
    indices = torch.arange(start_offset, start_offset + n, device="cuda")
    multipliers = scale[(indices // w) // 128, (indices % w) // 128]
    expected = (inp.float() * multipliers).to(output_dtype).view(torch.uint8)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not te.is_nvfp4_available(), reason="NVFP4 requires supported GPU and toolkit")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("start_offset,n", [(0, 2048), (30, 1043)])
def test_nvfp4_partial_cast_encoded_scales(input_dtype, batched, start_offset, n):
    """Cast with expanded encoded scales, including a shard crossing tile rows."""
    h, w = 32, 64
    inp = torch.linspace(-5, 5, n, device="cuda", dtype=input_dtype)
    tile_scales = torch.tensor([[1, 2, 4, 1], [2, 4, 1, 2]], device="cuda", dtype=torch.float32)
    scales = tile_scales.repeat_interleave(16, dim=0).to(torch.float8_e4m3fn).view(torch.uint8)
    global_scale = torch.tensor([2.0], device="cuda")
    # The final high nibble is outside an odd-length shard and must be preserved.
    output = torch.full(((n + 1) // 2,), 0xA5, device="cuda", dtype=torch.uint8)
    if batched:
        tex.nvfp4_multi_tensor_2d_partial_cast(
            [inp], [output], [scales], [global_scale], [h], [w], [start_offset], 16
        )
    else:
        tex.nvfp4_2d_partial_cast(inp, output, scales, global_scale, h, w, start_offset, 16)
    indices = torch.arange(start_offset, start_offset + n, device="cuda")
    inverse_scales = tile_scales[(indices // w) // 16, (indices % w) // 16]
    scaled = inp.float() * 2.0 / inverse_scales
    # Nearest-level FP4 reference with round-to-nearest-even ties.
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    distances = (scaled.abs().unsqueeze(1) - levels).abs()
    nearest = distances.argmin(dim=1)
    ties = (distances == distances.min(dim=1, keepdim=True).values).sum(dim=1) > 1
    nearest = torch.where(ties & (nearest % 2 == 1), nearest + 1, nearest)
    expected = (nearest | (torch.signbit(scaled).long() << 3)).to(torch.uint8)
    actual = torch.stack((output & 0xF, output >> 4), dim=1).flatten()[:n]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if n % 2:
        assert (output[-1] >> 4).item() == 0xA


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("offset,length", [(0, 0), (31, 3), (1, 4094)])
@pytest.mark.parametrize("with_columnwise_scales", [False, True])
def test_rowwise_partial_kernels(offset, length, with_columnwise_scales):
    inp = torch.randn(length, device="cuda", dtype=torch.bfloat16)
    row = torch.zeros((128, 4), device="cuda", dtype=inp.dtype)
    ref = torch.zeros_like(row)
    col = torch.zeros((4, 128), device="cuda", dtype=inp.dtype)
    # The PyTorch extension must normalize empty views with backing storage.
    omitted = col.flatten()[:0].view(0, 0)
    tex.mxfp8_scaling_compute_partial_amax(inp, row, omitted, 64, 64, offset)
    tex.mxfp8_scaling_compute_partial_amax(inp, ref, col, 64, 64, offset)
    torch.testing.assert_close(row, ref, rtol=0, atol=0)
    scales = torch.full((128, 4), 127, dtype=torch.uint8, device="cuda")
    columnwise_scales = (
        torch.full((4, 128), 126, dtype=torch.uint8, device="cuda")
        if with_columnwise_scales
        else scales.flatten()[:0].view(0, 0)
    )
    out = torch.empty(length, dtype=torch.uint8, device="cuda")
    tex.mxfp8_scaling_partial_cast(inp, out, out[:0], scales, columnwise_scales, 64, 64, offset)
    torch.testing.assert_close(out, inp.to(torch.float8_e4m3fn).view(torch.uint8), rtol=0, atol=0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_mxfp8_partial_cast_requires_columnwise_scales():
    inp = torch.randn(3, device="cuda", dtype=torch.bfloat16)
    row = torch.empty(3, device="cuda", dtype=torch.uint8)
    col = torch.empty_like(row)
    scales = torch.full((128, 4), 127, device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="Columnwise output requires columnwise scales"):
        tex.mxfp8_scaling_partial_cast(
            inp, row, col, scales, scales.flatten()[:0].view(0, 0), 64, 64, 31
        )
