# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Validate compact-input requantization against independent PyTorch arithmetic."""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
cuda = pytest.importorskip("cuda.bindings.driver")

from cutlass.cute.runtime import from_dlpack
from transformer_engine.common.CuTeDSL.cast.mxfp8 import requantize_mxfp8
from transformer_engine.common.CuTeDSL.cast.mxfp8.requantize_mxfp8 import (
    GroupedRequantize,
    GroupedRequantizeWarpTile,
)
from mxfp8_utils import swizzle_mxfp8_scale

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="Grouped MXFP8 requantization requires SM100+",
)


@pytest.fixture(params=["auto", "portable"])
def decode_backend(request, monkeypatch):
    """Exercise the old-compiler sequence even when the compiler supports PTX 9.2."""
    if request.param == "portable":
        monkeypatch.setattr(requantize_mxfp8, "_supports_scaled_bf16_conversion", lambda: False)
        monkeypatch.setattr(requantize_mxfp8, "_supports_bf16x2_to_e4m3_conversion", lambda: False)


def columnwise_reference(values):
    """Quantize BF16-decoded values in independent 32-row blocks."""
    rows, hidden = values.shape
    blocks = values.float().reshape(rows // 32, 32, hidden)
    amax = torch.where(torch.isnan(blocks), 0.0, blocks.abs()).amax(1)
    exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
    exponent = torch.where(amax == 0, -127, exponent)
    inverse = torch.exp2(-exponent).clamp(max=torch.finfo(torch.float32).max)
    quantized = (blocks * inverse[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.reshape(rows, hidden).view(torch.uint8), (exponent + 127).to(torch.uint8)


@pytest.mark.parametrize("warp_tile", [False, True])
@pytest.mark.parametrize("return_dequantized", [False, True])
@pytest.mark.parametrize("hidden", [128, 256, 384, 512, 768, 4096, 7168])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("layout", ["compact", "column_only", "both", "uniform", "many_groups"])
@pytest.mark.usefixtures("decode_backend")
def test_compact_input_contract(hidden, input_dtype, layout, return_dequantized, warp_tile):
    """Check both scale layouts, optional output, group boundaries and capacity tails."""
    if warp_tile and (layout == "uniform" or return_dequantized):
        pytest.skip("The warp-tile kernel serves varying first dims without BF16 output")
    if not warp_tile and hidden % 512 and hidden > 512:
        pytest.skip("The TMA kernel requires a 16-byte scale row stride")
    torch.manual_seed(42)
    rows = 1024
    sizes = [256] * 4 if layout == "uniform" else [128, 0, 256, 128, 0]
    if layout == "many_groups":
        # Empty groups on both sides of index 32 and at the end, and a capacity tail.
        rows = 4096
        sizes = [0 if i % 2 or i in (31, 32, 33) or i >= 45 else 128 for i in range(48)]
    live = sum(sizes)
    data = (torch.randn(rows, hidden, device="cuda") * 8).to(input_dtype)
    scales = torch.randint(100, 155, (rows, hidden // 32), device="cuda", dtype=torch.uint8)
    decoded = (data.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)).to(
        torch.bfloat16
    )
    offsets = torch.tensor([0] + sizes, device="cuda", dtype=torch.int64).cumsum(0) * hidden
    dequantized = torch.full((rows, hidden), -123, device="cuda", dtype=torch.bfloat16)
    dst = torch.full((rows, hidden), 0xA5, device="cuda", dtype=torch.uint8)
    row_sf = torch.full_like(scales.flatten(), 0xA5)
    col_sf = torch.full_like(row_sf, 0xA5)
    swizzled = layout != "compact"
    row_output = layout in ("both", "uniform", "many_groups")
    dtype = cutlass.Float8E4M3FN if input_dtype == torch.float8_e4m3fn else cutlass.Float8E5M2
    if warp_tile:
        kernel = GroupedRequantizeWarpTile(
            hidden, input_dtype=dtype, swizzled=swizzled, rowwise_output=row_output
        )
    else:
        kernel = GroupedRequantize(
            hidden,
            len(sizes),
            rows,
            input_dtype=dtype,
            swizzled=swizzled,
            rowwise_output=row_output,
            return_dequantized=return_dequantized,
            uniform_rows=256 if layout == "uniform" else 0,
            sm_count=torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count,
        )
    tensors = (
        data,
        scales,
        None if layout == "uniform" else offsets,
        dst.view(torch.float8_e4m3fn),
        row_sf if row_output else None,
        col_sf,
        dequantized if return_dequantized else None,
    )
    args = tuple(
        from_dlpack(tensor, assumed_align=16) if tensor is not None else None for tensor in tensors
    ) + (cuda.CUstream(torch.cuda.current_stream().cuda_stream),)
    compiled = cute.compile(kernel, *args)
    compiled(*args)
    torch.cuda.synchronize()
    start = 0
    expected_data, expected_sf = [], []
    for size in sizes:
        if size:
            quantized, sf = columnwise_reference(decoded[start : start + size])
            expected_data.append(quantized.flatten())
            if swizzled:
                sf = swizzle_mxfp8_scale(size, hidden, sf, True)
            expected_sf.append(sf.flatten())
        start += size
    torch.testing.assert_close(
        dst.flatten()[: live * hidden], torch.cat(expected_data), rtol=0, atol=0
    )
    torch.testing.assert_close(
        col_sf[: live * hidden // 32], torch.cat(expected_sf), rtol=0, atol=0
    )
    if row_output:
        expected_row = swizzle_mxfp8_scale(live, hidden, scales[:live], False).flatten()
        torch.testing.assert_close(row_sf[: expected_row.numel()], expected_row, rtol=0, atol=0)
    if return_dequantized:
        torch.testing.assert_close(dequantized[:live], decoded[:live], rtol=0, atol=0)
    assert (dequantized[live if return_dequantized else 0 :] == -123).all()
    assert (dst[live:] == 0xA5).all()
    assert (col_sf[live * hidden // 32 :] == 0xA5).all()
    assert (row_sf[(live * hidden // 32 if row_output else 0) :] == 0xA5).all()


@pytest.mark.parametrize("warp_tile", [False, True])
@pytest.mark.parametrize("return_dequantized", [False, True])
@pytest.mark.parametrize("scale", [0, 1, 127, 254, 255])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.usefixtures("decode_backend")
def test_extreme_scale_bf16_decode(scale, return_dequantized, input_dtype, warp_tile):
    """Document tiny-value BF16 rounding and check saturated/NaN scale behavior."""
    if warp_tile and return_dequantized:
        pytest.skip("The warp-tile kernel has no BF16 output")
    hidden, rows = 256, 128
    # Cover every FP8 encoding, including signed zero, subnormals, NaNs and E5M2 infinities.
    data = torch.arange(hidden, device="cuda", dtype=torch.uint8).repeat(rows, 1).view(input_dtype)
    sf = torch.full((rows, hidden // 32), scale, device="cuda", dtype=torch.uint8)
    dst = torch.empty((rows, hidden), device="cuda", dtype=torch.float8_e4m3fn)
    col_sf = torch.empty(rows * hidden // 32, device="cuda", dtype=torch.uint8)
    dtype = cutlass.Float8E4M3FN if input_dtype == torch.float8_e4m3fn else cutlass.Float8E5M2
    if warp_tile:
        kernel = GroupedRequantizeWarpTile(
            hidden, input_dtype=dtype, swizzled=False, rowwise_output=False
        )
    else:
        kernel = GroupedRequantize(
            hidden,
            1,
            rows,
            input_dtype=dtype,
            swizzled=False,
            rowwise_output=False,
            uniform_rows=rows,
            return_dequantized=return_dequantized,
        )
    offsets = torch.tensor([0, rows * hidden], device="cuda", dtype=torch.int64)
    dequantized = torch.empty((rows, hidden), device="cuda", dtype=torch.bfloat16)
    args = tuple(
        from_dlpack(tensor, assumed_align=16) if tensor is not None else None
        for tensor in (
            data,
            sf,
            offsets if warp_tile else None,
            dst,
            None,
            col_sf,
            dequantized if return_dequantized else None,
        )
    ) + (cuda.CUstream(torch.cuda.current_stream().cuda_stream),)
    cute.compile(kernel, *args)(*args)
    decoded = (data.float() * (float("nan") if scale == 255 else 2.0 ** (scale - 127))).to(
        torch.bfloat16
    )
    if return_dequantized:
        torch.testing.assert_close(dequantized, decoded, rtol=0, atol=0, equal_nan=True)
    expected, expected_sf = columnwise_reference(decoded)
    torch.testing.assert_close(dst.view(torch.uint8), expected, rtol=0, atol=0)
    torch.testing.assert_close(col_sf, expected_sf.flatten(), rtol=0, atol=0)


@pytest.mark.parametrize("swizzled", [False, True])
def test_warp_tile_compiled_once_for_every_capacity(swizzled):
    """One production compile of the warp-tile kernel serves any capacity and group count."""
    hidden = 512
    compiled = requantize_mxfp8._compile_warp_tile_requantization(
        cutlass.Float8E4M3FN, hidden, swizzled, swizzled
    )
    for rows, sizes in ((512, [128, 0, 256]), (1024, [256, 128, 0, 384, 128])):
        torch.manual_seed(rows)
        live = sum(sizes)
        data = (torch.randn(rows, hidden, device="cuda") * 8).to(torch.float8_e4m3fn)
        scales = torch.randint(100, 155, (rows, hidden // 32), device="cuda", dtype=torch.uint8)
        decoded = (data.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)).to(
            torch.bfloat16
        )
        offsets = torch.tensor([0] + sizes, device="cuda", dtype=torch.int64).cumsum(0) * hidden
        dst = torch.full((rows, hidden), 0xA5, device="cuda", dtype=torch.uint8)
        row_sf = torch.full_like(scales.flatten(), 0xA5)
        col_sf = torch.full_like(row_sf, 0xA5)
        e8m0 = torch.float8_e8m0fnu
        compiled(
            data,
            scales.view(e8m0),
            offsets,
            dst.view(torch.float8_e4m3fn),
            row_sf.view(e8m0) if swizzled else None,
            col_sf.view(e8m0),
            None,
            torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        start, expected_data, expected_sf = 0, [], []
        for size in sizes:
            if size:
                quantized, sf = columnwise_reference(decoded[start : start + size])
                expected_data.append(quantized.flatten())
                if swizzled:
                    sf = swizzle_mxfp8_scale(size, hidden, sf, True)
                expected_sf.append(sf.flatten())
            start += size
        torch.testing.assert_close(
            dst.flatten()[: live * hidden], torch.cat(expected_data), rtol=0, atol=0
        )
        torch.testing.assert_close(
            col_sf[: live * hidden // 32], torch.cat(expected_sf), rtol=0, atol=0
        )
        if swizzled:
            expected_row = swizzle_mxfp8_scale(live, hidden, scales[:live], False).flatten()
            torch.testing.assert_close(row_sf[: expected_row.numel()], expected_row, rtol=0, atol=0)
        assert (dst[live:] == 0xA5).all()
        assert (col_sf[live * hidden // 32 :] == 0xA5).all()


def test_reject_unsupported_scale_stride():
    """Do not create an invalid TMA descriptor for a wide nonaligned scale stride."""
    with pytest.raises(ValueError, match="TMA row stride"):
        GroupedRequantize(640, 1, 128)


@pytest.mark.parametrize("warp_tile", [False, True])
@pytest.mark.parametrize("return_dequantized", [False, True])
def test_empty_groups_leave_capacity_untouched(return_dequantized, warp_tile):
    """A device metadata update can select no live rows without touching outputs."""
    if warp_tile and return_dequantized:
        pytest.skip("The warp-tile kernel has no BF16 output")
    rows, hidden = 256, 512
    data = torch.zeros((rows, hidden), device="cuda", dtype=torch.uint8)
    sf = torch.zeros((rows, hidden // 32), device="cuda", dtype=torch.uint8)
    offsets = torch.zeros(4, device="cuda", dtype=torch.int64)
    outputs = [
        torch.full_like(data, 0xA5),
        torch.full_like(sf.flatten(), 0xA5),
        torch.full_like(sf.flatten(), 0xA5),
    ]
    dequantized = torch.full((rows, hidden), -123, device="cuda", dtype=torch.bfloat16)
    tensors = (data, sf, offsets, *outputs, dequantized if return_dequantized else None)
    args = tuple(
        from_dlpack(tensor, assumed_align=16) if tensor is not None else None for tensor in tensors
    ) + (cuda.CUstream(torch.cuda.current_stream().cuda_stream),)
    kernel = (
        GroupedRequantizeWarpTile(hidden)
        if warp_tile
        else GroupedRequantize(hidden, 3, rows, return_dequantized=return_dequantized)
    )
    cute.compile(kernel, *args)(*args)
    torch.cuda.synchronize()
    for output in outputs:
        assert (output == 0xA5).all()
    assert (dequantized == -123).all()
