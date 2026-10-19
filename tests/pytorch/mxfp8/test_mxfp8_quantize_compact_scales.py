# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer, MXFP8Tensor

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)

CANARY = 0xAA
GUARD = 256  # Bytes on each side of a scale view that quantization must leave untouched.


def _make_quantizer(rowwise, columnwise, with_2d_quantization):
    quantizer = MXFP8Quantizer(
        tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
        with_2d_quantization=with_2d_quantization,
    )
    quantizer.optimize_for_gemm = False
    return quantizer


def _make_output(shape, scale_shapes, rowwise, columnwise, with_2d_quantization, scale_offset=0):
    """MXFP8 tensor whose unswizzled scales are views guarded by canary bytes on both sides."""
    guards = {}

    def scale_view(name):
        numel = scale_shapes[name][0] * scale_shapes[name][1]
        start = GUARD + scale_offset
        buf = torch.full((start + numel + GUARD,), CANARY, dtype=torch.uint8, device="cuda")
        guards[name] = buf, start, numel
        return buf[start : start + numel].view(scale_shapes[name])

    def data():
        return torch.empty(shape, dtype=torch.uint8, device="cuda")

    out = MXFP8Tensor(
        shape=shape,
        dtype=torch.bfloat16,
        device="cuda",
        rowwise_data=data() if rowwise else None,
        rowwise_scale_inv=scale_view("rowwise") if rowwise else None,
        columnwise_data=data() if columnwise else None,
        columnwise_scale_inv=scale_view("columnwise") if columnwise else None,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=_make_quantizer(rowwise, columnwise, with_2d_quantization),
        with_gemm_swizzled_scales=False,
    )
    return out, guards


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "shape",
    # Row and column counts that are not multiples of the kernels' 64- and 128-element
    # tiles, rows of fewer than four scales, and one scale per row or column block.
    [(32, 256), (96, 128), (64, 64), (512, 96), (160, 224), (32, 32)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("usage", ["rowwise", "columnwise", "both"])
@pytest.mark.parametrize("with_2d_quantization", [False, True])
@pytest.mark.parametrize("scale_offset", [0, 4], ids=["aligned", "unaligned"])
def test_quantize_into_compact_scales(shape, dtype, usage, with_2d_quantization, scale_offset):
    """Compact unswizzled scales hold the same values as TE's padded scales.

    Quantization must not write past either end of a compact scale buffer, which may be one
    of several planes packed into a larger allocation.
    """
    rows, cols = shape
    rowwise = usage in ("rowwise", "both")
    columnwise = usage in ("columnwise", "both")
    torch.manual_seed(1234)
    src = (8 * torch.randn(shape, device="cuda")).to(dtype)

    ref = _make_quantizer(rowwise, columnwise, with_2d_quantization)(src)
    out, guards = _make_output(
        shape,
        {"rowwise": (rows, cols // 32), "columnwise": (rows // 32, cols)},
        rowwise,
        columnwise,
        with_2d_quantization,
        scale_offset,
    )
    out.quantize_(src)

    for name, (buf, start, numel) in guards.items():
        outside = torch.cat([buf[:start], buf[start + numel :]])
        assert torch.all(outside == CANARY), f"{name} scales were written outside their buffer"
    if rowwise:
        torch.testing.assert_close(out._rowwise_data, ref._rowwise_data, rtol=0, atol=0)
        torch.testing.assert_close(
            out._rowwise_scale_inv, ref._rowwise_scale_inv[:rows, : cols // 32], rtol=0, atol=0
        )
    if columnwise:
        torch.testing.assert_close(out._columnwise_data, ref._columnwise_data, rtol=0, atol=0)
        torch.testing.assert_close(
            out._columnwise_scale_inv,
            ref._columnwise_scale_inv[: rows // 32, :cols],
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("usage", ["rowwise", "columnwise"])
def test_quantize_rejects_undersized_scales(usage):
    """Scales one block short of the tensor are rejected instead of being partially written."""
    shape = (64, 128)
    rowwise = usage == "rowwise"
    out, _ = _make_output(
        shape,
        {"rowwise": (63, 4), "columnwise": (1, 128)},
        rowwise,
        not rowwise,
        with_2d_quantization=False,
    )
    with pytest.raises(RuntimeError, match="scale_inv must have at least shape"):
        out.quantize_(torch.randn(shape, device="cuda"))
