# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GEMM-swizzled scale layout test for the MXFP8 cast-and-transpose op.

When with_gemm_swizzled_scales=True, the new op must write row-wise scale
bytes directly in the layout consumed by MXFP8 GEMM (the same layout produced
by the standard MXFP8Quantizer.quantize(..., with_gemm_swizzled_scales=True)
path) instead of the compact layout. This test compares the swizzled scales
emitted by the new op against the swizzled scales produced by re-quantizing
the transposed source through the standard quantizer with swizzled output.
"""

from __future__ import annotations

import pytest
import torch

te = pytest.importorskip("transformer_engine")
tex = pytest.importorskip("transformer_engine_torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not hasattr(tex, "mxfp8_scaling_transpose_cast"):
    pytest.skip("Built TE missing mxfp8_scaling_transpose_cast", allow_module_level=True)

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


def _make_source(rows: int, cols: int, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda", generator=g) * 4.0


def _quantize_native_swizzled_transpose(source: torch.Tensor):
    """Reference: re-quantize the actual transpose with the standard quantizer
    and swizzled scales. The byte content of the row-wise scales for source.T
    is what the new op should produce."""
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    q.optimize_for_gemm = True
    q.set_usage(rowwise=True, columnwise=False)
    return q.quantize(source.t().contiguous())


@pytest.mark.parametrize("rows,cols", [(128, 256), (256, 4096)])
def test_swizzled_scales_match_native_transpose(rows, cols):
    source = _make_source(rows, cols)
    fp8_dtype = tex.DType.kFloat8E4M3

    column_quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    column_quantizer.optimize_for_gemm = False
    column_quantizer.set_usage(rowwise=True, columnwise=True)
    column_mxfp8 = column_quantizer.quantize(source)

    helper_quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=False)
    helper_quantizer.optimize_for_gemm = True
    transposed = helper_quantizer.quantize_rowwise_transpose(
        source,
        column_mxfp8._columnwise_scale_inv.contiguous(),
        with_gemm_swizzled_scales=True,
    )

    native_t = _quantize_native_swizzled_transpose(source)

    # Payload bytes (no swizzling on payload) must match native transposed
    # quantization byte-for-byte, since both paths quantize the same source
    # blocks with the same E8M0 scales.
    assert torch.equal(
        transposed._rowwise_data.view(torch.uint8), native_t._rowwise_data.view(torch.uint8)
    ), "Swizzled transpose-emit payload bytes differ from native transposed quantization"

    # Scales must also be exact byte-equal — both paths target the GEMM
    # swizzled layout for the same logical row-wise tensor.
    assert torch.equal(transposed._rowwise_scale_inv, native_t._rowwise_scale_inv), (
        "Swizzled row-wise scale bytes differ from native transposed quantization"
    )
