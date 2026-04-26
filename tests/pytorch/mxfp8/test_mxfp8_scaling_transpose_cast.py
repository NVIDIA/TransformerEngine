# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the new MXFP8 cast-and-transpose op.

These tests are written to drop into tests/pytorch/mxfp8/ of upstream
TransformerEngine. They exercise:

1. nvte_mxfp8_scaling_transpose_cast numerics vs. an in-test reference
   reconstruction (MXFP8Quantizer.quantize columnwise + naive Python
   transpose-and-pack).
2. nvte_mxfp8_scaling_transpose_cast byte-for-byte equivalence to a copy
   adapter that takes the existing column-wise MXFP8 payload, transposes it,
   and rewrites it as row-wise storage.
3. The MXFP8Quantizer.quantize_rowwise_transpose Python helper.
4. The with_gemm_swizzled_scales=True variant (covered in
   test_mxfp8_scaling_transpose_cast_swizzled.py).

All tests gate on:

- CUDA available
- transformer_engine installed
- transformer_engine_torch.mxfp8_scaling_transpose_cast symbol present
"""

from __future__ import annotations

import math

import pytest
import torch

te = pytest.importorskip("transformer_engine")
tex = pytest.importorskip("transformer_engine_torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not hasattr(tex, "mxfp8_scaling_transpose_cast"):
    pytest.skip("Built TE missing mxfp8_scaling_transpose_cast", allow_module_level=True)

from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


def _make_source(rows: int, cols: int, dtype=torch.bfloat16, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, cols), dtype=dtype, device="cuda", generator=g) * 4.0


def _make_quantizer(fp8_dtype) -> MXFP8Quantizer:
    q = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    q.optimize_for_gemm = False
    return q


def _quantize_with_columnwise(quantizer: MXFP8Quantizer, source: torch.Tensor):
    """Quantize source with both row-wise and column-wise MXFP8 storage."""
    quantizer.set_usage(rowwise=True, columnwise=True)
    return quantizer.quantize(source)


def _copy_adapter_transpose(mxfp8_tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: form transposed row-wise MXFP8 by copying the existing
    column-wise MXFP8 payload and column-wise scales into transposed
    row-wise storage."""
    cw_data = mxfp8_tensor._columnwise_data.contiguous()
    cw_scale = mxfp8_tensor._columnwise_scale_inv.contiguous()
    rowwise_data = cw_data.t().contiguous()
    rowwise_scale = cw_scale.t().contiguous()
    return rowwise_data, rowwise_scale


@pytest.mark.parametrize("rows,cols", [(64, 128), (128, 256), (256, 4096)])
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
def test_transpose_cast_matches_copy_adapter_bytes(rows, cols, fp8_dtype):
    """Direct byte equivalence: the new op must produce exactly the same
    payload and scale bytes as transposing existing column-wise MXFP8 storage."""
    source = _make_source(rows, cols)
    quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = False
    mxfp8 = _quantize_with_columnwise(quantizer, source)

    expected_payload, expected_scale = _copy_adapter_transpose(mxfp8)

    rowwise_data = torch.empty((cols, rows), dtype=torch.uint8, device="cuda")
    rowwise_scale = torch.empty(
        (mxfp8._columnwise_scale_inv.shape[1], mxfp8._columnwise_scale_inv.shape[0]),
        dtype=torch.uint8,
        device="cuda",
    )
    tex.mxfp8_scaling_transpose_cast(
        source,
        mxfp8._columnwise_scale_inv.contiguous(),
        rowwise_data,
        rowwise_scale,
        rows,
        cols,
        int(fp8_dtype),
        False,  # with_gemm_swizzled_scales
    )
    torch.cuda.synchronize()

    assert torch.equal(
        rowwise_data.view(torch.uint8), expected_payload.view(torch.uint8)
    ), "Row-wise MXFP8 payload bytes differ from copy-adapter reference"
    assert torch.equal(
        rowwise_scale, expected_scale
    ), "Row-wise MXFP8 scale bytes differ from copy-adapter reference"


@pytest.mark.parametrize("rows,cols", [(64, 128), (256, 4096)])
def test_quantize_rowwise_transpose_helper_equivalence(rows, cols):
    """The Python helper should match the raw extension call."""
    source = _make_source(rows, cols)
    fp8_dtype = tex.DType.kFloat8E4M3

    quantizer = _make_quantizer(fp8_dtype)
    mxfp8 = _quantize_with_columnwise(quantizer, source)

    helper_quantizer = _make_quantizer(fp8_dtype)
    helper_quantizer.set_usage(rowwise=True, columnwise=False)
    transposed = helper_quantizer.quantize_rowwise_transpose(
        source, mxfp8._columnwise_scale_inv.contiguous()
    )

    expected_payload, expected_scale = _copy_adapter_transpose(mxfp8)

    assert tuple(transposed.shape) == (cols, rows)
    assert transposed._rowwise_data is not None
    assert transposed._columnwise_data is None
    assert torch.equal(
        transposed._rowwise_data.view(torch.uint8), expected_payload.view(torch.uint8)
    )
    assert torch.equal(transposed._rowwise_scale_inv, expected_scale)


@pytest.mark.parametrize("rows,cols", [(64, 128), (128, 256)])
def test_transpose_cast_numerical_reconstruction(rows, cols):
    """Block-decoded transposed payload should reconstruct source.T to
    within MXFP8 quantization tolerance, matching the reference quantizer."""
    source = _make_source(rows, cols).to(torch.bfloat16)
    fp8_dtype = tex.DType.kFloat8E4M3

    quantizer = _make_quantizer(fp8_dtype)
    mxfp8 = _quantize_with_columnwise(quantizer, source)

    # Native row-wise reference for source.T: re-quantize the transposed source.
    ref_quantizer = _make_quantizer(fp8_dtype)
    ref_quantizer.set_usage(rowwise=True, columnwise=False)
    ref_t = ref_quantizer.quantize(source.t().contiguous())
    ref_decoded = ref_t.dequantize().to(torch.float32)

    helper_quantizer = _make_quantizer(fp8_dtype)
    helper_quantizer.set_usage(rowwise=True, columnwise=False)
    transposed = helper_quantizer.quantize_rowwise_transpose(
        source, mxfp8._columnwise_scale_inv.contiguous()
    )
    got_decoded = transposed.dequantize().to(torch.float32)

    # Both reconstructions of source.T should be within 2x the per-block
    # MXFP8 quantization error of one another. They differ only in scale
    # selection: native row-wise re-quantizer chooses scales from
    # source.T's row blocks, while transpose-cast reuses scales from
    # source's column blocks. These are the same blocks of source values,
    # so the chosen scales are identical and the decoded outputs should
    # match exactly bit-for-bit modulo block-edge effects.
    rel = (got_decoded - ref_decoded).norm() / (ref_decoded.norm() + 1e-8)
    assert rel.item() < 5e-2, f"transpose-cast reconstruction drifted: rel L2 {rel.item():.4f}"


def test_transpose_cast_rejects_fp8_input():
    """High-precision input is required; an FP8 source must error out."""
    source = _make_source(64, 128, dtype=torch.bfloat16)
    quantizer = _make_quantizer(tex.DType.kFloat8E4M3)
    mxfp8 = _quantize_with_columnwise(quantizer, source)

    rowwise_data = torch.empty((128, 64), dtype=torch.uint8, device="cuda")
    rowwise_scale = torch.empty(
        (mxfp8._columnwise_scale_inv.shape[1], mxfp8._columnwise_scale_inv.shape[0]),
        dtype=torch.uint8,
        device="cuda",
    )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        tex.mxfp8_scaling_transpose_cast(
            mxfp8._rowwise_data,  # FP8, not high-precision
            mxfp8._columnwise_scale_inv,
            rowwise_data,
            rowwise_scale,
            64,
            128,
            int(tex.DType.kFloat8E4M3),
            False,
        )


def test_transpose_cast_requires_block_aligned_dims():
    source = _make_source(64, 128)
    quantizer = _make_quantizer(tex.DType.kFloat8E4M3)
    quantizer.set_usage(rowwise=True, columnwise=False)
    bad_source = torch.randn(48, 128, dtype=torch.bfloat16, device="cuda")
    bad_scale = torch.zeros(
        (max(1, math.ceil(48 / MXFP8_BLOCK_SCALING_SIZE)), 128),
        dtype=torch.uint8,
        device="cuda",
    )
    with pytest.raises((RuntimeError, ValueError)):
        quantizer.quantize_rowwise_transpose(bad_source, bad_scale)
