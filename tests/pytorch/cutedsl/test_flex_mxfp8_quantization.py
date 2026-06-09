# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

from transformer_engine.common.cutedsl.cutedsl_utils import str_to_te_dtype
import transformer_engine.pytorch  # noqa: F401  (loads libtransformer_engine.so)
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.common.cutedsl.cast.mxfp8_quantization import (
    get_mxfp8_quantizer,
)

MXFP8_BLOCK = 32  # MXFP8 scale block size; valid shapes must be multiples of this.

# 2 aligned (no scale padding) + 2 padded (partial tiles);
SHAPES = [(256, 256), (128, 512), (96, 224), (160, 96)]

def get_dtype_combinations():
    dtype_row = ("e4m3", "e5m2", "none")
    dtype_column = ("e4m3", "e5m2", "none")
    return [(r, c) for r in dtype_row for c in dtype_column]

DTYPE_PAIRS = get_dtype_combinations()

def reference_quantize(x, fp8_type, rowwise, columnwise, swizzle):
    q = MXFP8Quantizer(fp8_dtype=str_to_te_dtype(fp8_type), rowwise=rowwise, columnwise=columnwise)
    q.optimize_for_gemm = swizzle  # makes the native kernel emit swizzled scales
    ref = tex.quantize(x.clone(), q)
    return ref

@pytest.mark.parametrize("swizzle", [False, True])
@pytest.mark.parametrize("dtype_pair", DTYPE_PAIRS)
@pytest.mark.parametrize("shape", SHAPES)
def test_flex_mxfp8_bitexact(shape, dtype_pair, swizzle):
    M, N = shape
    dtype_row, dtype_column = dtype_pair
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # No direction is invalid -- the quantizer must reject it at construction.
    if dtype_row == "none" and dtype_column == "none":
        with pytest.raises(ValueError):
            get_mxfp8_quantizer(x, dtype_row, dtype_column, with_gemm_swizzled_scales=swizzle)
        return

    flex_q = get_mxfp8_quantizer(
        x, dtype_row=dtype_row, dtype_col=dtype_column, with_gemm_swizzled_scales=swizzle
    )
    flex = tex.quantize(x, flex_q)
    torch.cuda.synchronize()

    if dtype_row != "none":
        scale_M, scale_N = M, N // MXFP8_BLOCK
        # Reference for this direction uses THIS direction's dtype.
        ref = reference_quantize(x, dtype_row, rowwise=True, columnwise=False, swizzle=swizzle)
        assert ref._rowwise_data.shape == flex._rowwise_data.shape, "rowwise data shape mismatch"
        assert ref._rowwise_scale_inv.shape == flex._rowwise_scale_inv.shape, "rowwise scale shape mismatch"
        torch.testing.assert_close(flex._rowwise_data, ref._rowwise_data, rtol=0, atol=0)  # bit-identical
        if swizzle:
            torch.testing.assert_close(flex._rowwise_scale_inv, ref._rowwise_scale_inv, rtol=0, atol=0)
        else:
            torch.testing.assert_close(
                flex._rowwise_scale_inv[:scale_M, :scale_N],
                ref._rowwise_scale_inv[:scale_M, :scale_N], 
                rtol=0, atol=0
            )
    else:
        assert flex._rowwise_data is None, "row=none must not produce rowwise data"

    if dtype_column != "none":
        scale_M, scale_N = M // MXFP8_BLOCK, N
        ref = reference_quantize(x, dtype_column, rowwise=False, columnwise=True, swizzle=swizzle)
        assert ref._columnwise_data.shape == flex._columnwise_data.shape, "columnwise data shape mismatch"
        assert ref._columnwise_scale_inv.shape == flex._columnwise_scale_inv.shape, "columnwise scale shape mismatch"
        torch.testing.assert_close(flex._columnwise_data, ref._columnwise_data, rtol=0, atol=0)  # bit-identical
        if swizzle:
            torch.testing.assert_close(flex._columnwise_scale_inv, ref._columnwise_scale_inv, rtol=0, atol=0)
        else:
            torch.testing.assert_close(
                flex._columnwise_scale_inv[:scale_M, :scale_N],
                ref._columnwise_scale_inv[:scale_M, :scale_N], 
                rtol=0, atol=0
            )
    else:
        assert flex._columnwise_data is None, "col=none must not produce colwise data"

def test_flex_mxfp8_wrong_shape():
    """A quantizer is compiled for a specific (M, N); using it on a different N
    must error rather than silently mis-quantize.

    The kernel name is the cache key encoding the baked (constexpr) shape, so
    the registered kernel only accepts tensors of that shape -- feeding it a
    different N trips the compiled entry's shape guarantee.
    """
    M, N = (128, 256)
    x1 = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    flex_q = get_mxfp8_quantizer(x1, dtype_row="e4m3", dtype_col="e5m2")

    tex.quantize(x1, flex_q)  # sanity: works for the shape it was compiled for

    # Changed N: the AOT entry was compiled with literal shapes, so its baked
    # per-arg shape check rejects the mismatched tensor before the kernel runs
    # (e.g. "Mismatched mX.shape[1] ..."), rather than silently mis-quantizing.
    # `match` keeps the test from passing on some unrelated failure.
    x2 = torch.randn(M, N * 2, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="[Mm]ismatch"):
        tex.quantize(x2, flex_q)
        torch.cuda.synchronize()
