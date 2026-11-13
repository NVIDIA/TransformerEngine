# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused BHSD permute + GEMM swizzle of MXFP8 attention scales (Triton) against the
separate permute and swizzle kernels."""

import copy

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.attention.dot_product_attention import (
    utils as dpa_utils,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fused = dpa_utils._fused_transpose_swizzle_module()
pytestmark = [
    pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
    pytest.mark.skipif(fused is None, reason="Triton is not available"),
]


def _quantize(src_format, S, B, H, dims, rowwise=True, columnwise=True, seed=0):
    """MXFP8 tensors with data and scales in src_format, not yet permuted or swizzled."""
    torch.manual_seed(seed)
    pairs = []
    for D in dims:
        shape = (S, B, H, D) if src_format == "sbhd" else (B, S, H, D)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        x = x * torch.rand(shape[:-1] + (1,), device="cuda") * 10
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise
        )
        pairs.append((x, quantizer))
    return dpa_utils.mxfp8_quantize_only(pairs, src_format)


def _clone(tensors):
    out = []
    for t in tensors:
        c = copy.copy(t)
        for name in ("_rowwise_scale_inv", "_columnwise_scale_inv"):
            scale = getattr(t, name)
            setattr(c, name, None if scale is None else scale.clone())
        out.append(c)
    return out


def _assert_same_scales(ref, out):
    for a, b in zip(ref, out):
        assert a._with_gemm_swizzled_scales and b._with_gemm_swizzled_scales
        for name in ("_rowwise_scale_inv", "_columnwise_scale_inv"):
            x, y = getattr(a, name), getattr(b, name)
            if x is None:
                assert y is None, name
                continue
            assert x.shape == y.shape, (name, x.shape, y.shape)
            torch.testing.assert_close(y, x, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("src_format", ["sbhd", "bshd"])
@pytest.mark.parametrize(
    "S,B,H,dims",
    [
        (512, 2, 16, (192, 192, 128)),
        (256, 1, 128, (192, 128)),
        (384, 3, 8, (64, 96, 32)),
        (128, 1, 4, (160,)),
    ],
)
@pytest.mark.parametrize("rowwise,columnwise", [(True, True), (True, False), (False, True)])
def test_fused_transpose_swizzle(src_format, S, B, H, dims, rowwise, columnwise):
    """Byte-identical scales to the separate permute and swizzle kernels."""
    tensors = _quantize(src_format, S, B, H, dims, rowwise, columnwise)
    assert fused.can_fuse_transpose_swizzle(tensors, src_format)
    ref, out = _clone(tensors), _clone(tensors)
    dpa_utils._mxfp8_transpose_swizzle_unfused(ref, src_format)
    dpa_utils.mxfp8_transpose_swizzle(out, src_format)
    _assert_same_scales(ref, out)


def test_fused_transpose_swizzle_fallback():
    """More than three tensors take the separate kernels."""
    tensors = _quantize("sbhd", 256, 1, 4, (64, 64, 64, 64))
    assert not fused.can_fuse_transpose_swizzle(tensors, "sbhd")
    ref, out = _clone(tensors), _clone(tensors)
    dpa_utils._mxfp8_transpose_swizzle_unfused(ref, "sbhd")
    dpa_utils.mxfp8_transpose_swizzle(out, "sbhd")
    _assert_same_scales(ref, out)


def test_fused_transpose_swizzle_cuda_graph():
    """A captured launch replays on new scales in the same buffers."""
    dims = (192, 192, 128)
    tensors = _clone(_quantize("sbhd", 256, 1, 8, dims, seed=3))
    sources = [(t._rowwise_scale_inv, t._columnwise_scale_inv) for t in tensors]
    fused.fused_transpose_swizzle(_clone(tensors), "sbhd")  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused.fused_transpose_swizzle(tensors, "sbhd")
    new = _quantize("sbhd", 256, 1, 8, dims, seed=4)
    for (rowwise, columnwise), t in zip(sources, new):
        rowwise.copy_(t._rowwise_scale_inv)
        columnwise.copy_(t._columnwise_scale_inv)
    graph.replay()
    ref = _clone(new)
    dpa_utils._mxfp8_transpose_swizzle_unfused(ref, "sbhd")
    torch.cuda.synchronize()
    _assert_same_scales(ref, tensors)


@pytest.mark.parametrize("qkv_format", ["sbhd", "bshd"])
def test_mxfp8_attention_fused_swizzle(monkeypatch, qkv_format):
    """MXFP8 DotProductAttention output and gradients do not change with the fused kernels."""
    S, B, H, D_QK, D_V = 512, 2, 16, 192, 128
    recipe = MXFP8BlockScaling(fp8_dpa=True)
    torch.manual_seed(1234)
    dpa = te.DotProductAttention(
        num_attention_heads=H,
        kv_channels=(D_QK, D_V),
        attention_dropout=0.0,
        qkv_format=qkv_format,
        attn_mask_type="causal",
    ).cuda()

    def shape(d):
        return (S, B, H, d) if qkv_format == "sbhd" else (B, S, H, d)

    q, k = (torch.randn(shape(D_QK), dtype=torch.bfloat16, device="cuda") for _ in range(2))
    v = torch.randn(shape(D_V), dtype=torch.bfloat16, device="cuda")
    dout = torch.randn(shape(D_V), dtype=torch.bfloat16, device="cuda").flatten(-2)

    calls = []
    fused_fn = fused.fused_transpose_swizzle

    def spy(*args):
        calls.append(1)
        fused_fn(*args)

    def run():
        inputs = [t.clone().requires_grad_() for t in (q, k, v)]
        with te.autocast(enabled=True, recipe=recipe):
            out = dpa(*inputs)
        out.backward(dout)
        FP8GlobalStateManager.reset()
        return [out] + [t.grad for t in inputs]

    monkeypatch.setattr(dpa_utils, "_fused_transpose_swizzle_module", lambda: None)
    ref = run()
    monkeypatch.setattr(dpa_utils, "_fused_transpose_swizzle_module", lambda: fused)
    monkeypatch.setattr(fused, "fused_transpose_swizzle", spy)
    out = run()
    if not calls:
        pytest.skip("The selected attention backend does not use mxfp8_transpose_swizzle")
    for name, a, b in zip(("out", "dq", "dk", "dv"), out, ref):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)
