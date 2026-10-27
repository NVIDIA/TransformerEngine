# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for NVTE_NVFP4_FC1_DY_AMAX_FROM_DGLU.

The fused grouped MLP can quantize the FC1 grad output with amaxes derived from the FC2
dGLU epilogue amax: the exact rowwise amax, and ``4 * rowwise amax`` as an upper bound on
the post-RHT columnwise amax. These tests check the amax kernel, that the bound holds, that
the rowwise quantization is unchanged, and that the wgrad error with the bound stays at
the level of the exact amax, with and without stochastic rounding."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor import NVFP4Quantizer
from transformer_engine.pytorch.triton.nvfp4_dglu_amax import (
    NVFP4_RHT_AMAX_BOUND,
    nvfp4_amax_from_dglu_amax,
)

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)


def _quantizers(stochastic_rounding):
    """Recipe defaults for a grouped linear's grad output (dy) and input (x)."""
    dy_quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=stochastic_rounding,
        with_random_sign_mask=True,
    )
    x_quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=True,
    )
    return dy_quantizer, x_quantizer


def _grad_like(tokens, features, seed, outliers=True):
    """Gradient-like tensor: per-token magnitudes over several decades, sparse outliers."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dy = torch.randn(tokens, features, generator=gen, device="cuda")
    dy *= torch.exp(2.0 * torch.randn(tokens, 1, generator=gen, device="cuda"))
    if outliers:
        num = max(1, tokens * features // 50000)
        idx = torch.randint(0, tokens * features, (num,), generator=gen, device="cuda")
        dy.view(-1)[idx] *= 50.0
    return dy.to(torch.bfloat16)


def _quantize_with_bound(dy, quantizer):
    amax = dy.abs().amax().float().view(1)
    return tex.nvfp4_quantize_with_amax(dy, quantizer, amax, amax * NVFP4_RHT_AMAX_BOUND)


def _wgrad(dy, x):
    out, *_ = general_gemm(x, dy, out_dtype=torch.float32, layout="NT")
    return out


def _rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("num_groups", [1, 8, 64, 128, 257])
@pytest.mark.parametrize("with_rht", [True, False])
def test_amax_kernel(num_groups, with_rht):
    torch.manual_seed(num_groups)
    scale = torch.logspace(-6, 3, num_groups, device="cuda")[:, None, None]
    dglu_amax = torch.rand(num_groups, 2, 1, device="cuda") * scale
    if num_groups > 1:
        dglu_amax[1] = float("-inf")  # expert without tokens
    rowwise, columnwise = nvfp4_amax_from_dglu_amax(dglu_amax, num_groups, with_rht)
    ref = dglu_amax.view(num_groups, 2).amax(dim=1).clamp_min(0.0)
    ref = ref.to(torch.bfloat16).float()
    assert torch.equal(rowwise, ref)
    assert torch.equal(columnwise, ref * (NVFP4_RHT_AMAX_BOUND if with_rht else 1.0))


def test_amax_kernel_rejects_bad_shape():
    with pytest.raises(ValueError):
        nvfp4_amax_from_dglu_amax(torch.zeros(3, device="cuda"), 2, True)


def test_bf16_rounded_amax_equals_output_amax():
    """The epilogue reduces FP32 values; the output is their BF16 rounding."""
    for seed in range(10):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        scale = 10 ** torch.randn(1, generator=gen, device="cuda")
        x = torch.randn(4096, 1024, generator=gen, device="cuda") * scale
        assert (
            x.abs().amax().to(torch.bfloat16).float() == x.to(torch.bfloat16).abs().amax().float()
        )


def test_bound_is_upper_bound():
    dy_quantizer, _ = _quantizers(stochastic_rounding=False)
    for seed in range(5):
        dy = _grad_like(1024, 4096, seed=seed)
        post_rht_amax = dy_quantizer(dy)._amax_columnwise.float().item()
        assert post_rht_amax <= NVFP4_RHT_AMAX_BOUND * dy.abs().amax().float().item()


@pytest.mark.parametrize("tokens,features,in_features", [(512, 4096, 1024), (2048, 4096, 7168)])
@pytest.mark.parametrize("outliers", [False, True])
def test_wgrad_error_with_bound(tokens, features, in_features, outliers):
    dy = _grad_like(tokens, features, seed=1, outliers=outliers)
    x = (torch.randn(tokens, in_features, device="cuda") * 0.5).to(torch.bfloat16)
    ref = dy.float().t() @ x.float()
    dy_quantizer, x_quantizer = _quantizers(stochastic_rounding=False)
    x_q = x_quantizer(x)
    dy_exact = dy_quantizer(dy)
    dy_bound = _quantize_with_bound(dy, dy_quantizer)
    # The rowwise (dgrad) operand is unchanged.
    assert torch.equal(dy_exact._rowwise_data, dy_bound._rowwise_data)
    assert torch.equal(dy_exact._rowwise_scale_inv, dy_bound._rowwise_scale_inv)
    err_exact = _rel_err(_wgrad(dy_exact, x_q), ref)
    err_bound = _rel_err(_wgrad(dy_bound, x_q), ref)
    assert err_bound <= err_exact * 1.05 + 1e-4, (err_bound, err_exact)


def test_stochastic_rounding_stays_unbiased():
    tokens, features, in_features = 512, 2048, 512
    dy = _grad_like(tokens, features, seed=7)
    x = (torch.randn(tokens, in_features, device="cuda") * 0.5).to(torch.bfloat16)
    ref = dy.float().t() @ x.float()
    dy_quantizer, x_quantizer = _quantizers(stochastic_rounding=True)
    x_q = x_quantizer(x)
    num_draws = 64
    acc_bound = torch.zeros_like(ref)
    acc_exact = torch.zeros_like(ref)
    for _ in range(num_draws):
        acc_bound += _wgrad(_quantize_with_bound(dy, dy_quantizer), x_q)
        acc_exact += _wgrad(dy_quantizer(dy), x_q)
    err_single = _rel_err(_wgrad(_quantize_with_bound(dy, dy_quantizer), x_q), ref)
    err_mean_bound = _rel_err(acc_bound / num_draws, ref)
    err_mean_exact = _rel_err(acc_exact / num_draws, ref)
    # Averaging draws shrinks the error (unbiased up to the deterministic rounding of x).
    assert err_mean_bound < err_single * 0.9
    assert err_mean_bound <= err_mean_exact * 1.10 + 1e-4


def test_cuda_graph_replay():
    tokens, features, num_groups = 512, 2048, 1
    dy_quantizer, _ = _quantizers(stochastic_rounding=False)
    dy = _grad_like(tokens, features, seed=3)
    dglu_amax = torch.empty(num_groups, 2, 1, device="cuda")

    def step():
        # Stand-in for the dGLU epilogue amax: one value per GLU half.
        halves = dy.view(tokens, 2, features // 2).abs().float()
        torch.amax(halves, dim=(0, 2), out=dglu_amax.view(2))
        rowwise, columnwise = nvfp4_amax_from_dglu_amax(dglu_amax, num_groups, True)
        return tex.nvfp4_quantize_with_amax(dy, dy_quantizer, rowwise, columnwise)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    dy.copy_(_grad_like(tokens, features, seed=4))
    graph.replay()
    torch.cuda.synchronize()
    eager = _quantize_with_bound(dy, dy_quantizer)
    assert torch.equal(out._rowwise_data, eager._rowwise_data)
    assert torch.equal(out._rowwise_scale_inv, eager._rowwise_scale_inv)
    assert torch.equal(out._columnwise_data, eager._columnwise_data)
    assert torch.equal(out._columnwise_scale_inv, eager._columnwise_scale_inv)
