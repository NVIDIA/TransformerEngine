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


@pytest.fixture(name="fc1_dy_amax_from_dglu_flag")
def _fc1_dy_amax_from_dglu_flag_fixture(monkeypatch):
    """Set ``NVTE_NVFP4_FC1_DY_AMAX_FROM_DGLU`` and drop the cached flag around the test."""
    from transformer_engine.pytorch.ops.fused import grouped_mlp as grouped_mlp_module

    def set_flag(enabled):
        monkeypatch.setenv("NVTE_NVFP4_FC1_DY_AMAX_FROM_DGLU", "1" if enabled else "0")
        grouped_mlp_module._use_fc1_dy_amax_from_dglu.cache_clear()

    grouped_mlp_module._use_fc1_dy_amax_from_dglu.cache_clear()
    yield set_flag
    monkeypatch.delenv("NVTE_NVFP4_FC1_DY_AMAX_FROM_DGLU", raising=False)
    grouped_mlp_module._use_fc1_dy_amax_from_dglu.cache_clear()


def _run_fused_grouped_mlp(split_sizes, hidden_size, expert_scales, stochastic_rounding):
    """Forward and backward of the fused NVFP4 grouped SwiGLU MLP.

    Returns the gradients, plus the FC1 grad output quantization calls recorded by spies on
    the grouped MLP module: the BF16 dy, the dGLU epilogue amax and the amaxes derived from
    it (when the new branch runs), and the quantized grouped dy.
    """
    from transformer_engine.common.recipe import NVFP4BlockScaling
    from transformer_engine.pytorch.ops.fused import grouped_mlp as grouped_mlp_module

    num_groups = len(split_sizes)
    num_tokens = sum(split_sizes)
    fc1_out_features = 2 * hidden_size
    device = "cuda"
    recipe = NVFP4BlockScaling(
        disable_rht=False,
        disable_stochastic_rounding=not stochastic_rounding,
        disable_2d_quantization=True,
    )

    torch.manual_seed(1234)
    x = (torch.rand(num_tokens, hidden_size, device=device) - 0.5) * 0.5
    x = x.to(torch.bfloat16).requires_grad_()
    probs = torch.rand(num_tokens, device=device).to(torch.bfloat16).requires_grad_()
    # Very different per-expert gradient magnitudes, so that a misplaced per-expert amax
    # would show up.
    dy = (torch.rand(num_tokens, hidden_size, device=device) - 0.5) * 0.5
    dy *= torch.repeat_interleave(
        torch.tensor(expert_scales, device=device), torch.tensor(split_sizes, device=device)
    )[:, None]
    dy = dy.to(torch.bfloat16)
    fc1_ws = [
        ((torch.rand(fc1_out_features, hidden_size, device=device) - 0.5) * 0.5)
        for _ in range(num_groups)
    ]
    fc2_ws = [
        ((torch.rand(hidden_size, hidden_size, device=device) - 0.5) * 0.5)
        for _ in range(num_groups)
    ]

    fc1 = te.ops.GroupedLinear(
        num_groups,
        hidden_size,
        fc1_out_features,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
        split_alignment=256,
    )
    fc2 = te.ops.GroupedLinear(
        num_groups,
        hidden_size,
        hidden_size,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
        split_alignment=256,
    )
    with torch.no_grad():
        for idx in range(num_groups):
            getattr(fc1, f"weight{idx}").copy_(fc1_ws[idx])
            getattr(fc2, f"weight{idx}").copy_(fc2_ws[idx])
    module = te.ops.Sequential(fc1, te.ops.ScaledSwiGLU(glu_interleave_size=32), fc2)

    calls = []
    fc1_dy_quantizations = []
    orig_amax = grouped_mlp_module.nvfp4_amax_from_dglu_amax
    orig_quantize = grouped_mlp_module._group_quantize_for_grouped_mlp
    orig_quantize_with_amax = grouped_mlp_module._group_quantize_with_amax_for_grouped_mlp

    def spy_amax(dglu_amax, num_groups, with_rht):
        rowwise, columnwise = orig_amax(dglu_amax, num_groups, with_rht)
        calls.append(
            {
                "dglu_amax": dglu_amax.detach().clone(),
                "num_groups": num_groups,
                "with_rht": with_rht,
                "rowwise_amax": rowwise,
                "columnwise_amax": columnwise,
            }
        )
        return rowwise, columnwise

    def snapshot(out):
        return {
            name: None if getattr(out, name) is None else getattr(out, name).detach().clone()
            for name in ("rowwise_data", "scale_inv", "amax", "columnwise_amax")
        }

    def is_fc1_dy(tensor):
        return tensor.dim() == 2 and tensor.shape == (num_tokens, fc1_out_features)

    def spy_quantize(tensor, *args, **kwargs):
        out = orig_quantize(tensor, *args, **kwargs)
        if is_fc1_dy(tensor):
            fc1_dy_quantizations.append(
                {"dy": tensor.detach().clone(), "out": snapshot(out), "path": "default"}
            )
        return out

    def spy_quantize_with_amax(
        tensor, quantizer, num_groups, split_sizes, rowwise, columnwise, **kw
    ):
        out = orig_quantize_with_amax(
            tensor, quantizer, num_groups, split_sizes, rowwise, columnwise, **kw
        )
        if calls and rowwise is calls[-1]["rowwise_amax"]:
            assert columnwise is calls[-1]["columnwise_amax"]
            fc1_dy_quantizations.append(
                {
                    "dy": tensor.detach().clone(),
                    "out": snapshot(out),
                    "path": "dglu_amax",
                    "rowwise_amax": rowwise.clone(),
                    "columnwise_amax": columnwise.clone(),
                }
            )
        return out

    split_sizes_t = torch.tensor(split_sizes, dtype=torch.int, device=device)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(grouped_mlp_module, "nvfp4_amax_from_dglu_amax", spy_amax)
        patch.setattr(grouped_mlp_module, "_group_quantize_for_grouped_mlp", spy_quantize)
        patch.setattr(
            grouped_mlp_module,
            "_group_quantize_with_amax_for_grouped_mlp",
            spy_quantize_with_amax,
        )
        with te.autocast(enabled=True, recipe=recipe):
            y = module(x, split_sizes_t, probs, split_sizes_t)
        y.backward(dy)
    torch.cuda.synchronize()

    backward_ops = module._module_groups[0]._backward_ops
    assert len(backward_ops) == 1
    assert isinstance(backward_ops[0][0], te.ops.fused.GroupedMLP_CuTeGEMMGLU)

    grads = {"x": x.grad, "probs": probs.grad}
    for idx in range(num_groups):
        grads[f"fc1_w{idx}"] = getattr(fc1, f"weight{idx}").grad
        grads[f"fc2_w{idx}"] = getattr(fc2, f"weight{idx}").grad
    return grads, calls, fc1_dy_quantizations, x.detach()


def _grad_or_zeros(grad):
    return torch.zeros(1, device="cuda") if grad is None else grad


def _deinterleave_glu(t, interleave=32):
    """Split a 32-wide interleaved GLU tensor into its two halves."""
    rows, cols = t.shape
    t = t.view(rows, cols // (2 * interleave), 2, interleave)
    return t[:, :, 0, :].reshape(rows, -1), t[:, :, 1, :].reshape(rows, -1)


@pytest.mark.parametrize("stochastic_rounding", [False, True], ids=["rn", "sr"])
def test_fused_grouped_mlp_backward(fc1_dy_amax_from_dglu_flag, stochastic_rounding):
    """Real fused grouped MLP backward with the flag on, including empty experts.

    Checks that the dGLU-amax branch runs, that the amaxes it uses are the exact amaxes of
    the BF16 FC1 grad output per expert (and four times that for the post-RHT columnwise
    amax), and compares the gradients against the default path.
    """
    if not te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported():
        pytest.skip("Fused grouped MLP is not supported on this system")
    split_sizes = [512, 0, 256, 768, 0, 256]
    expert_scales = [1.0, 1.0, 64.0, 1.0 / 32.0, 1.0, 4.0]
    hidden_size = 256
    num_groups = len(split_sizes)
    bounds = [0]
    for size in split_sizes:
        bounds.append(bounds[-1] + size)

    fc1_dy_amax_from_dglu_flag(False)
    grads_off, calls_off, quant_off, x = _run_fused_grouped_mlp(
        split_sizes, hidden_size, expert_scales, stochastic_rounding
    )
    fc1_dy_amax_from_dglu_flag(True)
    grads_on, calls_on, quant_on, _ = _run_fused_grouped_mlp(
        split_sizes, hidden_size, expert_scales, stochastic_rounding
    )

    # The new branch runs only with the flag.
    assert not calls_off
    assert [q["path"] for q in quant_off] == ["default"]
    assert len(calls_on) == 1
    assert [q["path"] for q in quant_on] == ["dglu_amax"]
    call, q_on, q_off = calls_on[0], quant_on[0], quant_off[0]
    assert call["num_groups"] == num_groups
    assert call["with_rht"]
    assert call["dglu_amax"].numel() == 2 * num_groups

    # The BF16 dy is the same in both runs (the FC2 dgrad does not depend on the flag).
    dy = q_on["dy"]
    assert torch.equal(dy, q_off["dy"])

    rowwise_amax = q_on["rowwise_amax"]
    dglu_amax = call["dglu_amax"].view(num_groups, 2)
    for idx in range(num_groups):
        dy_expert = dy[bounds[idx] : bounds[idx + 1]]
        if dy_expert.numel() == 0:
            # Empty expert: no amax from the epilogue, clamped to zero.
            assert rowwise_amax[idx].item() == 0.0
            continue
        expected = dy_expert.abs().amax().float()
        # Bitwise: the epilogue reduces FP32 values, the output is their BF16 rounding.
        assert rowwise_amax[idx] == expected, (idx, rowwise_amax[idx].item(), expected.item())
        # Each epilogue amax is the amax of one GLU half (in either order).
        halves = sorted(h.abs().amax().float().item() for h in _deinterleave_glu(dy_expert))
        epilogue = sorted(dglu_amax[idx].to(torch.bfloat16).float().tolist())
        assert epilogue == halves, (idx, epilogue, halves)
    assert torch.equal(q_on["columnwise_amax"], rowwise_amax * NVFP4_RHT_AMAX_BOUND)

    # The default path's exact rowwise amax matches; the bound is >= its post-RHT amax.
    out_on, out_off = q_on["out"], q_off["out"]
    assert torch.equal(out_on["amax"].view(-1), out_off["amax"].view(-1))
    assert torch.equal(out_on["amax"].view(-1), rowwise_amax)
    assert torch.all(out_on["columnwise_amax"].view(-1) >= out_off["columnwise_amax"].view(-1))
    if not stochastic_rounding:
        # Row-wise (dgrad) operand is bit-identical to the default path.
        assert torch.equal(out_on["rowwise_data"], out_off["rowwise_data"])
        assert torch.equal(out_on["scale_inv"], out_off["scale_inv"])

    # Gradients that do not depend on the FC1 dy columnwise operand are unchanged.
    names = ["probs"] + [f"fc2_w{idx}" for idx in range(num_groups)]
    if not stochastic_rounding:
        names.append("x")
    for name in names:
        assert torch.equal(_grad_or_zeros(grads_on[name]), _grad_or_zeros(grads_off[name])), name
    if stochastic_rounding:
        torch.testing.assert_close(grads_on["x"], grads_off["x"], rtol=0.25, atol=0.5)

    # FC1 wgrad: against the BF16 reference dy^T x per expert, the error with the looser
    # columnwise amax stays within 5% of the error with the exact amax.
    for idx in range(num_groups):
        lo, hi = bounds[idx], bounds[idx + 1]
        if lo == hi:
            assert not torch.any(_grad_or_zeros(grads_on[f"fc1_w{idx}"]))
            assert not torch.any(_grad_or_zeros(grads_off[f"fc1_w{idx}"]))
            continue
        ref = dy[lo:hi].float().t() @ x[lo:hi].float()
        err_on = _rel_err(grads_on[f"fc1_w{idx}"].float(), ref)
        err_off = _rel_err(grads_off[f"fc1_w{idx}"].float(), ref)
        print(
            f"[dglu] sr={stochastic_rounding} expert {idx}: wgrad rel err {err_on:.4f} (dglu amax)"
            f" vs {err_off:.4f} (exact)"
        )
        assert err_on <= err_off * 1.05 + 1e-3, (idx, err_on, err_off)
