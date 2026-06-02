# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Smoke tests for NVFP4 per-token recipe end-to-end through nn.Linear.

These tests do NOT validate numerics tightly — they just exercise the
recipe-driven wiring (forward + backward) and assert outputs are finite
and roughly the right magnitude.

Forward path:

  recipe (NVFP4PerTokenBlockScaling)
    -> NVFP4Quantizer(per_token=True) for input + weight
    -> tex.quantize -> nvte_nvfp4_per_token_quantize
    -> general_gemm (TN) -> tex.nvfp4_cutlass_per_token_gemm
    -> bf16 output

Backward path (this is what makes this file different from a pure
fwd-only smoke):

  loss.backward()
    -> grad_output cast via NVFP4Quantizer(per_token=True)
       (rowwise + columnwise both populated by per-token K1+K2)
    -> general_gemm (NN) for dgrad: dY.rowwise @ W.columnwise.T
    -> general_gemm (NT) for wgrad: dY.columnwise @ X.columnwise.T
    -> bf16 dX (grad_input) and bf16 dW (grad_weight)

If any of those edges has a bad ABI / shape / dtype contract, this test
crashes (segfault, NVTE_CHECK throw, or shape-assert) before any
numerical comparison.

Numeric tightness is the job of test_nvfp4_cutlass_per_token_gemm.py +
test_nvfp4_per_token_quantize.py — both already exist and don't go
through the recipe / module stack. This test fills the recipe-level gap.
"""

# IMPORTANT: import order — `transformer_engine.pytorch` MUST come before
# `transformer_engine_torch` to dlopen libtransformer_engine.so so the
# typeinfo symbols resolve. See te-python-import-order.mdc.
import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Skip the entire file if NVFP4 isn't available on this device.
nvfp4_available, _reason = te.is_nvfp4_available(return_reason=True)
pytestmark = pytest.mark.skipif(not nvfp4_available, reason=_reason)


def _make_per_token_recipe() -> recipe.NVFP4PerTokenBlockScaling:
    """Construct the per-token recipe with the forced mutex flags applied."""
    return recipe.NVFP4PerTokenBlockScaling()


def test_recipe_construction_forces_mutex_flags():
    """Recipe __post_init__ must zero out flags incompatible with per-token."""
    r = _make_per_token_recipe()
    assert r.disable_rht is True, "per-token recipe must force disable_rht=True"
    assert (
        r.disable_stochastic_rounding is True
    ), "per-token recipe must force SR off (no SR kernel yet)"
    assert (
        r.disable_2d_quantization is True
    ), "per-token recipe must force 2D quant off (mutex with per-token amax)"
    assert (
        r.row_scaled_activation is False
    ), "per-token already encodes per-row amax; row_scaled_activation must be off"
    assert r.nvfp4_4over6 == "none", "per-token recipe must force 4over6 off"

    # Recipe.nvfp4() must still return True (backward compat for all the
    # `recipe.nvfp4()` checks scattered through the runtime).
    assert r.nvfp4() is True
    # And the new-class accessor must distinguish per-token from prod NVFP4.
    assert r.nvfp4_per_token() is True
    # With the env var unset (default), a plain NVFP4BlockScaling stays prod.
    assert recipe.NVFP4BlockScaling().nvfp4_per_token() is False


def test_env_var_flips_base_recipe_to_per_token(monkeypatch):
    """``NVTE_NVFP4_PER_TOKEN=1`` must flip a *plain* ``NVFP4BlockScaling``
    into per-token mode, with all mutex flags forced off — identical to
    the explicit ``NVFP4PerTokenBlockScaling`` subclass.

    This is the Megatron-core integration path: mcore already builds a
    bare ``NVFP4BlockScaling`` for ``--fp8-recipe nvfp4``; the env var
    opts that exact object into per-token with zero framework code edit.
    """
    monkeypatch.setenv("NVTE_NVFP4_PER_TOKEN", "1")

    r = recipe.NVFP4BlockScaling()  # plain base recipe, NOT the subclass

    # Routing flag flips on.
    assert r.nvfp4_per_token() is True
    # Mutex-incompatible knobs forced off (same contract as the subclass).
    assert r.disable_rht is True
    assert r.disable_stochastic_rounding is True
    assert r.disable_2d_quantization is True
    assert r.row_scaled_activation is False
    assert r.nvfp4_4over6 == "none"
    # QParams built from the forced flags must carry no RHT / SR / 2D.
    assert r.fp4_quant_fwd_inp.random_hadamard_transform is False
    assert r.fp4_quant_fwd_weight.fp4_2d_quantization is False
    assert r.fp4_quant_bwd_grad.stochastic_rounding is False
    # repr must surface the per-token state so logs aren't ambiguous.
    assert "per_token=True" in repr(r)
    # Still an NVFP4 recipe for the rest of the runtime's `nvfp4()` checks.
    assert r.nvfp4() is True


def test_env_var_unset_keeps_base_recipe_prod(monkeypatch):
    """Sanity inverse: with the env var explicitly unset, the base recipe
    must stay prod NVFP4 (RHT/SR/2D at their normal defaults)."""
    monkeypatch.delenv("NVTE_NVFP4_PER_TOKEN", raising=False)

    r = recipe.NVFP4BlockScaling()
    assert r.nvfp4_per_token() is False
    assert "per_token=True" not in repr(r)


@pytest.mark.parametrize(
    # Combined kernel constraints (intersection of cast + GEMM) under 1-CTA:
    #   - cast (nvte_nvfp4_per_token_quantize): M % 128 == 0, K % 128 == 0
    #   - GEMM (nvte_nvfp4_cutlass_per_token_gemm) with MmaTile (128, 128, 256),
    #     ClusterShape (1, 1, 1): M % 128, N % 128, K % 128
    #     (K_tile = 256 is the mainloop step, NOT a K alignment requirement —
    #      CUTLASS predicates the last K-residue tile.)
    # Smallest legal shape is therefore (128, 128, 128). The shapes below
    # actively exercise the dimensions that were over-constrained by the
    # previous %256-everywhere boilerplate, so we guard against regression.
    #
    # (Switching to a 2-CTA cluster — path B — would tighten M to %256.)
    "M,N,K",
    [
        (128, 128, 128),  # absolute smallest (was %256-rejected on M, N, K)
        (128, 256, 128),  # K=128 stress (was %256-rejected)
        (256, 128, 128),  # K=128 stress, axis-swap guard
        (256, 256, 128),  # K-only at minimum, K-residue predication test
        (512, 1024, 256),  # asymmetric mid-size, K = 1 full mainloop tile
        (1024, 1024, 768),  # K not power-of-2 multiple of 128 (matches cublas test)
        (4096, 4096, 4096),  # production-class
    ],
)
def test_linear_fwd_smoke(M, N, K):
    """nn.Linear fwd under per-token recipe must run end-to-end without crashing.

    We deliberately allocate a fresh module per shape so we don't reuse a
    stale FP4 weight cache from a previous shape (the recipe's quantizer
    state is per-module).
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build module + input. Bias on to also exercise the bf16 bias-add
    # epilogue path in `_nvfp4_per_token_gemm`.
    linear = te.Linear(K, N, bias=True, params_dtype=dtype).to(device=device)

    # Ensure the weight + bias init is bf16 so the per-token cast (bf16-only)
    # accepts it. te.Linear accepts params_dtype=bf16 above, but cast it
    # explicitly to be safe.
    for p in linear.parameters():
        with torch.no_grad():
            p.copy_(p.to(dtype=dtype))

    x = torch.randn((M, K), dtype=dtype, device=device)

    rec = _make_per_token_recipe()
    with te.fp8_autocast(enabled=True, fp8_recipe=rec):
        y = linear(x)

    assert y.shape == (M, N), f"unexpected output shape {y.shape}, want {(M, N)}"
    assert y.dtype == dtype, f"unexpected output dtype {y.dtype}, want {dtype}"
    assert torch.isfinite(y).all(), "per-token GEMM produced non-finite output"

    # Sanity: should not be all-zero (which would be the failure mode if
    # outer amax = 0 and we silently divided by zero somewhere).
    assert y.abs().max().item() > 0.0


def test_linear_fwd_vs_bf16_loose_numerics():
    """Sanity-check that the per-token FP4 output is in the same ballpark as bf16.

    NVFP4 has ~5–8 bits of effective dynamic range, so we expect a relatively
    loose match (~30–50% relative error on extreme values). We just check the
    fp4 output isn't catastrophically wrong (e.g. all NaN, completely
    different magnitude).
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    # Absolute smallest legal shape under 1-CTA (M, N, K all % 128, K-residue
    # predicated). See nvfp4_cutlass_gemm.cu:414 comment for derivation.
    M, N, K = 128, 128, 128

    torch.manual_seed(0)
    linear = te.Linear(K, N, bias=True, params_dtype=dtype).to(device=device)
    x = torch.randn((M, K), dtype=dtype, device=device)

    # bf16 reference (no quantization). Sigh — use the same module so the
    # weight is identical; just disable autocast.
    with torch.no_grad():
        y_bf16 = linear(x)

    rec = _make_per_token_recipe()
    with te.fp8_autocast(enabled=True, fp8_recipe=rec):
        y_fp4 = linear(x)

    # Magnitude ratio should be in [0.3, 3.0] (well within the FP4 range
    # even for unfavorable seeds). If this fails, we're either dividing by
    # the wrong amax or scaling by the wrong direction.
    bf16_mag = y_bf16.abs().mean().item()
    fp4_mag = y_fp4.abs().mean().item()
    ratio = fp4_mag / max(bf16_mag, 1e-6)
    assert 0.3 <= ratio <= 3.0, (
        f"per-token FP4 magnitude ratio {ratio:.3f} is outside [0.3, 3.0]; "
        f"bf16_mag={bf16_mag:.4e}, fp4_mag={fp4_mag:.4e}"
    )


# ------------------------------------------------------------------------
# Backward smoke (fwd + bwd through nn.Linear under per-token recipe).
# ------------------------------------------------------------------------
#
# Bwd exercises BOTH dgrad (NN layout) and wgrad (NT layout) through the
# extended _nvfp4_per_token_gemm dispatch. The per-token K1+K2 cast must
# emit BOTH rowwise + columnwise data on the grad_output tensor (the
# dgrad GEMM consumes grad_output rowwise, the wgrad GEMM consumes
# grad_output columnwise). If either direction is missing, the dispatch
# raises a clean RuntimeError instead of silently producing garbage.


@pytest.mark.parametrize(
    # Bwd shapes are the same as fwd (cast + GEMM share the M/N/K%128
    # alignment contract; nothing about bwd loosens it).
    "M,N,K",
    [
        (128, 128, 128),  # absolute smallest
        (256, 256, 128),  # K-only at minimum -- exercises K-residue
        (256, 128, 128),  # axis-swap guard
        (512, 1024, 256),  # asymmetric, full K-tile
        (1024, 1024, 768),  # K not power-of-2
    ],
)
def test_linear_bwd_smoke(M, N, K):
    """nn.Linear fwd + bwd under per-token recipe must run end-to-end.

    Specifically validates:
      * dgrad GEMM (NN layout) computes a finite dX of shape (M, K).
      * wgrad GEMM (NT layout) computes a finite dW of shape (N, K).
      * grad_bias accumulates (handled outside the GEMM EVT) into a
        finite db of shape (N,).
      * No NaN / inf anywhere in the gradient pipeline.

    We deliberately use ``loss = y.sum()`` rather than a more elaborate
    objective so the gradient signal is well-conditioned (constant 1
    upstream gradient, no normalizer instability).
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    linear = te.Linear(K, N, bias=True, params_dtype=dtype).to(device=device)
    for p in linear.parameters():
        with torch.no_grad():
            p.copy_(p.to(dtype=dtype))

    x = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)

    rec = _make_per_token_recipe()
    with te.fp8_autocast(enabled=True, fp8_recipe=rec):
        y = linear(x)
        loss = y.sum()
    loss.backward()

    # Output / gradient shape and dtype contracts.
    assert y.shape == (M, N), f"y.shape={y.shape}, want {(M, N)}"
    assert x.grad is not None, "x.grad is None -- dgrad never ran"
    assert x.grad.shape == (M, K), f"dX shape {x.grad.shape}, want {(M, K)}"
    assert x.grad.dtype == dtype, f"dX dtype {x.grad.dtype}, want {dtype}"

    weight_grad = linear.weight.grad
    assert weight_grad is not None, "linear.weight.grad is None -- wgrad never ran"
    # te.Linear stores W as (N, K) so dW is also (N, K).
    assert weight_grad.shape == (N, K), f"dW shape {weight_grad.shape}, want {(N, K)}"

    # Bias gradient (db) flows through a separate sum-reduction path
    # (NOT through the per-token GEMM EVT). Just sanity-check it's
    # finite and shaped correctly.
    bias_grad = linear.bias.grad
    assert bias_grad is not None, "linear.bias.grad is None -- bwd skipped bias"
    assert bias_grad.shape == (N,), f"db shape {bias_grad.shape}, want {(N,)}"

    # Finiteness checks. A NaN here would typically mean either the
    # outer-amax was 0 and we silently divided, or the columnwise SF
    # was uninitialized.
    assert torch.isfinite(x.grad).all(), "dX contains NaN/inf"
    assert torch.isfinite(weight_grad).all(), "dW contains NaN/inf"
    assert torch.isfinite(bias_grad).all(), "db contains NaN/inf"

    # Sanity: gradients should not be all-zero (silent divide-by-zero
    # signature).
    assert x.grad.abs().max().item() > 0.0, "dX is identically zero"
    assert weight_grad.abs().max().item() > 0.0, "dW is identically zero"


def test_linear_bwd_vs_bf16_loose_numerics():
    """Loose magnitude check on dX / dW: per-token bwd should be in the
    same ballpark as a pure-bf16 reference run on the same module/input.

    Catches the failure mode where dgrad or wgrad picks the *wrong*
    rowwise/columnwise direction and computes a structurally-correct
    but numerically-wrong gradient (e.g. axis-permuted dY * W product).
    Tolerance is loose (0.3, 3.0) for the same reason as the fwd
    counterpart -- NVFP4 is a 4-bit format; we're catching plumbing
    bugs, not measuring quant noise.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    M, N, K = 256, 256, 256  # mid-sized so dY/W/X all have non-trivial mass

    torch.manual_seed(0xCAFE)
    linear = te.Linear(K, N, bias=True, params_dtype=dtype).to(device=device)
    x = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)

    # Reference path: bf16 forward, bf16 manual dgrad / wgrad. We
    # bypass autograd on the reference side so we can use the exact
    # same weight tensor as the FP4 path (no need to reset grads
    # between runs).
    weight = linear.weight.detach().clone()
    bias = linear.bias.detach().clone()

    with torch.no_grad():
        y_ref = (x.detach() @ weight.t()) + bias  # (M, N)
        # loss_ref = y_ref.sum() -> dy_ref = ones(M, N) bf16
        dy_ref = torch.ones_like(y_ref)
        # dX = dY @ W  -> (M, N) @ (N, K) = (M, K)
        dx_ref = dy_ref.float() @ weight.float()
        # dW = dY^T @ X -> (N, M) @ (M, K) = (N, K)
        dw_ref = dy_ref.float().t() @ x.detach().float()

    # FP4 path: forward + .backward() under the per-token recipe.
    rec = _make_per_token_recipe()
    with te.fp8_autocast(enabled=True, fp8_recipe=rec):
        y_fp4 = linear(x)
        y_fp4.sum().backward()
    dx_fp4 = x.grad
    dw_fp4 = linear.weight.grad

    # Loose magnitude ratio on dX. Inside [0.3, 3.0] = same axis
    # convention. Outside = either dispatcher picked the wrong (rowwise
    # vs columnwise) view of dY or W, OR per-token cast scrambled an
    # axis.
    dx_ref_mag = dx_ref.abs().mean().item()
    dx_fp4_mag = dx_fp4.float().abs().mean().item()
    dx_ratio = dx_fp4_mag / max(dx_ref_mag, 1e-6)
    assert 0.3 <= dx_ratio <= 3.0, (
        f"dX magnitude ratio {dx_ratio:.3f} outside [0.3, 3.0]; "
        f"ref_mag={dx_ref_mag:.4e}, fp4_mag={dx_fp4_mag:.4e}. "
        "Likely a dgrad operand-direction bug (NN-layout dispatch)."
    )

    # Same loose check on dW.
    dw_ref_mag = dw_ref.abs().mean().item()
    dw_fp4_mag = dw_fp4.float().abs().mean().item()
    dw_ratio = dw_fp4_mag / max(dw_ref_mag, 1e-6)
    assert 0.3 <= dw_ratio <= 3.0, (
        f"dW magnitude ratio {dw_ratio:.3f} outside [0.3, 3.0]; "
        f"ref_mag={dw_ref_mag:.4e}, fp4_mag={dw_fp4_mag:.4e}. "
        "Likely a wgrad operand-direction bug (NT-layout dispatch)."
    )


# ------------------------------------------------------------------------
# N-D (3D) input regression: [batch, seq, hidden] activations.
# ------------------------------------------------------------------------
#
# Every test above feeds a 2D [tokens, features] activation, where
# `data.shape[0]` happens to equal the flattened token count. Real
# transformers feed 3D [batch, seq, hidden], and the NVFP4Tensor stores
# rowwise data with the ORIGINAL leading dims ([batch, seq, K/2]) while the
# per-token amax is computed over the FLATTENED rows (batch*seq). The
# per-token GEMM dispatch (`_nvfp4_per_token_gemm`) must therefore:
#   (1) recover M from the flattened row count, not `data.shape[0]`
#       (else amax-vector mismatch: ka_amax=(batch*seq,) vs M=batch), and
#   (2) restore the activation's N-D leading dims on the fwd output
#       (else the 2D [batch*seq, N] return breaks the residual/bias add
#        downstream, since te.Linear does NOT reshape the fwd GEMM output).
# This was a real bug exposed only by 3D inputs; these cases lock it down.
#
# The (batch, seq) split is deliberately chosen so `batch` ALONE is NOT a
# multiple of 128 -- only the flattened batch*seq is. That way any
# regression back to `M = data.shape[0]` fails loudly (wrong M *and* a
# %128 alignment violation) instead of silently limping.


@pytest.mark.parametrize(
    "B,S,K,N",
    [
        (1, 128, 128, 128),  # smallest; flattened M=128
        (3, 256, 128, 256),  # batch=3 (NOT %128); flattened M=768
        (2, 256, 512, 256),  # flattened M=512, larger K
        (32, 128, 256, 512),  # the LLM-script shape that first hit the bug
    ],
)
def test_linear_3d_input_fwd_bwd_smoke(B, S, K, N):
    """nn.Linear fwd+bwd with a 3D [batch, seq, hidden] activation.

    Guards the N-D flatten (M recovery) + leading-dim restore in
    ``_nvfp4_per_token_gemm``. The headline assertions are structural:
      * fwd output is 3D ``(B, S, N)`` -- NOT a flattened ``(B*S, N)``.
      * dX is 3D ``(B, S, K)`` matching the input.
      * dW is 2D ``(N, K)``.
    plus finiteness / non-zero sanity on all grads.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    linear = te.Linear(K, N, bias=True, params_dtype=dtype).to(device=device)
    for p in linear.parameters():
        with torch.no_grad():
            p.copy_(p.to(dtype=dtype))

    x = torch.randn((B, S, K), dtype=dtype, device=device, requires_grad=True)

    rec = _make_per_token_recipe()
    with te.fp8_autocast(enabled=True, fp8_recipe=rec):
        y = linear(x)
        loss = y.sum()
    loss.backward()

    # (1) fwd output must keep the 3D shape -- the original bug returned a
    # flattened 2D (B*S, N), which silently broke residual adds upstream.
    assert y.ndim == 3, f"fwd output collapsed to {y.ndim}D, want 3D (B, S, N)"
    assert y.shape == (B, S, N), f"y.shape={tuple(y.shape)}, want {(B, S, N)}"
    assert y.dtype == dtype
    assert torch.isfinite(y).all(), "3D fwd produced non-finite output"
    assert y.abs().max().item() > 0.0

    # (2) dX must mirror the 3D input shape.
    assert x.grad is not None, "x.grad is None -- dgrad never ran"
    assert x.grad.shape == (B, S, K), f"dX shape {tuple(x.grad.shape)}, want {(B, S, K)}"
    assert torch.isfinite(x.grad).all(), "dX contains NaN/inf"
    assert x.grad.abs().max().item() > 0.0, "dX is identically zero"

    # (3) dW stays 2D (weights are 2D regardless of activation rank).
    weight_grad = linear.weight.grad
    assert weight_grad is not None, "linear.weight.grad is None -- wgrad never ran"
    assert weight_grad.shape == (N, K), f"dW shape {tuple(weight_grad.shape)}, want {(N, K)}"
    assert torch.isfinite(weight_grad).all(), "dW contains NaN/inf"
    assert weight_grad.abs().max().item() > 0.0, "dW is identically zero"

    bias_grad = linear.bias.grad
    assert bias_grad is not None and bias_grad.shape == (N,)
    assert torch.isfinite(bias_grad).all(), "db contains NaN/inf"
