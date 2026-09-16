# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numerical tests for the cuDNN FROST attention backend.

These exist because the CP tests cannot catch what this backend is most likely to get wrong.
run_attention_with_cp.py compares a context-parallel run against a non-CP run *of the same
backend*, which validates the ring plumbing and nothing about the kernel: a systematic error --
a wrong softmax scale, a causal mask anchored to the wrong corner, an LSE in the wrong log base
-- appears identically on both sides and cancels. Everything here is anchored to an independent
fp32 reference instead.

The pass criterion is the one FlashAttention applies to itself: the kernel's error against an
fp32 reference must stay within 2x the error that comes from feeding the same reference bf16
inputs. That floor is measured per case rather than hard-coded, so the bar tracks the shape and
dtype instead of encoding a number that silently rots.
"""

import math

import pytest
import torch

from transformer_engine.pytorch import get_device_compute_capability


def _frost_availability():
    """Why FrostAttention cannot run here, or None if it can."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    if get_device_compute_capability() not in ((10, 0), (10, 3)):
        return "FrostAttention requires SM100/SM103 (the cuDNN d512 backward is Blackwell-only)."
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        is_frost_attention_available,
    )

    ok, reason = is_frost_attention_available()
    return None if ok else reason


_SKIP = _frost_availability()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))

# head_dim 512 is the whole point of the backend; 320 checks the interior of the (256, 512] range
# rather than only its endpoint.
_SHAPES = [
    # b, hq, hkv, sq, skv, d
    (2, 8, 4, 1024, 1024, 512),  # Gemma-4 global layer, GQA
    (2, 8, 8, 512, 512, 512),  # MHA
    (1, 4, 4, 256, 512, 512),  # sq != skv, which is where mask alignment matters
    (2, 4, 4, 512, 512, 320),  # interior head_dim
]


def _reference(q, k, v, scale, mask):
    """Attention in fp32, computed independently of TE and of cuDNN."""
    qq, kk, vv = q.float(), k.float(), v.float()
    rep = qq.shape[1] // kk.shape[1]
    kk = kk.repeat_interleave(rep, dim=1)
    vv = vv.repeat_interleave(rep, dim=1)
    s = (qq @ kk.transpose(-1, -2)) * scale
    if mask != "no_mask":
        sq, skv = qq.shape[2], kk.shape[2]
        # Top-left for "causal", bottom-right for "causal_bottom_right". These coincide only when
        # sq == skv, which is exactly why _SHAPES includes a rectangular case.
        offset = 0 if mask == "causal" else skv - sq
        causal = torch.ones(sq, skv, device=q.device, dtype=torch.bool).triu(offset + 1)
        s = s.masked_fill(causal, float("-inf"))
    p = s.softmax(-1)
    return p @ vv, torch.logsumexp(s, dim=-1)


def _floor(q32, k32, v32, scale, mask, dtype):
    """The error `dtype` inputs alone cause, and the exact answer to measure the kernel against.

    The inputs must originate in fp32: rounding an already-rounded tensor is a no-op, which would
    collapse the floor to zero and turn the criterion below into an impossible bound.
    """
    exact, exact_lse = _reference(q32, k32, v32, scale, mask)
    lossy, lossy_lse = _reference(
        q32.to(dtype).float(), k32.to(dtype).float(), v32.to(dtype).float(), scale, mask
    )
    return (
        (exact - lossy).abs().max().item(),
        (exact_lse - lossy_lse).abs().max().item(),
        exact,
        exact_lse,
    )


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "b%d_hq%d_hkv%d_sq%d_skv%d_d%d" % s)
@pytest.mark.parametrize("mask", ["no_mask", "causal", "causal_bottom_right"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_frost_forward_matches_fp32_reference(shape, mask, dtype):
    """Forward output and LSE against an independent fp32 reference."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        frost_attn_fwd,
    )

    b, hq, hkv, sq, skv, d = shape
    torch.manual_seed(0)
    # Generate in fp32 so there is a true high-precision original to measure against, then cast
    # for the kernel. [b, h, s, d] views over bshd-contiguous memory is what the backend consumes.
    mk = lambda s_, h_: torch.randn(b, s_, h_, d, device="cuda").permute(0, 2, 1, 3).contiguous()
    q32, k32, v32 = mk(sq, hq), mk(skv, hkv), mk(skv, hkv)
    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)
    scale = 1.0 / math.sqrt(d)

    out, lse = frost_attn_fwd(q, k, v, attn_scale=scale, attn_mask_type=mask)

    floor_o, floor_l, ref_o, ref_lse = _floor(q32, k32, v32, scale, mask, dtype)
    err_o = (out.float() - ref_o).abs().max().item()
    err_l = (lse.float() - ref_lse).abs().max().item()

    assert torch.isfinite(out).all(), "forward produced non-finite values"
    # A floor of exactly zero would make the ratio meaningless; guard with a small absolute term.
    assert err_o <= 2 * floor_o + 1e-3, "out err %.3e exceeds 2x the %s floor %.3e" % (
        err_o,
        dtype,
        floor_o,
    )
    # The LSE convention is what the CP ring correction depends on, so check it explicitly: a
    # log2-based or unscaled LSE would still give a plausible-looking output above.
    assert err_l <= 2 * floor_l + 1e-3, "lse err %.3e exceeds 2x the %s floor %.3e" % (
        err_l,
        dtype,
        floor_l,
    )
    assert lse.shape == (b, hq, sq), "lse must be [b, h, s]; got %s" % (tuple(lse.shape),)
    assert lse.dtype == torch.float32, "lse must be fp32; got %s" % lse.dtype


@pytest.mark.parametrize("shape", _SHAPES[:2], ids=lambda s: "b%d_hq%d_hkv%d_sq%d_skv%d_d%d" % s)
@pytest.mark.parametrize("mask", ["no_mask", "causal"])
def test_frost_backward_matches_fp32_reference(shape, mask):
    """dq/dk/dv against autograd on the same independent fp32 reference."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        frost_attn_bwd,
        frost_attn_fwd,
    )

    b, hq, hkv, sq, skv, d = shape
    dtype = torch.bfloat16
    torch.manual_seed(0)
    mk = lambda s_, h_: torch.randn(b, s_, h_, d, device="cuda").permute(0, 2, 1, 3).contiguous()
    q32, k32, v32 = mk(sq, hq), mk(skv, hkv), mk(skv, hkv)
    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)
    scale = 1.0 / math.sqrt(d)

    out, lse = frost_attn_fwd(q, k, v, attn_scale=scale, attn_mask_type=mask)
    dout = torch.randn_like(out)
    dq, dk, dv = frost_attn_bwd(q, k, v, out, lse, dout, attn_scale=scale, attn_mask_type=mask)

    qr = q32.detach().clone().requires_grad_(True)
    kr = k32.detach().clone().requires_grad_(True)
    vr = v32.detach().clone().requires_grad_(True)
    ref_o, _ = _reference(qr, kr, vr, scale, mask)
    ref_o.backward(dout.float())

    for name, got, want in (("dq", dq, qr.grad), ("dk", dk, kr.grad), ("dv", dv, vr.grad)):
        assert torch.isfinite(got).all(), "%s has non-finite values" % name
        assert got.shape == want.shape, "%s shape %s != %s" % (name, got.shape, want.shape)
        err = (got.float() - want).abs().max().item()
        # Gradients accumulate over the sequence, so scale the bar with skv rather than reusing
        # the forward's floor. This is a sanity bound on systematic error, not a tight check.
        assert err <= 0.05 * want.abs().max().item() + 1e-2, "%s max|err|=%.3e vs ref max %.3e" % (
            name,
            err,
            want.abs().max().item(),
        )


def test_frost_declines_unsupported_configs():
    """The selector must decline what the kernels do not serve, rather than computing wrongly."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        is_frost_attention_supported,
    )

    base = dict(head_dim_qk=512, head_dim_v=512, qkv_dtype=torch.bfloat16, attn_mask_type="causal")
    assert is_frost_attention_supported(**base)[0], "the supported case must be accepted"

    for override, why in (
        (dict(head_dim_qk=256, head_dim_v=256), "head_dim at the exclusive lower bound"),
        (dict(head_dim_v=256), "asymmetric head_dim"),
        (dict(qkv_dtype=torch.float32), "fp32"),
        (dict(dropout=0.1), "dropout"),
        (dict(attn_bias_type="post_scale_bias"), "attention bias"),
        (dict(attn_mask_type="padding_causal"), "padding mask"),
        (dict(attn_mask_type="arbitrary"), "arbitrary mask"),
    ):
        cfg = dict(base)
        cfg.update(override)
        ok, reason = is_frost_attention_supported(**cfg)
        assert not ok, "%s must be declined" % why
        assert reason, "a decline must explain itself"


def test_frost_rejects_mismatched_kv():
    """k and v must agree: the graphs declare v with k's shape and stride."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        frost_attn_fwd,
    )

    b, h, s, d = 2, 4, 512, 512
    dtype = torch.bfloat16
    mk = lambda hh: torch.randn(b, s, hh, d, device="cuda", dtype=dtype).permute(0, 2, 1, 3)
    q, k = mk(h).contiguous(), mk(h).contiguous()

    with pytest.raises(ValueError, match="same shape"):
        frost_attn_fwd(q, k, mk(h * 2).contiguous())
    with pytest.raises(ValueError, match="same layout"):
        # Same shape, different stride order: a cache hit would otherwise run a graph built for
        # k's layout over v's memory and read the wrong elements silently.
        v_odd = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
        frost_attn_fwd(q, k, v_odd)
    with pytest.raises(ValueError, match="match q"):
        frost_attn_fwd(q, k, k.to(torch.float32))
