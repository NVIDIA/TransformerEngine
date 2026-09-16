# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numerical tests for the cuDNN FROST attention backend.

These exist because the CP tests cannot catch what this backend is most likely to get wrong.
run_attention_with_cp.py compares a context-parallel run against a non-CP run *of the same
backend*, which validates the ring plumbing and nothing about the kernel: a systematic error --
a wrong softmax scale, a causal mask anchored to the wrong corner, an LSE in the wrong log base
-- appears identically on both sides and cancels. Everything here is anchored to an independent
float64 reference instead.

The pass criterion is the one FlashAttention applies to itself: the kernel's error against that
reference must stay within 2x the error the reference itself incurs from reduced-precision
inputs. That floor is measured per case rather than hard-coded, so the bar tracks the shape and
dtype instead of encoding a number that silently rots.

The reference is float64, not float32. torch uses TF32 for fp32 matmuls on Ampere and newer, and
TF32's significand is 11 bits -- the same as fp16 -- so an fp32 reference is no more accurate
than an fp16 kernel and the floor collapses to nothing. Measured on B200: the fp16 floor came out
at 3e-08 instead of ~1e-03, which turned the bound into the bare absolute slack.
"""

import math
import os

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
# Mirrors NVTE_GDN_TEST_REQUIRED in test_gdn_attention.py. These tests skip on any machine that
# cannot reach the backend, which on most CI hardware is every machine; setting this on a lane
# that is supposed to cover FROST turns a silent skip into a loud failure.
if os.getenv("NVTE_FROST_TEST_REQUIRED", "0") == "1" and _SKIP is not None:
    raise RuntimeError("NVTE_FROST_TEST_REQUIRED=1, but FrostAttention is unavailable: %s" % _SKIP)
# Applied per test rather than as a module-level pytestmark: the ONNX-export regression
# below guards a code path that runs on every GPU, so gating it on Blackwell would skip it
# exactly where the bug it covers can still occur.
requires_frost = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))

# head_dim 512 is the whole point of the backend; 320 checks the interior of the (256, 512] range
# rather than only its endpoint.
_SHAPES = [
    # b, hq, hkv, sq, skv, d
    (2, 8, 4, 1024, 1024, 512),  # Gemma-4 global layer, GQA
    (2, 8, 8, 512, 512, 512),  # MHA
    (1, 4, 4, 256, 512, 512),  # sq != skv, which is where mask alignment matters
    (2, 4, 4, 512, 512, 320),  # interior head_dim
]


def _reference(q, k, v, scale, mask, window=None):
    """Attention in float64, computed independently of TE and of cuDNN.

    float64 rather than float32 on purpose. torch uses TF32 for fp32 matmuls on Ampere and newer,
    and TF32 carries an 11-bit significand -- the same as fp16. An fp32 reference is therefore no
    more accurate than the fp16 kernel it is meant to judge, which silently collapses the error
    floor below and makes the comparison meaningless. float64 is immune to that and to whatever
    the ambient TF32 flags happen to be.
    """
    qq, kk, vv = q.double(), k.double(), v.double()
    rep = qq.shape[1] // kk.shape[1]
    kk = kk.repeat_interleave(rep, dim=1)
    vv = vv.repeat_interleave(rep, dim=1)
    s = (qq @ kk.transpose(-1, -2)) * scale
    sq, skv = qq.shape[2], kk.shape[2]
    left, right = (-1, -1) if window is None else tuple(window)
    # TE's rule, from the SWA construction in utils.py: a causal mask type pins the right bound to
    # the diagonal, -1 means unbounded on that side, and a window applies to ANY mask type -- so
    # no_mask with (w, 0) is a causal band of width w, not an unmasked attention. Top-left for
    # "causal", bottom-right for "causal_bottom_right"; the two coincide only when sq == skv.
    if mask in ("causal", "causal_bottom_right"):
        right = 0
    offset = skv - sq if mask == "causal_bottom_right" else 0
    blocked = torch.zeros(sq, skv, device=q.device, dtype=torch.bool)
    if right != -1:
        blocked |= torch.ones(sq, skv, device=q.device, dtype=torch.bool).triu(offset + right + 1)
    if left != -1:
        blocked |= torch.ones(sq, skv, device=q.device, dtype=torch.bool).tril(offset - left - 1)
    if bool(blocked.any()):
        s = s.masked_fill(blocked, float("-inf"))
    p = s.softmax(-1)
    return p @ vv, torch.logsumexp(s, dim=-1)


def _floor(q32, k32, v32, scale, mask, dtype, window=None):
    """The error `dtype` inputs alone cause, and the exact answer to measure the kernel against.

    The inputs must originate in higher precision: rounding an already-rounded tensor is a no-op,
    which would collapse the floor to zero and turn the criterion below into an impossible bound.
    """
    exact, exact_lse = _reference(q32, k32, v32, scale, mask, window)
    lossy, lossy_lse = _reference(
        q32.to(dtype).double(), k32.to(dtype).double(), v32.to(dtype).double(), scale, mask, window
    )
    return (
        (exact - lossy).abs().max().item(),
        (exact_lse - lossy_lse).abs().max().item(),
        exact,
        exact_lse,
    )


@requires_frost
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "b%d_hq%d_hkv%d_sq%d_skv%d_d%d" % s)
@pytest.mark.parametrize("mask", ["no_mask", "causal", "causal_bottom_right"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_frost_forward_matches_reference(shape, mask, dtype):
    """Forward output and LSE against an independent float64 reference."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        frost_attn_fwd,
    )

    b, hq, hkv, sq, skv, d = shape
    torch.manual_seed(0)
    # Generate in fp32 so there is a true high-precision original to measure against, then cast
    # for the kernel. [b, h, s, d] views over bshd-contiguous memory is what the backend consumes.
    # A bshd VIEW, which is what the backend receives: to_frost_layout permutes a bshd-contiguous
    # tensor and hands the result over without a copy. Materialising with .contiguous() here would
    # produce bhsd strides instead and leave the stride-keyed plan cache untested.
    mk = lambda s_, h_: torch.randn(b, s_, h_, d, device="cuda").permute(0, 2, 1, 3)
    q32, k32, v32 = mk(sq, hq), mk(skv, hkv), mk(skv, hkv)
    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)
    scale = 1.0 / math.sqrt(d)

    out, lse = frost_attn_fwd(q, k, v, attn_scale=scale, attn_mask_type=mask)

    floor_o, floor_l, ref_o, ref_lse = _floor(q32, k32, v32, scale, mask, dtype)
    err_o = (out.double() - ref_o).abs().max().item()
    err_l = (lse.double() - ref_lse).abs().max().item()

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


@requires_frost
@pytest.mark.parametrize("window", [(256, 0), (128, 0), (0, 0)], ids=lambda w: "win%d" % w[0])
@pytest.mark.parametrize("mask", ["causal", "causal_bottom_right", "no_mask"])
@pytest.mark.parametrize("sq,skv", [(1024, 1024), (512, 1024)], ids=["square", "rect"])
def test_frost_sliding_window_matches_reference(mask, window, sq, skv):
    """Sliding window against the float64 reference.

    The engine advertises swa support, and cuDNN expresses a window as a left bound on the same
    diagonal band that gives causal masking, so this shares a code path with the cases above. It
    is worth its own test because a left bound that is off by one, or silently dropped, still
    produces finite plausible-looking output -- the reference is the only thing that catches it.
    """
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        frost_attn_fwd,
    )

    # The rectangular case is the one that matters for alignment: top-left and bottom-right
    # coincide when sq == skv, so a swapped alignment is invisible in square shapes.
    b, hq, hkv, d = 2, 8, 4, 512
    dtype = torch.bfloat16
    torch.manual_seed(0)
    # A bshd VIEW, which is what the backend receives: to_frost_layout permutes a bshd-contiguous
    # tensor and hands the result over without a copy. Materialising with .contiguous() here would
    # produce bhsd strides instead and leave the stride-keyed plan cache untested.
    mk = lambda s_, h_: torch.randn(b, s_, h_, d, device="cuda").permute(0, 2, 1, 3)
    q32, k32, v32 = mk(sq, hq), mk(skv, hkv), mk(skv, hkv)
    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)
    scale = 1.0 / math.sqrt(d)

    out, _ = frost_attn_fwd(q, k, v, attn_scale=scale, attn_mask_type=mask, window_size=window)

    floor_o, _, ref_o, _ = _floor(q32, k32, v32, scale, mask, dtype, window)
    err = (out.double() - ref_o).abs().max().item()
    assert torch.isfinite(out).all(), "sliding-window forward produced non-finite values"
    assert err <= 2 * floor_o + 1e-3, "out err %.3e exceeds 2x the floor %.3e for window %s" % (
        err,
        floor_o,
        window,
    )

    # A window must actually change the result; if the bound were dropped this would match the
    # unwindowed output and the check above would still pass.
    full, _ = frost_attn_fwd(q, k, v, attn_scale=scale, attn_mask_type=mask)
    assert not torch.equal(out, full), "window %s produced the same output as no window" % (window,)


@requires_frost
@pytest.mark.parametrize("shape", _SHAPES[:2], ids=lambda s: "b%d_hq%d_hkv%d_sq%d_skv%d_d%d" % s)
@pytest.mark.parametrize("mask", ["no_mask", "causal"])
@pytest.mark.parametrize("window", [None, (128, 0)], ids=["nowin", "win128"])
def test_frost_backward_matches_reference(shape, mask, window):
    """dq/dk/dv against autograd on the same independent float64 reference."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        frost_attn_bwd,
        frost_attn_fwd,
    )

    b, hq, hkv, sq, skv, d = shape
    dtype = torch.bfloat16
    torch.manual_seed(0)
    # A bshd VIEW, which is what the backend receives: to_frost_layout permutes a bshd-contiguous
    # tensor and hands the result over without a copy. Materialising with .contiguous() here would
    # produce bhsd strides instead and leave the stride-keyed plan cache untested.
    mk = lambda s_, h_: torch.randn(b, s_, h_, d, device="cuda").permute(0, 2, 1, 3)
    q32, k32, v32 = mk(sq, hq), mk(skv, hkv), mk(skv, hkv)
    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)
    scale = 1.0 / math.sqrt(d)

    out, lse = frost_attn_fwd(q, k, v, attn_scale=scale, attn_mask_type=mask, window_size=window)
    dout = torch.randn_like(out)
    dq, dk, dv = frost_attn_bwd(
        q, k, v, out, lse, dout, attn_scale=scale, attn_mask_type=mask, window_size=window
    )

    qr = q32.detach().clone().requires_grad_(True)
    kr = k32.detach().clone().requires_grad_(True)
    vr = v32.detach().clone().requires_grad_(True)
    ref_o, _ = _reference(qr, kr, vr, scale, mask, window)
    ref_o.backward(dout.double())

    for name, got, want in (("dq", dq, qr.grad), ("dk", dk, kr.grad), ("dv", dv, vr.grad)):
        assert torch.isfinite(got).all(), "%s has non-finite values" % name
        assert got.shape == want.shape, "%s shape %s != %s" % (name, got.shape, want.shape)
        err = (got.double() - want).abs().max().item()
        # Gradients accumulate over the sequence, so scale the bar with skv rather than reusing
        # the forward's floor. This is a sanity bound on systematic error, not a tight check.
        assert err <= 0.05 * want.abs().max().item() + 1e-2, "%s max|err|=%.3e vs ref max %.3e" % (
            name,
            err,
            want.abs().max().item(),
        )


@requires_frost
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
        # window_size reaches _mask_spec through is_frost_attention_supported, so its validation
        # is part of the selector contract rather than an internal detail.
        (dict(window_size=(-1, 5)), "a right window past the diagonal"),
        (dict(window_size=(128,)), "a malformed window pair"),
        (dict(window_size=(-2, 0)), "a left window below -1"),
        (dict(window_size=7), "a non-iterable window"),
        # The engine pads head_dim to a multiple of 8, so an in-range but unpadded dim has to be
        # declined here rather than failing later at plan selection.
        (dict(head_dim_qk=260, head_dim_v=260), "head_dim not a multiple of 8"),
    ):
        cfg = dict(base)
        cfg.update(override)
        ok, reason = is_frost_attention_supported(**cfg)
        assert not ok, "%s must be declined" % why
        assert reason, "a decline must explain itself"


@requires_frost
@pytest.mark.parametrize(
    "cp_comm_type,window,expect_frost",
    [
        ("all_gather", (128, 0), True),
        ("a2a", (128, 0), True),
        ("p2p", (128, 0), False),
        ("a2a+p2p", (128, 0), False),
        ("p2p", (-1, 0), True),
        ("p2p", (-1, -1), True),
    ],
)
def test_frost_sliding_window_selection_by_cp_comm_type(cp_comm_type, window, expect_frost):
    """Which context-parallel paths may serve a sliding window.

    all_gather and a2a each see a contiguous KV range, so the window applies unchanged. The p2p
    ring shards KV across steps, so a bound measured against the full sequence does not survive
    the per-step tiles -- the same rule FusedAttention carries. The cases without a real window
    must still select FROST, since the decline has to key on the window and not on p2p itself.
    """
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        AttentionParams,
        get_attention_backend,
    )

    params = AttentionParams(
        qkv_dtype=torch.bfloat16,
        qkv_layout="bshd_bshd_bshd",
        batch_size=2,
        num_heads=8,
        num_gqa_groups=4,
        max_seqlen_q=4096,
        max_seqlen_kv=4096,
        head_dim_qk=512,
        head_dim_v=512,
        attn_mask_type="causal",
        window_size=window,
        context_parallel=True,
        cp_comm_type=cp_comm_type,
        is_training=True,
    )
    use_frost = get_attention_backend(params)[5]
    assert (
        bool(use_frost) == expect_frost
    ), "cp_comm_type=%s window=%s: expected use_frost_attention=%s, got %s" % (
        cp_comm_type,
        window,
        expect_frost,
        bool(use_frost),
    )


@requires_frost
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
        # k's layout over v's memory and read the wrong elements silently. Build it as sbhd and
        # permute, so the strides genuinely differ -- a [b, h, s, d] contiguous tensor would come
        # out with exactly k's strides and prove nothing.
        v_odd = torch.randn(s, b, h, d, device="cuda", dtype=dtype).permute(1, 2, 0, 3)
        assert v_odd.shape == k.shape and v_odd.stride() != k.stride()
        frost_attn_fwd(q, k, v_odd)
    with pytest.raises(ValueError, match="match q"):
        frost_attn_fwd(q, k, k.to(torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_dot_product_attention_runs_in_onnx_export_mode():
    """The ONNX-export branch must bind every backend flag the availability check reads.

    Deliberately not gated on FROST: that branch skips get_attention_backend entirely and sets the
    flags by hand, so leaving use_frost_attention unbound there raised UnboundLocalError for every
    user on every GPU, whether or not FROST could run. A plain head_dim-64 config reproduces it --
    the failure is in the selector bookkeeping, not in any kernel.
    """
    from transformer_engine.pytorch import DotProductAttention
    from transformer_engine.pytorch.export import onnx_export

    b, h, s, d = 2, 4, 128, 64
    dtype = torch.bfloat16
    qkv = [torch.randn(s, b, h, d, device="cuda", dtype=dtype) for _ in range(3)]
    block = DotProductAttention(
        h, d, qkv_format="sbhd", attn_mask_type="causal", attention_dropout=0.0
    ).to(dtype=dtype, device="cuda")

    with onnx_export(enabled=True):
        out = block(*qkv)

    assert out.numel() == s * b * h * d
    assert torch.isfinite(out).all()
