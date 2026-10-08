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
# Mirrors NVTE_GDN_TEST_REQUIRED: these skip on any machine that cannot reach the backend.
if os.getenv("NVTE_FROST_TEST_REQUIRED", "0") == "1" and _SKIP is not None:
    raise RuntimeError("NVTE_FROST_TEST_REQUIRED=1, but FrostAttention is unavailable: %s" % _SKIP)
# Per test, not a module-level pytestmark: the ONNX-export regression below runs on every GPU, so
# gating it on Blackwell would skip it exactly where its bug can still occur.
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


def _shape_id(s):
    return "b%d_hq%d_hkv%d_sq%d_skv%d_d%d" % s


def _fwd(q, k, v, mask, scale, window=None):
    """The forward through the fused signature, which is the only entry point the backend has.

    Everything is bshd here, because the shim derives one qkv_format and makes the tensors
    contiguous; that is exactly what the dispatcher hands it in production.
    """
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        fused_attn_fwd,
    )

    out, aux = fused_attn_fwd(
        True,
        q.shape[1],
        k.shape[1],
        None,
        None,
        q,
        k,
        v,
        None,
        None,
        attn_scale=scale,
        qkv_layout="bshd_bshd_bshd",
        o_format="bshd",
        attn_mask_type=mask,
        window_size=(-1, -1) if window is None else window,
    )
    return out, aux[0]


def _bwd(q, k, v, out, lse, dout, mask, scale, window=None):
    """The backward through the fused signature. aux_ctx_tensors is what the forward returned."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        fused_attn_bwd,
    )

    dq, dk, dv, _ = fused_attn_bwd(
        q.shape[1],
        k.shape[1],
        None,
        None,
        q,
        k,
        v,
        out,
        dout,
        None,
        [lse, torch.empty(2, dtype=torch.int64, device=q.device)],
        None,
        attn_scale=scale,
        qkv_layout="bshd_bshd_bshd",
        o_format="bshd",
        do_format="bshd",
        dqkv_layout="bshd_bshd_bshd",
        attn_mask_type=mask,
        window_size=(-1, -1) if window is None else window,
    )
    return dq, dk, dv


def _bhsd(t):
    """A [b, h, s, d] view of a bshd tensor. The reference works in that order; the kernel does
    not, since it takes TE's format and reorders the cuDNN descriptors instead."""
    return t.permute(0, 2, 1, 3)


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


# bf16 everywhere, fp16 on two shapes. What fp16 risks that bf16 does not is its narrower
# exponent range, and that surfaces in the backward, which runs both dtypes on every shape it
# covers. Crossing it with every forward shape pays for the same information twice.
_FWD_CASES = [(s, torch.bfloat16) for s in _SHAPES] + [(s, torch.float16) for s in _SHAPES[:2]]
_FWD_IDS = [
    "%s_%s" % (_shape_id(s), "bf16" if d is torch.bfloat16 else "fp16") for s, d in _FWD_CASES
]


@requires_frost
@pytest.mark.parametrize("shape,dtype", _FWD_CASES, ids=_FWD_IDS)
@pytest.mark.parametrize("mask", ["no_mask", "causal", "causal_bottom_right"])
def test_frost_forward_matches_reference(shape, mask, dtype):
    """Forward output and LSE against an independent float64 reference."""
    b, hq, hkv, sq, skv, d = shape
    torch.manual_seed(0)
    # Generate in fp32 so there is a true high-precision original to measure against, then cast
    # for the kernel. bshd is what the backend takes now: it is never permuted, only described.
    mk = lambda s_, h_, d_: torch.randn(b, s_, h_, d_, device="cuda")
    q32, k32, v32 = mk(sq, hq, d), mk(skv, hkv, d), mk(skv, hkv, d)
    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)
    scale = 1.0 / math.sqrt(d)

    out, lse = _fwd(q, k, v, mask, scale)

    floor_o, floor_l, ref_o, ref_lse = _floor(
        _bhsd(q32), _bhsd(k32), _bhsd(v32), scale, mask, dtype
    )
    err_o = (_bhsd(out).double() - ref_o).abs().max().item()
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
# (128, 0) is the ordinary case and (0, 0) the degenerate diagonal-only one, which is where an
# off-by-one in the band would show. A second ordinary width tests the same arithmetic again.
@pytest.mark.parametrize("window", [(128, 0), (0, 0)], ids=lambda w: "win%d" % w[0])
@pytest.mark.parametrize("mask", ["causal", "causal_bottom_right", "no_mask"])
@pytest.mark.parametrize("sq,skv", [(1024, 1024), (512, 1024)], ids=["square", "rect"])
def test_frost_sliding_window_matches_reference(mask, window, sq, skv):
    """Sliding window against the float64 reference.

    The engine advertises swa support, and cuDNN expresses a window as a left bound on the same
    diagonal band that gives causal masking, so this shares a code path with the cases above. It
    is worth its own test because a left bound that is off by one, or silently dropped, still
    produces finite plausible-looking output -- the reference is the only thing that catches it.
    """
    # The rectangular case is the one that matters for alignment: top-left and bottom-right
    # coincide when sq == skv, so a swapped alignment is invisible in square shapes.
    b, hq, hkv, d = 2, 8, 4, 512
    dtype = torch.bfloat16
    torch.manual_seed(0)
    mk = lambda s_, h_: torch.randn(b, s_, h_, d, device="cuda")
    q32, k32, v32 = mk(sq, hq), mk(skv, hkv), mk(skv, hkv)
    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)
    scale = 1.0 / math.sqrt(d)

    out, _ = _fwd(q, k, v, mask, scale, window)

    floor_o, _, ref_o, _ = _floor(_bhsd(q32), _bhsd(k32), _bhsd(v32), scale, mask, dtype, window)
    err = (_bhsd(out).double() - ref_o).abs().max().item()
    assert torch.isfinite(out).all(), "sliding-window forward produced non-finite values"
    assert err <= 2 * floor_o + 1e-3, "out err %.3e exceeds 2x the floor %.3e for window %s" % (
        err,
        floor_o,
        window,
    )

    # A window must actually change the result; if the bound were dropped this would match the
    # unwindowed output and the check above would still pass.
    full, _ = _fwd(q, k, v, mask, scale)
    assert not torch.equal(out, full), "window %s produced the same output as no window" % (window,)


@requires_frost
@pytest.mark.parametrize("shape", _SHAPES[:2] + _SHAPES[-1:], ids=_shape_id)
@pytest.mark.parametrize("mask", ["no_mask", "causal"])
@pytest.mark.parametrize("window", [None, (128, 0)], ids=["nowin", "win128"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_frost_backward_matches_reference(shape, mask, window, dtype):
    """dq/dk/dv against autograd on the same independent float64 reference.

    Both dtypes, not just bf16: fp16 has a much narrower exponent range, and the backward is where
    that would show first -- the gradient of a softmax involves a subtraction of similarly sized
    terms, so a range problem surfaces there before it surfaces in the forward.
    """
    b, hq, hkv, sq, skv, d = shape
    torch.manual_seed(0)
    mk = lambda s_, h_, d_: torch.randn(b, s_, h_, d_, device="cuda")
    q32, k32, v32 = mk(sq, hq, d), mk(skv, hkv, d), mk(skv, hkv, d)
    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)
    scale = 1.0 / math.sqrt(d)

    out, lse = _fwd(q, k, v, mask, scale, window)
    dout = torch.randn_like(out)
    dq, dk, dv = _bwd(q, k, v, out, lse, dout, mask, scale, window)

    # The reference works in [b, h, s, d], so it takes views and returns grads in that order.
    qr = _bhsd(q32).detach().clone().requires_grad_(True)
    kr = _bhsd(k32).detach().clone().requires_grad_(True)
    vr = _bhsd(v32).detach().clone().requires_grad_(True)
    ref_o, _ = _reference(qr, kr, vr, scale, mask, window)
    ref_o.backward(_bhsd(dout).double())

    for name, got, want in (
        ("dq", _bhsd(dq), qr.grad),
        ("dk", _bhsd(dk), kr.grad),
        ("dv", _bhsd(dv), vr.grad),
    ):
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


def _frost_params(**overrides):
    """A FusedAttentionParams for a config FROST serves, with fields overridable by name."""
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        FusedAttentionParams,
    )
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        AttnBiasType,
        AttnMaskType,
        QKVFormat,
        QKVLayout,
        SoftmaxType,
    )
    from transformer_engine.pytorch.constants import TE_DType

    fields = dict(
        head_dim_qk=512,
        head_dim_v=512,
        qkv_dtype=TE_DType[torch.bfloat16],
        attn_mask_type=AttnMaskType["causal"],
        bias_type=AttnBiasType["no_bias"],
        softmax_type=SoftmaxType["vanilla"],
        qkv_layout=QKVLayout["bshd_bshd_bshd"],
        o_format=QKVFormat["bshd"],
        window_size_left=-1,
        window_size_right=-1,
        bottom_right_diagonal=False,
    )
    fields.update(overrides)
    return FusedAttentionParams(**fields)


@requires_frost
def test_frost_declines_unsupported_configs():
    """The selector must decline what the kernels do not serve, rather than computing wrongly."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        is_frost_attention_supported,
    )
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        AttnBiasType,
        AttnMaskType,
        FusedAttnBackend,
        QKVFormat,
        QKVLayout,
    )
    from transformer_engine.pytorch.constants import TE_DType

    assert (
        is_frost_attention_supported(_frost_params())[0] == FusedAttnBackend.FROST
    ), "the supported case must be accepted"

    for override, why in (
        (dict(head_dim_qk=256, head_dim_v=256), "head_dim at the exclusive lower bound"),
        (dict(head_dim_v=256), "head_dim_v below the range"),
        # The forward serves an asymmetric pair and the backward does not, so the selector
        # declines it rather than accepting a config whose backward cannot build.
        (dict(head_dim_v=320), "asymmetric head_dim"),
        (dict(qkv_dtype=TE_DType[torch.float32]), "fp32"),
        (dict(dropout=0.1), "dropout"),
        (dict(bias_type=AttnBiasType["post_scale_bias"]), "attention bias"),
        (dict(attn_mask_type=AttnMaskType["padding_causal"]), "padding mask"),
        (dict(qkv_layout=QKVLayout["thd_thd_thd"], o_format=QKVFormat["thd"]), "thd layout"),
        (dict(o_format=QKVFormat["sbhd"]), "an output format that differs from the input"),
        (dict(num_pages_k=4, num_pages_v=4), "paged KV"),
        (dict(return_max_logit=True), "max_logit"),
        (dict(cuda_graph=True), "CUDA graph capture"),
        (dict(deterministic=True, is_training=True), "a deterministic backward"),
        # window_size reaches _mask_spec through the selector, so its validation is part of the
        # selector contract rather than an internal detail.
        (dict(window_size_right=5), "a right window past the diagonal"),
        (dict(window_size_left=-2, window_size_right=0), "a left window below -1"),
        # A right-bounded window on a non-causal mask takes its anchor only from
        # bottom_right_diagonal; the all-gather ring measures its window bottom-right. Those
        # differ exactly when the lengths do.
        (
            dict(
                attn_mask_type=AttnMaskType["no_mask"],
                window_size_left=128,
                window_size_right=0,
                max_seqlen_q=512,
                max_seqlen_kv=640,
            ),
            "an ambiguous diagonal anchor",
        ),
        # The engine pads head_dim to a multiple of 8, so an in-range but unpadded dim has to be
        # declined here rather than failing later at plan selection.
        (dict(head_dim_qk=260, head_dim_v=260), "head_dim not a multiple of 8"),
        # Each dim is checked on its own, so v has to be covered as well as q.
        (dict(head_dim_v=260), "head_dim_v not a multiple of 8"),
    ):
        backend, reason = is_frost_attention_supported(_frost_params(**override))
        assert backend == FusedAttnBackend.No_Backend, "%s must be declined" % why
        assert reason, "a decline must explain itself"


@requires_frost
def test_frost_serves_an_unambiguous_window_on_a_non_causal_mask():
    """The anchor is only ambiguous when the q and kv lengths differ; equal lengths must serve."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        is_frost_attention_supported,
    )
    from transformer_engine.pytorch.cpp_extensions.fused_attn import AttnMaskType, FusedAttnBackend

    params = _frost_params(
        attn_mask_type=AttnMaskType["no_mask"],
        window_size_left=128,
        window_size_right=0,
        max_seqlen_q=4096,
        max_seqlen_kv=4096,
    )
    assert is_frost_attention_supported(params)[0] == FusedAttnBackend.FROST


@requires_frost
def test_frost_mask_spec_rejects_malformed_windows():
    """_mask_spec is the only validation between a caller-supplied window and a built band."""
    from transformer_engine.pytorch.attention.dot_product_attention.frost_attention import (
        _mask_spec,
    )

    for window, why in (
        ((128,), "a malformed window pair"),
        (7, "a non-iterable window"),
        ((-1, 5), "a right window past the diagonal"),
        ((-2, 0), "a left window below -1"),
    ):
        with pytest.raises(NotImplementedError):
            _mask_spec("causal", window), why


# Kept despite the CI-cost review: it launches no kernel, so removing it would have been
# coverage given up for no time back. It is the only end-to-end get_attention_backend call
# here, and the only place FROST, a sliding window and context parallelism meet.
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
    from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend

    use_fused, fused_backend = get_attention_backend(params)[2:4]
    use_frost = bool(use_fused) and fused_backend == FusedAttnBackend.FROST
    assert (
        use_frost == expect_frost
    ), "cp_comm_type=%s window=%s: expected the FROST sub-backend=%s, got %s" % (
        cp_comm_type,
        window,
        expect_frost,
        use_frost,
    )


@requires_frost
def test_frost_rejects_mismatched_kv():
    """v must index the same KV positions as k. head_dim is free; the rest is not."""
    b, h, s, d = 2, 4, 512, 512
    dtype = torch.bfloat16
    mk = lambda hh: torch.randn(b, s, hh, d, device="cuda", dtype=dtype)
    q, k = mk(h), mk(h)

    with pytest.raises(ValueError, match="batch, heads and seqlen"):
        _fwd(q, k, mk(h * 2), "no_mask", 1.0)
    with pytest.raises(ValueError, match="match q"):
        _fwd(q, k, k.to(torch.float32), "no_mask", 1.0)


def test_frost_engines_are_enabled_even_if_cudnn_was_imported_without_them():
    """Enabling the FROST engines must not depend on who imported cuDNN first.

    is_frost_attention_available imports cuDNN WITHOUT the engines, because enabling them reorders
    plan selection for every cuDNN consumer in the process and the checks after it may still
    decline. So the enabling cannot sit inside the "already imported?" memo: a process that
    probed availability first would otherwise leave FROST with a cuDNN that offers it no engine.
    That surfaces far from its cause, as "no cuDNN engine matching 'sdpa_fwd_prefill_sm100' was
    offered" on the first head_dim 512 forward, with a hint pointing at package versions that are
    in fact fine.

    No GPU and no real cuDNN: a stub stands in for the package, because what is under test is the
    order-dependence of our own wrapper. It also has to run in-process with the globals reset,
    since the real order is decided once per process and pytest gives us no second one.
    """
    import sys
    import types

    from transformer_engine.pytorch.attention.dot_product_attention import (
        cudnn_pygraph,
        frost_attention,
    )

    # The import memo and the switch live in the shared module; frost's wrapper only supplies the
    # default. Drive it through the wrapper, which is the real entry, and assert on the owner.
    env = "CUDNN_FRONTEND_ENABLE_FROST_ENGINES"
    saved = (
        cudnn_pygraph._cudnn,
        cudnn_pygraph._frost_engines_enabled,
        os.environ.get(env),
        sys.modules.get("cudnn"),
        sys.modules.get("cudnn.sdpa"),
    )
    try:
        stub = types.ModuleType("cudnn")
        stub.sdpa = types.ModuleType("cudnn.sdpa")
        sys.modules["cudnn"] = stub
        sys.modules["cudnn.sdpa"] = stub.sdpa
        cudnn_pygraph._cudnn = None
        cudnn_pygraph._frost_engines_enabled = False
        os.environ.pop(env, None)

        # The availability probe first, which must not enable anything.
        frost_attention._import_cudnn_frontend(enable_frost_engines=False)
        assert env not in os.environ, "the non-FROST caller must not set the switch"
        assert not cudnn_pygraph.frost_engines_enabled()

        # A use site second, on an already-imported cuDNN. This is the case that used to be
        # skipped.
        frost_attention._import_cudnn_frontend(enable_frost_engines=True)
        assert os.environ.get(env) == "1", "FROST was requested after the import and not enabled"
        assert cudnn_pygraph.frost_engines_enabled()
    finally:
        (
            cudnn_pygraph._cudnn,
            cudnn_pygraph._frost_engines_enabled,
            prior_env,
            prior_cudnn,
            prior_sdpa,
        ) = saved
        if prior_env is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = prior_env
        for name, module in (("cudnn", prior_cudnn), ("cudnn.sdpa", prior_sdpa)):
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_pinned_plan_decline_reports_the_engine_reason():
    """A pinned engine that refuses the graph must say why, not raise bare.

    The name lookup failing and the engine declining after selection are the two ways the strict
    path fails, and they read very differently: the second means the engine was there and judged
    this graph unservable, so cuDNN's own reason is the only thing identifying which constraint
    was missed. Without this the exception escaped with neither the reason framed nor the version
    hint attached.

    No GPU: a stub graph stands in, raising the real cuDNN exception type.
    """
    from transformer_engine.pytorch.attention.dot_product_attention import (
        cudnn_pygraph,
        frost_attention,
    )

    try:
        cudnn = frost_attention._import_cudnn_frontend()
    except ImportError:
        pytest.skip("cuDNN frontend Python package is required for the decline-reason path.")

    class _DeclinedGraph:
        """Offers the wanted plan, then refuses it at check_support."""

        def validate(self):
            pass

        def build_operation_graph(self):
            pass

        def create_execution_plans(self, _heuristics):
            pass

        def get_execution_plan_count(self):
            return 1

        def get_plan_name_at_index(self, _i):
            return "sdpa_fwd_prefill_sm100"

        def select_plan(self, _i):
            pass

        def check_support(self):
            raise cudnn.cudnnGraphNotSupportedError("head_dim 512 needs SM100; this is SM90")

    with pytest.raises(RuntimeError) as excinfo:
        cudnn_pygraph.finalize_plans(
            _DeclinedGraph(),
            backend_name="FrostAttention",
            heuristics=[cudnn.heur_mode.A],
            require_plan_token="sdpa_fwd_prefill_sm100",
            not_found_hint="nvidia-cutlass-dsl=4.8.0.",
        )

    message = str(excinfo.value)
    assert "sdpa_fwd_prefill_sm100" in message, "the message must name the engine that declined"
    assert "needs SM100" in message, "cuDNN's own reason must survive"
    assert "nvidia-cutlass-dsl" in message, "the version hint must be attached here too"
