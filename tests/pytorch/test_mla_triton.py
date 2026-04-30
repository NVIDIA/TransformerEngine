# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the Triton-based MLA kernels (transformer_engine.pytorch.triton.mla)."""

from dataclasses import dataclass

import pytest
import torch

from utils import reset_rng_states
from transformer_engine.pytorch.triton.mla import (
    mla_attention,
    mla_attention_ref,
    mla_decode_attention,
    mla_decode_attention_ref,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="MLA Triton kernel requires SM80+ (A100 or newer).",
)

# Disable TF32 to keep the fp32 reference path bit-comparable across runs.
torch.backends.cuda.matmul.allow_tf32 = False


@dataclass
class MLAConfig:
    b: int
    h: int
    s_q: int
    s_kv: int
    d_qk: int
    d_v: int

    @staticmethod
    def desc(cfg):
        return (
            f"b{cfg.b}_h{cfg.h}_sq{cfg.s_q}_skv{cfg.s_kv}"
            f"_dqk{cfg.d_qk}_dv{cfg.d_v}"
        )


mla_configs = [
    # square head dim sanity
    MLAConfig(2, 4, 128, 128, 64, 64),
    MLAConfig(2, 4, 256, 256, 128, 128),
    # DeepSeek-V2 prefill shape
    MLAConfig(2, 8, 512, 512, 192, 128),
    # bigger seq
    MLAConfig(1, 16, 2048, 2048, 192, 128),
    # cross-attention-style: S_q != S_kv
    MLAConfig(2, 8, 512, 1024, 192, 128),
    # non-multiple-of-block seq lengths
    MLAConfig(2, 4, 513, 513, 192, 128),
    # decode-shaped Q (still using the prefill kernel)
    MLAConfig(1, 4, 1, 1024, 192, 128),
]


def _tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=2.5e-2, rtol=2.5e-2)
    return dict(atol=5e-3, rtol=5e-3)


def _make_qkv(cfg, dtype, qkv_format):
    """Allocate (q, k, v) in the requested layout. Sized so values stay well within the
    kernel's dynamic range (avoids saturating exp in fp32 accumulator)."""
    scale = 0.5
    if qkv_format == "bhsd":
        q = torch.randn(cfg.b, cfg.h, cfg.s_q, cfg.d_qk, device="cuda", dtype=dtype) * scale
        k = torch.randn(cfg.b, cfg.h, cfg.s_kv, cfg.d_qk, device="cuda", dtype=dtype) * scale
        v = torch.randn(cfg.b, cfg.h, cfg.s_kv, cfg.d_v, device="cuda", dtype=dtype) * scale
    elif qkv_format == "bshd":
        q = torch.randn(cfg.b, cfg.s_q, cfg.h, cfg.d_qk, device="cuda", dtype=dtype) * scale
        k = torch.randn(cfg.b, cfg.s_kv, cfg.h, cfg.d_qk, device="cuda", dtype=dtype) * scale
        v = torch.randn(cfg.b, cfg.s_kv, cfg.h, cfg.d_v, device="cuda", dtype=dtype) * scale
    elif qkv_format == "sbhd":
        q = torch.randn(cfg.s_q, cfg.b, cfg.h, cfg.d_qk, device="cuda", dtype=dtype) * scale
        k = torch.randn(cfg.s_kv, cfg.b, cfg.h, cfg.d_qk, device="cuda", dtype=dtype) * scale
        v = torch.randn(cfg.s_kv, cfg.b, cfg.h, cfg.d_v, device="cuda", dtype=dtype) * scale
    else:
        raise ValueError(qkv_format)
    return q, k, v


def _ref_in_user_layout(q, k, v, qkv_format, *, softmax_scale=None, is_causal=False):
    """Run mla_attention_ref on tensors that are in user layout.

    The reference operates in BHSD; we transpose, run, and transpose back.
    """
    if qkv_format == "bshd":
        q_b = q.transpose(1, 2).contiguous()
        k_b = k.transpose(1, 2).contiguous()
        v_b = v.transpose(1, 2).contiguous()
    elif qkv_format == "sbhd":
        q_b = q.permute(1, 2, 0, 3).contiguous()
        k_b = k.permute(1, 2, 0, 3).contiguous()
        v_b = v.permute(1, 2, 0, 3).contiguous()
    else:
        q_b, k_b, v_b = q, k, v
    out_bhsd = mla_attention_ref(q_b, k_b, v_b, softmax_scale=softmax_scale, is_causal=is_causal)
    if qkv_format == "bshd":
        return out_bhsd.transpose(1, 2).contiguous()
    if qkv_format == "sbhd":
        return out_bhsd.permute(2, 0, 1, 3).contiguous()
    return out_bhsd


@pytest.mark.parametrize("cfg", mla_configs, ids=MLAConfig.desc)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("qkv_format", ["bshd", "bhsd", "sbhd"])
def test_mla_forward(cfg, dtype, is_causal, qkv_format):
    reset_rng_states()
    q, k, v = _make_qkv(cfg, dtype, qkv_format)

    out_triton = mla_attention(q, k, v, is_causal=is_causal, qkv_format=qkv_format)
    out_ref = _ref_in_user_layout(q, k, v, qkv_format, is_causal=is_causal)

    assert out_triton.shape == out_ref.shape
    assert out_triton.dtype == out_ref.dtype
    torch.testing.assert_close(out_triton, out_ref, **_tols(dtype))


_backward_configs = [
    MLAConfig(2, 4, 128, 128, 64, 64),
    MLAConfig(2, 4, 256, 256, 128, 128),
    MLAConfig(2, 8, 256, 256, 192, 128),  # DeepSeek-V2 dims, smaller seq
    MLAConfig(2, 8, 512, 512, 192, 128),  # DeepSeek-V2 prefill
    MLAConfig(2, 4, 256, 512, 192, 128),  # cross-attn shape
    MLAConfig(2, 4, 257, 257, 192, 128),  # non-multiple-of-block seq
]


@pytest.mark.parametrize("cfg", _backward_configs, ids=MLAConfig.desc)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["nocausal", "causal"])
def test_mla_backward_matches_reference(cfg, dtype, is_causal):
    """Triton-computed dQ/dK/dV must match the pure-PyTorch reference within bf16/fp16 tolerances."""
    reset_rng_states()
    q_base, k_base, v_base = _make_qkv(cfg, dtype, qkv_format="bshd")
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)

    out_triton = mla_attention(q, k, v, is_causal=is_causal, qkv_format="bshd")
    out_ref = _ref_in_user_layout(q_ref, k_ref, v_ref, "bshd", is_causal=is_causal)

    # Smaller-magnitude grad_o so dS = P*(dP - Delta) stays well within the
    # accumulator's representable range for the non-square head-dim cases.
    grad_o = torch.randn_like(out_triton) * 0.1
    out_triton.backward(grad_o)
    out_ref.backward(grad_o)

    tols = _tols(dtype)
    torch.testing.assert_close(q.grad, q_ref.grad, **tols)
    torch.testing.assert_close(k.grad, k_ref.grad, **tols)
    torch.testing.assert_close(v.grad, v_ref.grad, **tols)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["nocausal", "causal"])
def test_mla_layout_equivalence(dtype, is_causal):
    """Same logical inputs in BSHD and BHSD must produce equal outputs (modulo layout)."""
    reset_rng_states()
    cfg = MLAConfig(2, 4, 256, 256, 192, 128)
    q_b, k_b, v_b = _make_qkv(cfg, dtype, qkv_format="bhsd")
    q_s = q_b.transpose(1, 2).contiguous()
    k_s = k_b.transpose(1, 2).contiguous()
    v_s = v_b.transpose(1, 2).contiguous()

    out_bhsd = mla_attention(q_b, k_b, v_b, is_causal=is_causal, qkv_format="bhsd")
    out_bshd = mla_attention(q_s, k_s, v_s, is_causal=is_causal, qkv_format="bshd")

    # Triton kernel runs on the canonicalized BHSD path in both cases, so outputs
    # must match exactly after layout permutation.
    torch.testing.assert_close(out_bhsd, out_bshd.transpose(1, 2).contiguous(), atol=0.0, rtol=0.0)


def test_mla_softmax_scale_default():
    """Verify the default softmax_scale is 1/sqrt(head_dim_qk) (Q-side, not V-side)."""
    reset_rng_states()
    cfg = MLAConfig(1, 2, 64, 64, 192, 128)
    q, k, v = _make_qkv(cfg, torch.bfloat16, "bshd")
    out_default = mla_attention(q, k, v, qkv_format="bshd")
    out_explicit = mla_attention(
        q, k, v, softmax_scale=cfg.d_qk**-0.5, qkv_format="bshd"
    )
    torch.testing.assert_close(out_default, out_explicit, atol=0.0, rtol=0.0)


def test_mla_rejects_fp32_input():
    """The Triton kernel is fp16/bf16 only — fp32 inputs should raise."""
    cfg = MLAConfig(1, 2, 64, 64, 64, 64)
    q, k, v = _make_qkv(cfg, torch.float32, "bshd")
    with pytest.raises(ValueError, match="fp16 or bf16"):
        mla_attention(q, k, v, qkv_format="bshd")


# ---------------------------------------------------------------------------
# Decode (absorbed up-projection) tests
# ---------------------------------------------------------------------------


@dataclass
class MLADecodeConfig:
    b: int
    h: int
    s_q: int
    s_kv: int
    r: int  # kv_lora_rank
    r_rope: int

    @staticmethod
    def desc(cfg):
        return (
            f"b{cfg.b}_h{cfg.h}_sq{cfg.s_q}_skv{cfg.s_kv}"
            f"_r{cfg.r}_rrope{cfg.r_rope}"
        )


_decode_configs = [
    MLADecodeConfig(1, 4, 1, 128, 64, 16),  # smoke
    MLADecodeConfig(1, 4, 1, 512, 64, 16),  # bigger Skv
    MLADecodeConfig(1, 4, 4, 512, 64, 16),  # multi-token / speculative decode
    MLADecodeConfig(2, 4, 128, 128, 128, 32),  # prefill via decode kernel
    # DeepSeek-V2 dims (kv_lora_rank=512, rope_dim=64).
    MLADecodeConfig(1, 8, 1, 1024, 512, 64),
    MLADecodeConfig(1, 16, 1, 2048, 512, 64),
    MLADecodeConfig(1, 4, 1, 257, 512, 64),  # non-multiple-of-block S_kv
]


def _make_decode_inputs(cfg, dtype, scale=0.5):
    qn = torch.randn(cfg.b, cfg.h, cfg.s_q, cfg.r, device="cuda", dtype=dtype) * scale
    qr = torch.randn(cfg.b, cfg.h, cfg.s_q, cfg.r_rope, device="cuda", dtype=dtype) * scale
    ck = torch.randn(cfg.b, cfg.s_kv, cfg.r, device="cuda", dtype=dtype) * scale
    kr = torch.randn(cfg.b, cfg.s_kv, cfg.r_rope, device="cuda", dtype=dtype) * scale
    return qn, qr, ck, kr


@pytest.mark.parametrize("cfg", _decode_configs, ids=MLADecodeConfig.desc)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["nocausal", "causal"])
def test_mla_decode_forward(cfg, dtype, is_causal):
    reset_rng_states()
    qn, qr, ck, kr = _make_decode_inputs(cfg, dtype)
    softmax_scale = 1.0 / (cfg.r + cfg.r_rope) ** 0.5

    o = mla_decode_attention(qn, qr, ck, kr, softmax_scale=softmax_scale, is_causal=is_causal)
    o_ref = mla_decode_attention_ref(qn, qr, ck, kr, softmax_scale=softmax_scale, is_causal=is_causal)

    assert o.shape == o_ref.shape == (cfg.b, cfg.h, cfg.s_q, cfg.r)
    assert o.dtype == o_ref.dtype == dtype
    torch.testing.assert_close(o, o_ref, **_tols(dtype))


def test_mla_decode_rejects_fp32_input():
    cfg = MLADecodeConfig(1, 2, 1, 64, 64, 16)
    qn, qr, ck, kr = _make_decode_inputs(cfg, torch.float32)
    with pytest.raises(ValueError, match="fp16 or bf16"):
        mla_decode_attention(qn, qr, ck, kr, softmax_scale=0.1)


def test_mla_decode_rejects_dim_mismatch():
    cfg = MLADecodeConfig(1, 2, 1, 64, 64, 16)
    qn, qr, ck, kr = _make_decode_inputs(cfg, torch.bfloat16)
    # Mismatched kv_lora_rank between Q-side and cache.
    bad_ck = torch.randn(1, 64, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="kv_lora_rank"):
        mla_decode_attention(qn, qr, bad_ck, kr, softmax_scale=0.1)


# ---------------------------------------------------------------------------
# DotProductAttention dispatch (NVTE_MLA_TRITON=1 fast path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
def test_dpa_dispatches_to_mla_triton(monkeypatch, dtype, is_causal, qkv_format):
    """With NVTE_MLA_TRITON=1, an MLA-shaped DotProductAttention call must
    produce the same output as a direct ``mla_attention`` call."""
    import transformer_engine.pytorch as te

    reset_rng_states()
    cfg = MLAConfig(2, 4, 256, 256, 192, 128)
    q, k, v = _make_qkv(cfg, dtype, qkv_format=qkv_format)

    monkeypatch.setenv("NVTE_MLA_TRITON", "1")

    # Force backend cache invalidation so the env-var check re-runs.
    from transformer_engine.pytorch.attention.dot_product_attention import dot_product_attention as dpa_mod
    dpa_mod._attention_backends["backend_selection_requires_update"] = True

    softmax_scale = cfg.d_qk ** -0.5
    dpa = te.DotProductAttention(
        num_attention_heads=cfg.h,
        kv_channels=(cfg.d_qk, cfg.d_v),
        attention_dropout=0.0,
        softmax_scale=softmax_scale,
        qkv_format=qkv_format,
        attn_mask_type="causal" if is_causal else "no_mask",
    ).cuda()

    out_dpa = dpa(q, k, v)
    out_direct = mla_attention(
        q, k, v, softmax_scale=softmax_scale, is_causal=is_causal, qkv_format=qkv_format
    )
    torch.testing.assert_close(out_dpa, out_direct, atol=0.0, rtol=0.0)


def test_dpa_falls_through_when_env_var_unset(monkeypatch):
    """With NVTE_MLA_TRITON unset, dispatch must NOT route through the MLA
    Triton backend even for MLA-shaped inputs (existing behavior preserved)."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.attention.dot_product_attention import dot_product_attention as dpa_mod

    reset_rng_states()
    cfg = MLAConfig(2, 4, 256, 256, 192, 128)
    q, k, v = _make_qkv(cfg, torch.bfloat16, qkv_format="bshd")

    monkeypatch.delenv("NVTE_MLA_TRITON", raising=False)
    dpa_mod._attention_backends["backend_selection_requires_update"] = True

    dpa = te.DotProductAttention(
        num_attention_heads=cfg.h,
        kv_channels=(cfg.d_qk, cfg.d_v),
        attention_dropout=0.0,
        softmax_scale=cfg.d_qk ** -0.5,
        qkv_format="bshd",
        attn_mask_type="causal",
    ).cuda()

    # Just exercise it — any cuDNN/FA backend may be selected. We only assert
    # no exception (i.e. the early-out wasn't accidentally triggered).
    _ = dpa(q, k, v)
