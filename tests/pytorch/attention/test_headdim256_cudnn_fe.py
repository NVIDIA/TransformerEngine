# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Smoke test for head_dim=256 via the cuDNN frontend Python SDPA (CuTe DSL) on SM100+."""

from __future__ import annotations

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention.dot_product_attention import cudnn_fe_sdpa


def _sm100_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


pytestmark = pytest.mark.skipif(
    not (_sm100_or_newer() and cudnn_fe_sdpa.is_available()),
    reason="Requires SM100+ GPU and cudnn frontend Python SDPA d=256 kernels",
)


def _reference(q, k, v, mask=None, scale=None):
    """Plain-attention reference in FP32."""
    d = q.shape[-1]
    scale = scale if scale is not None else 1.0 / (d**0.5)
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    # q: (B, H, S, D), k: (B, H, S, D), v: (B, H, S, D)
    s = torch.einsum("bhqd,bhkd->bhqk", q32, k32) * scale
    if mask is not None:
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", p, v32)
    return out.to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("attn_mask_type", ["no_mask", "causal"])
@pytest.mark.parametrize("seqlen", [512, 2048])
def test_cudnn_fe_fwd_bwd_bshd(dtype, attn_mask_type, seqlen):
    batch, heads, head_dim = 2, 4, 256
    torch.manual_seed(0)
    device = "cuda"

    # BSHD layout (torch-contiguous)
    q = torch.randn(batch, seqlen, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, seqlen, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, seqlen, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    d_o = torch.randn(batch, seqlen, heads, head_dim, dtype=dtype, device=device)

    window_size = (-1, 0) if attn_mask_type == "causal" else (-1, -1)

    # cuDNN-FE direct call
    out, aux_ctx = cudnn_fe_sdpa.fused_attn_fwd(
        max_seqlen_q=seqlen,
        max_seqlen_kv=seqlen,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        q=q,
        k=k,
        v=v,
        qkv_format="bshd",
        attn_mask_type=attn_mask_type,
        attn_scale=None,
        window_size=window_size,
    )
    assert out.shape == q.shape, f"Unexpected fwd out shape {out.shape}"

    dq, dk, dv = cudnn_fe_sdpa.fused_attn_bwd(
        max_seqlen_q=seqlen,
        max_seqlen_kv=seqlen,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        q=q,
        k=k,
        v=v,
        o=out,
        d_o=d_o,
        aux_ctx_tensors=aux_ctx,
        qkv_format="bshd",
        attn_mask_type=attn_mask_type,
        attn_scale=None,
        window_size=window_size,
    )
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape

    # FP32 reference over (B, H, S, D)
    q_ref = q.detach().float().transpose(1, 2).contiguous().requires_grad_(True)
    k_ref = k.detach().float().transpose(1, 2).contiguous().requires_grad_(True)
    v_ref = v.detach().float().transpose(1, 2).contiguous().requires_grad_(True)

    mask = None
    if attn_mask_type == "causal":
        mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=device), diagonal=1)
    out_ref = _reference(q_ref, k_ref, v_ref, mask=mask)
    # transpose back to BSHD for comparison
    out_ref_bshd = out_ref.transpose(1, 2).contiguous().to(dtype)
    out_ref.backward(d_o.transpose(1, 2).contiguous().float())

    tol = {"atol": 5e-2, "rtol": 5e-2}
    torch.testing.assert_close(out.float(), out_ref_bshd.float(), **tol)
    torch.testing.assert_close(
        dq.float(), q_ref.grad.transpose(1, 2).contiguous().to(dtype).float(), **tol
    )
    torch.testing.assert_close(
        dk.float(), k_ref.grad.transpose(1, 2).contiguous().to(dtype).float(), **tol
    )
    torch.testing.assert_close(
        dv.float(), v_ref.grad.transpose(1, 2).contiguous().to(dtype).float(), **tol
    )


def test_cudnn_fe_fused_attention_module(monkeypatch):
    """Integration test: exercise through the DotProductAttention module."""
    # Scope env var mutations to this test — pytest shares a process across
    # tests, so a bare ``os.environ[...] = ...`` would leak these flags into
    # every later test in the session.
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")

    dtype = torch.bfloat16
    batch, seqlen, heads, head_dim = 2, 1024, 4, 256
    device = "cuda"

    torch.manual_seed(42)
    q = torch.randn(batch, seqlen, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, seqlen, heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, seqlen, heads, head_dim, dtype=dtype, device=device, requires_grad=True)

    dpa = te.DotProductAttention(
        num_attention_heads=heads,
        kv_channels=head_dim,
        qkv_format="bshd",
        attention_type="self",
    ).cuda()
    out = dpa(q, k, v, attention_mask=None)
    assert out.shape == (batch, seqlen, heads * head_dim)

    loss = out.sum()
    loss.backward()
    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape
