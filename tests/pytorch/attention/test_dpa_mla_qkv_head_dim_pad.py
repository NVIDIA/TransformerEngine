# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the optional MLA head-dim pad in DotProductAttention.

Covers:
  * `should_pad_qkv_head_dim` decides correctly (native unfused vs padded fused).
  * DPA with `head_dim_v > head_dim_qk` runs and produces a V-width output.
  * The pad-then-trim is an identity for both `qk > v` and `v > qk`: padding Q/K/V to the
    wider head dim, running with the equal (padded) shape, and trimming back equals the
    native mismatched-dim run.
"""

import math
import pathlib
import sys

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import dot_product_attention as dpa_module
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent)] + sys.path
from utils import reset_rng_states


def _build_dpa(qk, v, num_heads=4, qkv_format="thd", attn_mask_type="padding_causal",
               softmax_scale=None):
    return DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=(qk, v),
        attention_type="self",
        attn_mask_type=attn_mask_type,
        qkv_format=qkv_format,
        softmax_scale=softmax_scale,
    ).to(dtype=torch.bfloat16, device="cuda")


def _thd_inputs(qk, v, t=32, h=4):
    cu = torch.IntTensor([0, 6, 19, 22, t]).cuda()
    q = torch.randn(t, h, qk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(t, h, qk, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(t, h, v, device="cuda", dtype=torch.bfloat16)
    return q, k, v, cu


def _run_dpa(dpa, q, k, v, cu, max_seqlen=13):
    return dpa(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_kv=cu,
        max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen,
        attn_mask_type="padding_causal",
    )


# should_pad_qkv_head_dim
@pytest.mark.parametrize("native_unfused,padded_fused,expected", [
    (False, False, False),   # native already fused -> no pad
    (True, False, False),    # both unfused -> no upgrade -> no pad
    (True, True, True),      # native unfused, padded fused -> pad
])
def test_should_pad_qkv_head_dim(monkeypatch, native_unfused, padded_fused, expected):
    """`should_pad_qkv_head_dim` returns True iff native is unfused and padded is fused."""
    params = dpa_utils.AttentionParams(
        qkv_layout="thd_thd_thd",
        num_heads=4, num_gqa_groups=4,
        max_seqlen_q=13, max_seqlen_kv=13,
        head_dim_qk=96, head_dim_v=128,
        attn_mask_type="padding_causal",
        is_training=True,
        qkv_dtype=torch.bfloat16,
    )

    # get_attention_backend returns
    # (use_flash, flash_backend, use_fused, fused_backend, use_unfused, available)
    native = (False, None, not native_unfused, None, native_unfused,
               [False, not native_unfused, native_unfused])
    padded = (False, None, padded_fused, None, not padded_fused,
               [False, padded_fused, not padded_fused])

    def fake_backend(p):
        # native probe: real (mismatched) head_dim_qk/v; padded probe: both = max(qk, v).
        # Distinguish by head_dim_qk (native=96, padded=max(96,128)=128).
        is_padded = p.head_dim_qk != params.head_dim_qk
        return padded if is_padded else native

    monkeypatch.setattr(dpa_utils, "get_attention_backend", fake_backend)
    assert dpa_utils.should_pad_qkv_head_dim(params) is expected


def test_should_pad_qkv_head_dim_equal_dims():
    """No pad when head_dim_qk == head_dim_v."""
    params = dpa_utils.AttentionParams(
        qkv_layout="thd_thd_thd", num_heads=4, num_gqa_groups=4,
        max_seqlen_q=13, max_seqlen_kv=13,
        head_dim_qk=128, head_dim_v=128,
        attn_mask_type="padding_causal", is_training=True, qkv_dtype=torch.bfloat16,
    )
    assert dpa_utils.should_pad_qkv_head_dim(params) is False


# v > qk end-to-end
@pytest.mark.parametrize("qk,v", [(64, 192), (96, 192)])
def test_dpa_v_gt_qk_runs(qk, v):
    """DPA with head_dim_v > head_dim_qk runs and produces a V-width output."""
    reset_rng_states()
    dpa = _build_dpa(qk, v)
    q, k, v_t, cu = _thd_inputs(qk, v)
    out = _run_dpa(dpa, q, k, v_t, cu)
    assert tuple(out.shape) == (32, 4 * v), out.shape  # V-width
    out.float().sum().backward()  # backward must not crash


# pad-then-trim is an identity (both directions)
@pytest.mark.parametrize("qk,v", [(192, 128), (64, 192)])
def test_dpa_mla_pad_is_identity(qk, v):
    """Pad-then-trim is an identity: padding Q/K/V to the wider head dim, running with the equal
    (padded) shape, and trimming back equals the native mismatched-dim run -- for both qk > v and v
    > qk. Both runs use the same `softmax_scale` (`1/sqrt(qk)`) that the production forward keeps
    when padding.
    """
    reset_rng_states()
    m = max(qk, v)
    scale = 1.0 / math.sqrt(qk)
    cu = torch.IntTensor([0, 6, 19, 22, 32]).cuda()

    # Reference: native mismatched-dim run (the production forward; it pads internally
    # only when should_pad_qkv_head_dim upgrades the selected backend).
    dpa_ref = _build_dpa(qk, v)  # softmax_scale defaults to 1/sqrt(qk)
    q, k, v_t, _ = _thd_inputs(qk, v)
    out_ref = _run_dpa(dpa_ref, q, k, v_t, cu)
    assert tuple(out_ref.shape) == (32, 4 * v), out_ref.shape

    # Test: manually pad to the common width, run with the equal (padded) shape, trim.
    # Same softmax_scale as the reference so pad-then-trim is a true identity.
    dpa = _build_dpa(m, m, softmax_scale=scale)
    q_p, k_p, v_p, _, _ = dpa_module._pad_qkv_head_dim(q, k, v_t)
    assert q_p.shape[-1] == k_p.shape[-1] == v_p.shape[-1] == m
    out = _run_dpa(dpa, q_p, k_p, v_p, cu)
    # Trim back to the original V width.
    out = dpa_module._trim_output(out, 4, m, v)
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
    out.float().sum().backward()  # padded path backward must not crash
