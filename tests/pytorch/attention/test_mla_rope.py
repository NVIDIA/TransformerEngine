# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math

import pytest
import torch

from transformer_engine.pytorch.attention import mla_rope


@pytest.mark.parametrize("nope,rope,vdim", [(64, 32, 64), (48, 32, 64), (64, 48, 64), (64, 32, 48)])
def test_mla_rope_matches_pytorch(nope, rope, vdim):
    if not mla_rope.HAVE_TRITON:
        pytest.skip("Triton unavailable")
    s, b, h = 64, 2, 4
    cos, sin = mla_rope.build_rope_tables(s, rope, device="cuda")

    torch.manual_seed(0)
    q_leaf = torch.randn(s, b, h, nope + rope, device="cuda", requires_grad=True)
    kv_leaf = torch.randn(s, b, h, nope + vdim, device="cuda", requires_grad=True)
    pos_leaf = torch.randn(s, b, 1, rope, device="cuda", requires_grad=True)
    grad_q = torch.randn(s, b, h, nope + rope, device="cuda")
    grad_k = torch.randn(s, b, h, nope + rope, device="cuda")
    grad_v = torch.randn(s, b, h, vdim, device="cuda")

    def run(fmt):
        # non-leaf copies: the Triton q kernel rotates in place
        q, kv, pos = q_leaf * 1.0, kv_leaf * 1.0, pos_leaf * 1.0
        q_out = mla_rope.apply_mla_rope_q(q, cos, sin, nope, rope, fmt)
        k_out, v_out = mla_rope.apply_mla_rope_kv(kv, pos, cos, sin, nope, rope, vdim, fmt)
        # fresh grad clones: the Triton q backward modifies its input grad in place
        torch.autograd.backward(
            [q_out, k_out, v_out], [grad_q.clone(), grad_k.clone(), grad_v.clone()]
        )
        grads = (q_leaf.grad.clone(), kv_leaf.grad.clone(), pos_leaf.grad.clone())
        q_leaf.grad = kv_leaf.grad = pos_leaf.grad = None
        return (q_out.clone(), k_out, v_out), grads

    (q_t, k_t, v_t), grads_t = run("sbhd")

    seq_dim = 0
    q_ref = torch.cat(
        (
            (q_leaf * 1.0)[..., :nope],
            mla_rope._rotate_interleaved_to_neox((q_leaf * 1.0)[..., nope:], cos, sin, seq_dim),
        ),
        dim=-1,
    )
    k_ref = torch.cat(
        (
            (kv_leaf * 1.0)[..., :nope],
            mla_rope._rotate_interleaved_to_neox(pos_leaf * 1.0, cos, sin, seq_dim).expand(
                s, b, h, rope
            ),
        ),
        dim=-1,
    )
    v_ref = (kv_leaf * 1.0)[..., nope:]
    torch.autograd.backward([q_ref, k_ref, v_ref], [grad_q.clone(), grad_k.clone(), grad_v.clone()])

    torch.testing.assert_close(q_t, q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_t, k_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_t, v_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads_t[0], q_leaf.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads_t[1], kv_leaf.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads_t[2], pos_leaf.grad, rtol=1e-5, atol=1e-5)


def test_rope_tables_yarn():
    s, rope = 8192, 64
    cos, sin = mla_rope.build_rope_tables(s, rope, device="cuda")
    cos_none, sin_none = mla_rope.build_rope_tables(s, rope, device="cuda", scaling_factor=None)
    assert torch.equal(cos, cos_none) and torch.equal(sin, sin_none)

    yarn = dict(scaling_factor=40.0, original_max_position_embeddings=4096)
    cos_y, sin_y = mla_rope.build_rope_tables(s, rope, device="cuda", **yarn)
    factor = mla_rope.yarn_concentration_factor(40.0, 1.0, 0.0)
    assert factor == pytest.approx(0.1 * math.log(40.0) + 1.0)
    # amplitude scaled by the concentration factor
    torch.testing.assert_close(cos_y**2 + sin_y**2, torch.full_like(cos_y, factor**2))
    # high-frequency dims untouched, low-frequency dims interpolated by 1/scaling_factor
    torch.testing.assert_close(cos_y[:, 0] / factor, cos[:, 0])
    angle_y = torch.atan2(sin_y[:, rope // 2 - 1], cos_y[:, rope // 2 - 1])
    angle = torch.atan2(sin[:, rope // 2 - 1], cos[:, rope // 2 - 1])
    torch.testing.assert_close(angle_y[:64], angle[:64] / 40.0, atol=1e-4, rtol=0)
