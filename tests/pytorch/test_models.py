# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math

import pytest
import torch

from transformer_engine.pytorch.utils import deinterleave_glu_tensor
from transformer_engine.pytorch.models import DeepSeekV3MoE, MultiLatentAttention

SEQ_LEN = 128
BATCH = 2
HIDDEN = 256
HEADS = 4
DTYPE = torch.bfloat16

MLA_KWARGS = dict(
    q_lora_rank=96,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
)


def _input(requires_grad=True):
    torch.manual_seed(1234)
    return torch.randn(
        SEQ_LEN, BATCH, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=requires_grad
    )


def test_mla_rope_triton_matches_pytorch():
    from transformer_engine.pytorch.models.deepseek_v3 import mla_rope

    if not mla_rope.HAVE_TRITON:
        pytest.skip("Triton unavailable")
    s, b, h = 64, 2, 4
    nope, rope, vdim = 64, 32, 64
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
    from transformer_engine.pytorch.models.deepseek_v3 import mla_rope

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


@pytest.mark.parametrize("mscale_all_dim", [0.0, 1.0])
def test_mla_yarn_softmax_scale(mscale_all_dim):
    mla = MultiLatentAttention(
        HIDDEN,
        HEADS,
        params_dtype=DTYPE,
        rope_scaling_factor=40.0,
        original_max_position_embeddings=64,
        mscale_all_dim=mscale_all_dim,
        **MLA_KWARGS,
    )
    m = 0.1 * mscale_all_dim * math.log(40.0) + 1.0
    qk_head_dim = MLA_KWARGS["qk_nope_head_dim"] + MLA_KWARGS["qk_rope_head_dim"]
    assert mla.softmax_scale == pytest.approx(m * m / math.sqrt(qk_head_dim))


@pytest.mark.parametrize("shared", [False, True], ids=["no_shared", "shared"])
@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
@pytest.mark.parametrize("topk", [2, 4])
def test_moe_matches_dense_reference(shared, grouped, topk):
    """Routed output must equal the prob-weighted sum of the selected expert MLPs."""
    torch.manual_seed(0)
    num_experts = 4
    moe = DeepSeekV3MoE(
        HIDDEN,
        moe_ffn_hidden_size=128,
        num_experts=num_experts,
        topk=topk,
        num_groups=2 if grouped else None,
        group_topk=topk // 2 if grouped else None,
        shared_expert_ffn_hidden_size=128 if shared else None,
        params_dtype=DTYPE,
    )
    x = _input()
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert torch.isfinite(x.grad).all()

    tokens = x.detach().reshape(-1, HIDDEN)
    probs, _ = moe._route(moe.gate(tokens).float())
    assert (probs > 0).sum(dim=1).eq(topk).all()
    assert moe._last_tokens_per_expert.sum().item() == tokens.shape[0] * topk

    fc1, _, fc2 = moe.experts
    ref = torch.zeros_like(tokens)
    for e in range(num_experts):
        w1 = deinterleave_glu_tensor(getattr(fc1, f"weight{e}"), 32)
        w2 = getattr(fc2, f"weight{e}")
        gate_part, lin_part = (tokens @ w1.t()).chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate_part.float()) * lin_part.float()
        ref += (act.to(DTYPE) * probs[:, e : e + 1].to(DTYPE)) @ w2.t()
    if shared:
        ref += moe.shared_expert(tokens)
    torch.testing.assert_close(out.reshape(-1, HIDDEN), ref, rtol=0.05, atol=0.05)

    bias_before = moe.expert_bias.clone()
    moe.update_expert_bias()
    assert torch.isfinite(moe.expert_bias).all()
    if topk < num_experts:
        assert not torch.equal(bias_before, moe.expert_bias)
