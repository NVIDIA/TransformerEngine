# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

from transformer_engine.pytorch.utils import deinterleave_glu_tensor
from transformer_engine.pytorch.models import (
    DeepSeekV3Layer,
    DeepSeekV3MoE,
    MultiLatentAttention,
)

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


def test_mla_forward_backward():
    torch.manual_seed(0)
    mla = MultiLatentAttention(HIDDEN, HEADS, params_dtype=DTYPE, **MLA_KWARGS)
    x = _input()
    out = mla(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("shared", [False, True], ids=["no_shared", "shared"])
@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
def test_moe_forward_backward(shared, grouped):
    torch.manual_seed(0)
    moe = DeepSeekV3MoE(
        HIDDEN,
        moe_ffn_hidden_size=128,
        num_experts=8,
        topk=2,
        num_groups=4 if grouped else None,
        group_topk=2 if grouped else None,
        shared_expert_ffn_hidden_size=128 if shared else None,
        params_dtype=DTYPE,
    )
    x = _input()
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    counts = moe._last_tokens_per_expert
    assert counts.sum().item() == SEQ_LEN * BATCH * 2
    bias_before = moe.expert_bias.clone()
    moe.update_expert_bias()
    assert not torch.equal(bias_before, moe.expert_bias)


def test_moe_matches_dense_reference():
    """topk == num_experts with uniform probs must reduce to a sum of expert MLPs."""
    torch.manual_seed(0)
    num_experts = 4
    moe = DeepSeekV3MoE(
        HIDDEN,
        moe_ffn_hidden_size=128,
        num_experts=num_experts,
        topk=num_experts,
        routed_scaling_factor=1.0,
        params_dtype=DTYPE,
    )
    x = _input(requires_grad=False)
    out = moe(x)

    tokens = x.reshape(-1, HIDDEN)
    probs, _ = moe._route(moe.gate(tokens).float())
    fc1, _, fc2 = moe.experts
    ref = torch.zeros_like(tokens)
    for e in range(num_experts):
        w1 = deinterleave_glu_tensor(getattr(fc1, f"weight{e}"), 32)
        w2 = getattr(fc2, f"weight{e}")
        gate_part, lin_part = (tokens @ w1.t()).chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate_part.float()) * lin_part.float()
        ref += (act.to(DTYPE) * probs[:, e : e + 1].to(DTYPE)) @ w2.t()
    torch.testing.assert_close(out.reshape(-1, HIDDEN), ref, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("num_experts", [None, 8], ids=["dense", "moe"])
def test_layer_forward_backward(num_experts):
    torch.manual_seed(0)
    layer = (
        DeepSeekV3Layer(
            HIDDEN,
            HEADS,
            ffn_hidden_size=512,
            num_experts=num_experts,
            moe_ffn_hidden_size=128 if num_experts else None,
            topk=2 if num_experts else None,
            shared_expert_ffn_hidden_size=128 if num_experts else None,
            params_dtype=DTYPE,
            **MLA_KWARGS,
        )
        if num_experts
        else DeepSeekV3Layer(HIDDEN, HEADS, ffn_hidden_size=512, params_dtype=DTYPE, **MLA_KWARGS)
    )
    x = _input()
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
