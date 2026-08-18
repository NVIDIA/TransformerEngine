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
