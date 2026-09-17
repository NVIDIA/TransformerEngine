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


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_experts": 0}, "num_experts must be positive"),
        ({"topk": 0}, "topk must be in"),
        ({"topk": 9}, "topk must be in"),
        ({"num_groups": 2}, "must be provided together"),
        ({"group_topk": 1}, "must be provided together"),
        ({"num_groups": 0, "group_topk": 1}, "num_groups must be positive"),
        ({"num_groups": -2, "group_topk": 1}, "num_groups must be positive"),
        ({"num_groups": 3, "group_topk": 1}, "divide num_experts"),
        ({"num_groups": 2, "group_topk": 0}, "group_topk must be in"),
        ({"num_groups": 2, "group_topk": -1}, "group_topk must be in"),
        ({"num_groups": 2, "group_topk": 3}, "group_topk must be in"),
        ({"num_groups": 2, "group_topk": 2, "topk": 3}, "topk must be divisible"),
        ({"num_groups": 4, "group_topk": 1, "topk": 4}, "topk per group must not exceed"),
    ],
)
def test_moe_rejects_invalid_routing_config(kwargs, match):
    config = dict(num_experts=8, topk=2)
    config.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DeepSeekV3MoE(HIDDEN, moe_ffn_hidden_size=128, device="cpu", **config)


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
