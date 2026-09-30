# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.ops.fused.grouped_mlp import (
    GroupedMLP_CuTeGEMMGLU,
    fuse_glu_ops,
)
from transformer_engine.pytorch.ops.fuser import OperationFuser
from transformer_engine.pytorch.utils import deinterleave_glu_tensor
from transformer_engine.pytorch.models import DeepSeekV3MoE, MultiLatentAttention
from utils import make_recipe, quantization_tols

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


@pytest.mark.parametrize("quantization", ["mxfp8", "nvfp4"])
def test_moe_fused_quantized_uneven_expert_rows(monkeypatch, quantization):
    available, reason = getattr(te, f"is_{quantization}_available")(return_reason=True)
    if not available:
        pytest.skip(reason)

    recipe = make_recipe(quantization)
    torch.manual_seed(0)
    moe = DeepSeekV3MoE(HIDDEN, 128, num_experts=2, topk=1, params_dtype=DTYPE)
    fused_ops = fuse_glu_ops(list(moe.experts), recipe=recipe)
    if (
        fuse_glu_ops not in OperationFuser.forward_backward_fusion_functions
        or len(fused_ops) != 1
        or not isinstance(fused_ops[0], GroupedMLP_CuTeGEMMGLU)
    ):
        pytest.skip("requires the fused grouped MLP")
    reference = DeepSeekV3MoE(HIDDEN, 128, num_experts=2, topk=1, params_dtype=DTYPE)
    reference.load_state_dict(moe.state_dict())
    with torch.no_grad():
        for module in (moe, reference):
            module.gate.weight.zero_()
            module.gate.weight[0, 0] = 1
            module.gate.weight[1, 0] = -1

    tokens = torch.randn(512, HIDDEN, device="cuda", dtype=DTYPE)
    tokens[:128, 0] = 2
    tokens[128:, 0] = -2
    x = tokens.detach().requires_grad_()
    x_ref = tokens.detach().clone().requires_grad_()
    grad = torch.randn_like(tokens)
    splits = []
    moe.experts.register_forward_pre_hook(lambda _module, args: splits.append(args[1].clone()))

    with te.autocast(enabled=True, recipe=recipe):
        out = moe(x)
    out.backward(grad.clone())
    assert torch.equal(splits[0], torch.tensor([256, 512], device="cuda"))
    fused_op = moe.experts._module_groups[0]._forward_ops[0][0]
    assert isinstance(fused_op, GroupedMLP_CuTeGEMMGLU)

    monkeypatch.setattr(
        OperationFuser,
        "forward_backward_fusion_functions",
        [fn for fn in OperationFuser.forward_backward_fusion_functions if fn is not fuse_glu_ops],
    )
    ref_splits = []
    reference.experts.register_forward_pre_hook(
        lambda _module, args: ref_splits.append(args[1].clone())
    )
    with te.autocast(enabled=True, recipe=make_recipe(quantization)):
        ref_out = reference(x_ref)
    ref_out.backward(grad.clone())
    assert torch.equal(ref_splits[0], splits[0])

    tols = quantization_tols(quantization)
    torch.testing.assert_close(out, ref_out, **tols)
    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    ref_params = dict(reference.named_parameters())
    for name, param in moe.named_parameters():
        torch.testing.assert_close(param.grad, ref_params[name].grad, **tols)
