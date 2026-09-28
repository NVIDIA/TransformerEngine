# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoeDispatch and MoeCombine on the PyTorch backend: EP=1, no NCCL EP."""

import os
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.common.recipe import Float8CurrentScaling
from transformer_engine.pytorch import ops as te_ops
from transformer_engine.pytorch.ep import EpConfig


NUM_EXPERTS = 4
HIDDEN_DIM = 128
INTERMEDIATE_DIM = 64
NUM_TOKENS = 64
TOP_K = 2
DEVICE = "cuda"
TOLERANCES = {"rtol": 2e-2, "atol": 2e-2}


def _config(num_local_experts=NUM_EXPERTS, top_k=TOP_K):
    """EP=1 config: every expert of the group is local, so no buffer is needed."""
    return EpConfig(
        top_k=top_k,
        hidden_dim=HIDDEN_DIM,
        num_local_experts=num_local_experts,
        max_tokens_per_rank=NUM_TOKENS,
        recv_capacity_per_rank=None,
        ep_group=None,
    )


def _tokens(num_tokens=NUM_TOKENS):
    values = torch.linspace(-0.9, 0.9, num_tokens * HIDDEN_DIM, device=DEVICE)
    return values.reshape(num_tokens, HIDDEN_DIM).to(torch.bfloat16)


def _routing(num_tokens=NUM_TOKENS, num_experts=NUM_EXPERTS, top_k=TOP_K, seed=2026):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, generator=generator, device=DEVICE)
    topk_logits, topk_idx = torch.topk(logits, top_k, dim=-1)
    return topk_idx.to(torch.int64), torch.softmax(topk_logits, dim=-1).to(torch.float32)


def _expert_major_order(topk_idx):
    """Route of every recv row: ordered by expert, then by source token."""
    return (
        torch.nonzero((topk_idx.reshape(-1) >= 0), as_tuple=False)
        .flatten()
        .index_select(
            0, torch.argsort(topk_idx.reshape(-1)[topk_idx.reshape(-1) >= 0], stable=True)
        )
    )


def test_dispatch_gathers_tokens_in_expert_major_order():
    tokens = _tokens(4)
    topk_idx = torch.tensor([[2, 0], [1, 1], [0, 2], [2, 1]], dtype=torch.int64, device=DEVICE)
    topk_weights = torch.arange(1, 9, dtype=torch.float32, device=DEVICE).reshape(4, 2) / 10

    recv_tokens, tokens_per_expert, recv_weights = te_ops.MoeDispatch(_config())(
        tokens, topk_idx, topk_weights
    )

    recv_route = torch.tensor([1, 4, 2, 3, 7, 0, 5, 6], dtype=torch.int64, device=DEVICE)
    torch.testing.assert_close(recv_tokens, tokens[recv_route // TOP_K])
    torch.testing.assert_close(recv_weights, topk_weights.reshape(-1)[recv_route])
    torch.testing.assert_close(
        tokens_per_expert, torch.tensor([2, 3, 3, 0], dtype=torch.int64, device=DEVICE)
    )


def test_dispatch_then_combine_returns_the_source_tokens():
    tokens = _tokens()
    topk_idx, topk_weights = _routing()
    config = _config()

    recv_tokens, _, recv_weights = te_ops.MoeDispatch(config)(tokens, topk_idx, topk_weights)
    weighted = (recv_tokens.float() * recv_weights.unsqueeze(-1)).to(torch.bfloat16)
    output = te_ops.MoeCombine(config)(weighted, topk_idx)

    torch.testing.assert_close(output, tokens, **TOLERANCES)


def test_topk_one_round_trip():
    config = _config(top_k=1)
    tokens = _tokens()
    topk_idx, topk_weights = _routing(top_k=1)
    recv_tokens, _, recv_weights = te_ops.MoeDispatch(config)(tokens, topk_idx, topk_weights)
    output = te_ops.MoeCombine(config)(
        (recv_tokens.float() * recv_weights.unsqueeze(-1)).to(torch.bfloat16), topk_idx
    )
    torch.testing.assert_close(output, tokens, **TOLERANCES)


def test_two_outstanding_forwards_keep_their_routing():
    config = _config()
    dispatch = te_ops.MoeDispatch(config)
    combine = te_ops.MoeCombine(config)
    tokens_a = _tokens(4).requires_grad_(True)
    tokens_b = (-_tokens(4)).requires_grad_(True)
    topk_a, weights_a = _routing(num_tokens=4, seed=2026)
    topk_b, weights_b = _routing(num_tokens=4, seed=2027)

    recv_a, _, recv_weights_a = dispatch(tokens_a, topk_a, weights_a)
    recv_b, _, recv_weights_b = dispatch(tokens_b, topk_b, weights_b)
    output_a = combine(
        (recv_a.float() * recv_weights_a.unsqueeze(-1)).to(torch.bfloat16), topk_a
    )
    output_b = combine(
        (recv_b.float() * recv_weights_b.unsqueeze(-1)).to(torch.bfloat16), topk_b
    )
    (output_a.float().sum() + 2 * output_b.float().sum()).backward()

    torch.testing.assert_close(output_a, tokens_a, **TOLERANCES)
    torch.testing.assert_close(output_b, tokens_b, **TOLERANCES)
    torch.testing.assert_close(tokens_a.grad, torch.ones_like(tokens_a), **TOLERANCES)
    torch.testing.assert_close(tokens_b.grad, 2 * torch.ones_like(tokens_b), **TOLERANCES)


def test_combine_sums_routes_into_their_source_tokens():
    config = _config()
    topk_idx = torch.tensor([[1, 2], [0, 3]], dtype=torch.int64, device=DEVICE)
    expert_out = _tokens(4)

    # Expert-major rows hold (token 1, expert 0), then token 0's routes to experts
    # 1 and 2, then (token 1, expert 3).
    output = te_ops.MoeCombine(config)(expert_out, topk_idx)

    torch.testing.assert_close(output[0], expert_out[1] + expert_out[2], **TOLERANCES)
    torch.testing.assert_close(output[1], expert_out[0] + expert_out[3], **TOLERANCES)


def test_routing_gradients():
    tokens = _tokens().requires_grad_(True)
    topk_idx, topk_weights = _routing()
    topk_weights.requires_grad_(True)

    recv_tokens, _, recv_weights = te_ops.MoeDispatch(_config())(tokens, topk_idx, topk_weights)
    (recv_tokens.float() * recv_weights.unsqueeze(-1)).sum().backward()

    # A route's weight gradient is the hidden dimension sum of its source token.
    torch.testing.assert_close(
        topk_weights.grad,
        tokens.detach().float().sum(-1, keepdim=True).expand(-1, TOP_K),
        **TOLERANCES,
    )
    # A token's gradient is the sum of the weights of the routes it feeds.
    torch.testing.assert_close(
        tokens.grad.float(),
        topk_weights.detach().sum(-1, keepdim=True).expand(-1, HIDDEN_DIM),
        **TOLERANCES,
    )


def test_combine_backward_is_the_adjoint_of_combine():
    config = _config()
    topk_idx, _ = _routing()
    expert_out = _tokens(NUM_TOKENS * TOP_K).requires_grad_(True)
    output = te_ops.MoeCombine(config)(expert_out, topk_idx)
    grad_output = _tokens().float()
    (output.float() * grad_output).sum().backward()

    # Each recv row is summed into exactly one token, so it receives that token's
    # incoming gradient.
    token_index = _expert_major_order(topk_idx) // TOP_K
    torch.testing.assert_close(expert_out.grad.float(), grad_output[token_index], **TOLERANCES)


def test_dropped_routes_are_skipped():
    config = _config()
    tokens = _tokens(4)
    topk_idx = torch.tensor([[1, -1], [0, 1], [2, 2], [-1, 0]], dtype=torch.int64, device=DEVICE)
    topk_weights = torch.full((4, 2), 0.5, dtype=torch.float32, device=DEVICE)

    recv_tokens, tokens_per_expert, recv_weights = te_ops.MoeDispatch(config)(
        tokens, topk_idx, topk_weights
    )

    # Routes to expert 0 (tokens 1, 3), then expert 1 (0, 1), then expert 2 (2, 2).
    torch.testing.assert_close(
        recv_tokens,
        torch.cat([tokens[1:2], tokens[3:4], tokens[0:1], tokens[1:2], tokens[2:3], tokens[2:3]]),
    )
    torch.testing.assert_close(
        tokens_per_expert, torch.tensor([2, 2, 2, 0], dtype=torch.int64, device=DEVICE)
    )
    torch.testing.assert_close(recv_weights, torch.full((6,), 0.5, device=DEVICE))
    # Combine sums the routes: tokens 1 and 2 keep two of them, tokens 0 and 3 one.
    output = te_ops.MoeCombine(config)(recv_tokens, topk_idx)
    torch.testing.assert_close(
        output,
        tokens * torch.tensor([[1.0], [2.0], [2.0], [1.0]], dtype=torch.bfloat16, device=DEVICE),
        **TOLERANCES,
    )


def test_rejects_experts_that_are_not_local():
    config = _config()
    topk_idx, topk_weights = _routing()
    topk_idx[0, 0] = NUM_EXPERTS
    with pytest.raises(ValueError, match="Pass an EpBuffer to use NCCL EP"):
        te_ops.MoeDispatch(config)(_tokens(), topk_idx, topk_weights)
    with pytest.raises(ValueError, match="Pass an EpBuffer to use NCCL EP"):
        te_ops.MoeCombine(config)(_tokens(NUM_TOKENS * TOP_K), topk_idx)


def test_rejects_invalid_routing_metadata():
    config = _config()
    topk_idx, topk_weights = _routing()
    dispatch = te_ops.MoeDispatch(config)
    with pytest.raises(TypeError, match="topk_idx must be int64"):
        dispatch(_tokens(), topk_idx.to(torch.int32), topk_weights)
    with pytest.raises(ValueError, match="topk_idx must have shape"):
        dispatch(_tokens(), topk_idx[:, :1], topk_weights[:, :1])
    with pytest.raises(TypeError, match="topk_weights must be float32"):
        dispatch(_tokens(), topk_idx, topk_weights.to(torch.bfloat16))
    with pytest.raises(ValueError, match="must not contain expert ids below -1"):
        dispatch(_tokens(), topk_idx - 2, topk_weights)
    with pytest.raises(ValueError, match="routing metadata describes"):
        te_ops.MoeCombine(config)(_tokens(NUM_TOKENS), topk_idx)


def test_rejects_unsupported_local_communication_options():
    config = _config()
    unsupported = (
        (replace(config, dispatch_fwd_quant_recipe=Float8CurrentScaling()), "quantized"),
        (replace(config, recv_capacity_per_rank=8), "receive capacity"),
        (replace(config, drop_on_overflow=True), "overflow policy"),
    )
    for invalid_config, reason in unsupported:
        for op in (te_ops.MoeDispatch, te_ops.MoeCombine):
            with pytest.raises(NotImplementedError, match=reason):
                op(invalid_config)


def _build_moe_sequence():
    """The five-op MoE sequence, with routing held locally instead of by NCCL EP."""
    config = _config()
    dispatch = te_ops.MoeDispatch(config)
    # Per-expert weights: the grouped-tensor path that single grouped parameters
    # need is gated to particular architectures and recipes.
    previous_single_param = os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM")
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    try:
        fc1 = te_ops.GroupedLinear(
            NUM_EXPERTS,
            HIDDEN_DIM,
            2 * INTERMEDIATE_DIM,
            bias=False,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        activation = te_ops.ScaledSwiGLU()
        fc2 = te_ops.GroupedLinear(
            NUM_EXPERTS,
            INTERMEDIATE_DIM,
            HIDDEN_DIM,
            bias=False,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
    finally:
        if previous_single_param is None:
            del os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"]
        else:
            os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = previous_single_param
    combine = te_ops.MoeCombine(config)
    dispatch.set_extra_output_channel(0, "tokens_per_expert", output_to_caller=False)
    dispatch.set_extra_output_channel(1, "routing_weights", output_to_caller=False)
    fc1.set_extra_input_channel(0, "tokens_per_expert")
    activation.set_extra_input_channel(0, "routing_weights")
    fc2.set_extra_input_channel(0, "tokens_per_expert")
    model = te_ops.Sequential(dispatch, fc1, activation, fc2, combine)
    return model, fc1, fc2


def _dense_reference(
    tokens,
    topk_idx,
    topk_weights,
    fc1_weight,
    fc2_weight,
):
    """Plain-PyTorch MoE with the same weights, routing, and top-k weighting."""
    token_index = torch.arange(tokens.shape[0], device=DEVICE).repeat_interleave(TOP_K)
    route_expert = topk_idx.reshape(-1)
    gate_up = torch.einsum("th,eoh->teo", tokens, fc1_weight)[token_index, route_expert]
    gate, up = gate_up.split(INTERMEDIATE_DIM, dim=-1)
    intermediate = F.silu(gate) * up * topk_weights.reshape(-1).unsqueeze(-1)
    expert_out = torch.bmm(
        intermediate.unsqueeze(1),
        fc2_weight[route_expert].transpose(1, 2),
    ).squeeze(1)
    output = torch.zeros(tokens.shape[0], HIDDEN_DIM, dtype=torch.float32, device=DEVICE)
    output.index_add_(0, token_index, expert_out)
    return output


def _grouped_weights(op):
    """Stack per-expert weights into the reference (E, out, in) layout."""
    return torch.stack([weight.detach().float() for weight in _expert_weights(op)]).requires_grad_(
        True
    )


def _expert_weights(op):
    weights = tuple(op.parameters())
    assert len(weights) == NUM_EXPERTS
    return weights


def test_moe_sequence_matches_dense_reference():
    """The five-op sequence on the PyTorch backend matches a dense MoE, forward and backward."""
    torch.manual_seed(2026)
    model, fc1, fc2 = _build_moe_sequence()
    tokens = _tokens().requires_grad_(True)
    topk_idx, topk_weights = _routing()
    topk_weights.requires_grad_(True)

    output = model(tokens, topk_idx, topk_weights, topk_idx)
    assert output.dtype is torch.bfloat16

    fc1_weight, fc2_weight = _grouped_weights(fc1), _grouped_weights(fc2)
    reference_tokens = tokens.detach().float().requires_grad_(True)
    reference_weights = topk_weights.detach().clone().requires_grad_(True)
    reference = _dense_reference(
        reference_tokens, topk_idx, reference_weights, fc1_weight, fc2_weight
    )
    torch.testing.assert_close(output, reference.to(torch.bfloat16), **TOLERANCES)

    grad_output = _tokens().float()
    output.backward(grad_output.to(torch.bfloat16))
    (reference * grad_output).sum().backward()

    torch.testing.assert_close(tokens.grad.float(), reference_tokens.grad, **TOLERANCES)
    torch.testing.assert_close(topk_weights.grad, reference_weights.grad, **TOLERANCES)
    for op, reference_grad in ((fc1, fc1_weight.grad), (fc2, fc2_weight.grad)):
        for weight, expected in zip(_expert_weights(op), reference_grad.unbind(0)):
            torch.testing.assert_close(
                weight.grad.float(),
                expected,
                **TOLERANCES,
            )


def test_moe_sequence_without_gradients():
    """Inference skips the backward state entirely."""
    model, fc1, fc2 = _build_moe_sequence()
    tokens = _tokens()
    topk_idx, topk_weights = _routing()
    with torch.no_grad():
        output = model(tokens, topk_idx, topk_weights, topk_idx)
    reference = _dense_reference(
        tokens.float(), topk_idx, topk_weights, _grouped_weights(fc1), _grouped_weights(fc2)
    )
    torch.testing.assert_close(output, reference.to(torch.bfloat16), **TOLERANCES)
