# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the pure PyTorch MoE EP reference."""

from types import SimpleNamespace

import pytest
import torch

from transformer_engine.pytorch import ops as te_ops
from transformer_engine.pytorch.ep_reference import MoeEpReference
from transformer_engine.pytorch.ops.fused.moe_ep import FusedMoeEp


@pytest.mark.parametrize("compute_dtype", (torch.float32, torch.bfloat16))
def test_moe_ep_reference_compute_dtype(compute_dtype):
    """The configured dtype controls MLP, combine, and non-router gradients."""
    generator = torch.Generator().manual_seed(1234)
    activation = torch.randn(4, 8, generator=generator, dtype=torch.bfloat16)
    fc1_weight = torch.randn(2, 8, 8, generator=generator, dtype=torch.bfloat16)
    fc2_weight = torch.randn(2, 4, 8, generator=generator, dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    topk_weights = torch.ones(4, 1, dtype=torch.float32)

    reference = MoeEpReference(
        num_experts=2,
        hidden_size=8,
        intermediate_size=4,
        top_k=1,
        generate_c=True,
        compute_dtype=compute_dtype,
    )
    output, fc1_c, route_metadata = reference(
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
    )
    grads = reference.backward(
        torch.ones_like(output),
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        fc1_c,
        route_metadata,
    )

    assert output.dtype is torch.bfloat16
    assert grads[0].dtype is compute_dtype
    assert grads[1].dtype is compute_dtype
    assert grads[2].dtype is compute_dtype
    assert grads[3].dtype is torch.float32


def test_moe_ep_reference_default_compute_dtype_is_fp32():
    """Preserve the reference's pre-existing FP32 compute default."""
    reference = MoeEpReference(
        num_experts=1,
        hidden_size=8,
        intermediate_size=4,
        top_k=1,
    )
    assert reference.compute_dtype is torch.float32


def test_single_rank_bf16_moe_fusion_forward_backward():
    """Exercise the fuser contract without requiring the NCCL EP backend."""
    buffer = SimpleNamespace(
        num_local_experts=2,
        hidden_dim=8,
        top_k=1,
        max_tokens_per_rank=4,
        payload_dtype=torch.bfloat16,
        eager=True,
    )
    dispatch = te_ops.Dispatch(buffer)
    fc1 = te_ops.GroupedLinear(2, 8, 8, bias=False, device="cpu", dtype=torch.bfloat16)
    activation = te_ops.ScaledSwiGLU()
    fc2 = te_ops.GroupedLinear(2, 4, 8, bias=False, device="cpu", dtype=torch.bfloat16)
    combine = te_ops.Combine(buffer, num_local_tokens=4)
    dispatch.set_extra_output_channel(0, "tokens_per_expert")
    dispatch.set_extra_output_channel(1, "routing_weights")
    fc1.set_extra_input_channel(0, "tokens_per_expert")
    activation.set_extra_input_channel(0, "routing_weights")
    fc2.set_extra_input_channel(0, "tokens_per_expert")
    model = te_ops.Sequential(dispatch, fc1, activation, fc2, combine)

    x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    topk_idx = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    topk_weights = torch.ones(4, 1, dtype=torch.float32, requires_grad=True)
    output, counts, recv_weights = model(x, topk_idx, topk_weights)
    torch.autograd.backward(
        (output, recv_weights),
        (torch.ones_like(output), torch.ones_like(recv_weights)),
    )

    assert isinstance(model._module_groups[0]._forward_ops[0][0], FusedMoeEp)
    assert counts.dtype is torch.int64
    assert output.dtype is torch.bfloat16
    assert x.grad.dtype is torch.bfloat16
    assert topk_weights.grad.dtype is torch.float32
    for op in (fc1, fc2):
        for expert in range(op.num_groups):
            assert getattr(op, f"weight{expert}").grad.dtype is torch.bfloat16
