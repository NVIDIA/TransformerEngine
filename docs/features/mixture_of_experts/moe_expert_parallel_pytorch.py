# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_EXPERT_PARALLEL_PYTORCH
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.pytorch.ep import EpBuffer, EpConfig, ep_bootstrap

# ep_group:  process group the experts are sharded over
# tokens:    [num_tokens, hidden_size] bf16 tokens local to this rank
# topk_idx:  [num_tokens, top_k] global expert index per selected expert
# topk_w:    [num_tokens, top_k] fp32 routing weights from the router
ep_size = dist.get_world_size(ep_group)
num_local_experts = num_experts // ep_size
recv_capacity = ep_size * max_tokens_per_rank * top_k  # alignment=0
config = EpConfig(
    top_k=top_k,
    hidden_dim=hidden_size,
    num_local_experts=num_local_experts,
    max_tokens_per_rank=max_tokens_per_rank,
    recv_capacity_per_rank=recv_capacity,
    ep_group=ep_group,
    alignment=0,
)

# Once per process: sets up NCCL EP on ep_group's communicator.
ep_bootstrap(
    ep_group,
    num_experts=num_experts,
    max_tokens_per_rank=config.max_tokens_per_rank,
    hidden_dim=hidden_size,
    num_topk=top_k,
    recv_capacity_per_rank=recv_capacity,
)
# One buffer per in-flight layer call (e.g. per pipeline microbatch).
buffer = EpBuffer(
    top_k=config.top_k,
    max_tokens_per_rank=config.max_tokens_per_rank,
    recv_capacity_per_rank=config.recv_capacity_per_rank,
    hidden_dim=config.hidden_dim,
    num_local_experts=config.num_local_experts,
    alignment=config.alignment,
    device=tokens.device,
)

dispatch = te.ops.MoeDispatch(config, buffer)
fc1 = te.ops.GroupedLinear(
    num_local_experts,
    hidden_size,
    2 * ffn_hidden_size,
    bias=False,
    device=tokens.device,
    dtype=torch.bfloat16,
)
activation = te.ops.ScaledSwiGLU()
fc2 = te.ops.GroupedLinear(
    num_local_experts,
    ffn_hidden_size,
    hidden_size,
    bias=False,
    device=tokens.device,
    dtype=torch.bfloat16,
)
combine = te.ops.MoeCombine(config, buffer)

# Bind dispatch metadata before constructing or calling the sequence.
dispatch.set_extra_output_channel(0, "tokens_per_expert", output_to_caller=False)
dispatch.set_extra_output_channel(1, "routing_weights", output_to_caller=False)
fc1.set_extra_input_channel(0, "tokens_per_expert")
activation.set_extra_input_channel(0, "routing_weights")
fc2.set_extra_input_channel(0, "tokens_per_expert")
moe = te.ops.Sequential(dispatch, fc1, activation, fc2, combine)

# Dispatch takes indices and weights; combine takes the same indices.
output = moe(tokens, topk_idx, topk_w, topk_idx)  # [num_tokens, hidden_size]
# END_MOE_EXPERT_PARALLEL_PYTORCH
