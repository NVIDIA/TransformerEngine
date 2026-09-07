# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_EXPERT_PARALLEL_PYTORCH
import torch.distributed as dist
from transformer_engine.pytorch.ep import EpBuffer, ep_bootstrap, ep_dispatch, ep_combine

# ep_group:  process group the experts are sharded over
# tokens:    [num_tokens, hidden_size] bf16 tokens local to this rank
# topk_idx:  [num_tokens, top_k] global expert index per selected expert
# topk_w:    [num_tokens, top_k] fp32 routing weights from the router
# experts:   the MLP over the num_local_experts owned by this rank
ep_size = dist.get_world_size(ep_group)
num_local_experts = num_experts // ep_size
recv_capacity = ep_size * num_tokens * top_k  # dropless worst case per rank

# Once per process: sets up NCCL EP on ep_group's communicator.
ep_bootstrap(
    ep_group,
    num_experts=num_experts,
    max_tokens_per_rank=num_tokens,
    hidden_dim=hidden_size,
    num_topk=top_k,
    recv_capacity_per_rank=recv_capacity,
)
# One buffer per in-flight layer call (e.g. per pipeline microbatch).
buffer = EpBuffer(
    top_k=top_k,
    max_tokens_per_rank=num_tokens,
    recv_capacity_per_rank=recv_capacity,
    hidden_dim=hidden_size,
    num_local_experts=num_local_experts,
)

# Dispatch: all-to-all sends each token to the rank owning its expert.
# recv_tokens is [recv_capacity, hidden_size], one fixed-size slot range per
# local expert; tokens_per_expert holds the number of valid rows in each.
recv_tokens, recv_w, tokens_per_expert = ep_dispatch(buffer, tokens, topk_idx, topk_w)

# Local experts run on the receive buffer; apply the routing weights before combine.
expert_out = experts(recv_tokens, tokens_per_expert)
expert_out = expert_out * recv_w.unsqueeze(-1).to(expert_out.dtype)

# Combine: all-to-all returns the weighted outputs to the source rank and sums them.
output = ep_combine(buffer, expert_out)  # [num_tokens, hidden_size]
# END_MOE_EXPERT_PARALLEL_PYTORCH
