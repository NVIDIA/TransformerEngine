# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_LAYER_PYTORCH
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.router import fused_topk_with_score_function

num_tokens, hidden_size, num_experts, top_k = 16, 64, 4, 2
hidden_states = torch.randn(
    num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True,
)
gate = torch.nn.Linear(
    hidden_size, num_experts, bias=False, device="cuda", dtype=torch.bfloat16,
)
experts = te.GroupedLinear(
    num_experts,
    hidden_size,
    hidden_size,
    bias=False,
    params_dtype=torch.bfloat16,
    device="cuda",
)

# 1. Router: score the experts and pick the top-k for each token.
logits = gate(hidden_states)
probs, routing_map = fused_topk_with_score_function(
    logits, topk=top_k, use_pre_softmax=False, num_groups=None,
    group_topk=None, scaling_factor=None, score_function="softmax", expert_bias=None,
)

# 2. Dispatch: gather tokens into expert-contiguous order.
routing_map = routing_map.to(torch.int32)
permuted, row_id_map = te.moe_permute(
    hidden_states, routing_map, num_out_tokens=num_tokens * top_k,
)

# 3. Experts: one grouped call over all expert token blocks.
m_splits = routing_map.sum(dim=0).tolist()  # tokens routed to each expert
expert_out = experts(permuted, m_splits)

# 4. Combine: scatter the outputs back and merge the top-k contributions.
output = te.moe_unpermute(
    expert_out, row_id_map, merging_probs=probs, restore_shape=(num_tokens, hidden_size),
)
output.square().mean().backward()
# END_MOE_LAYER_PYTORCH
