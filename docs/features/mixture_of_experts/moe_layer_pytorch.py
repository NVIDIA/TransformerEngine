# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_LAYER_PYTORCH
import torch
from transformer_engine.pytorch import moe_permute, moe_unpermute
from transformer_engine.pytorch.router import fused_topk_with_score_function

# hidden_states: [num_tokens, hidden_size]
# gate:          torch.nn.Linear(hidden_size, num_experts), the router projection
# experts:       the per-expert MLP, built from te.GroupedLinear (see "Grouped GEMM");
#                a full expert MLP stacks two grouped GEMMs around an activation.
top_k = 2
num_tokens, hidden_size = hidden_states.shape

# 1. Router: score the experts and pick the top-k for each token.
logits = gate(hidden_states)
probs, routing_map = fused_topk_with_score_function(
    logits, topk=top_k, use_pre_softmax=False, num_groups=None,
    group_topk=None, scaling_factor=None, score_function="softmax", expert_bias=None,
)

# 2. Dispatch: gather tokens into expert-contiguous order.
routing_map = routing_map.to(torch.int32)
permuted, row_id_map = moe_permute(
    hidden_states, routing_map, num_out_tokens=num_tokens * top_k,
)

# 3. Experts: one grouped MLP call over all expert token blocks.
m_splits = routing_map.sum(dim=0).tolist()  # tokens routed to each expert
expert_out = experts(permuted, m_splits)

# 4. Combine: scatter the outputs back and merge the top-k contributions.
# restore_shape is the original token shape; it is needed whenever the permuted
# buffer has more rows than the input (top-k routing: num_out_tokens > num_tokens).
output = moe_unpermute(
    expert_out, row_id_map, merging_probs=probs, restore_shape=(num_tokens, hidden_size),
)
# output: [num_tokens, hidden_size]
# END_MOE_LAYER_PYTORCH
