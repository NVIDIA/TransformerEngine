# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_LAYER_JAX
import jax.numpy as jnp
from transformer_engine.jax import permutation as te_permutation
from transformer_engine.jax import dense as te_dense
from transformer_engine.jax.router import fused_topk_with_score_function

# hidden_states: [num_tokens, hidden_size]
# gate_kernel:   [hidden_size, num_experts], the router projection
# kernel, bias:  stacked per-expert weights/biases for the grouped GEMM that
#                stands in for the expert MLP here (see "Grouped GEMM").
top_k = 2
num_tokens = hidden_states.shape[0]

# 1. Router: score the experts and pick the top-k for each token.
logits = hidden_states @ gate_kernel
probs, routing_map = fused_topk_with_score_function(logits, topk=top_k, score_function="softmax")

# 2. Dispatch: gather tokens into expert-contiguous order.
permuted, _, row_id_map, _, group_sizes = te_permutation.token_dispatch(
    hidden_states, routing_map.astype(jnp.int32), num_out_tokens=num_tokens * top_k,
)

# 3. Experts: one grouped GEMM over all expert token blocks.
expert_out = te_dense.grouped_dense(permuted, kernel, group_sizes=group_sizes, bias=bias)

# 4. Combine: scatter the outputs back and merge the top-k contributions.
output = te_permutation.token_combine(expert_out, row_id_map, merging_probs=probs)
# output: [num_tokens, hidden_size]
# END_MOE_LAYER_JAX
