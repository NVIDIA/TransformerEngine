# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_LAYER_JAX
import jax
import jax.numpy as jnp
from transformer_engine.jax import permutation as te_permutation
from transformer_engine.jax import dense as te_dense
from transformer_engine.jax.router import fused_topk_with_score_function

num_tokens, hidden_size, num_experts, top_k = 16, 64, 4, 2
keys = jax.random.split(jax.random.key(0), 3)
hidden_states = jax.random.normal(keys[0], (num_tokens, hidden_size), dtype=jnp.bfloat16)
gate_kernel = jax.random.normal(keys[1], (hidden_size, num_experts), dtype=jnp.bfloat16)
kernel = jax.random.normal(
    keys[2], (num_experts, hidden_size, hidden_size), dtype=jnp.bfloat16,
)

@jax.jit
def moe_layer(tokens, gate_weight, expert_weights):
    # 1. Router: score the experts and pick the top-k for each token.
    logits = tokens @ gate_weight
    probs, routing_map = fused_topk_with_score_function(
        logits, topk=top_k, score_function="softmax",
    )

    # 2. Dispatch: gather tokens into expert-contiguous order.
    permuted, _, row_id_map, _, group_sizes = te_permutation.token_dispatch(
        tokens, routing_map.astype(jnp.int32), num_out_tokens=num_tokens * top_k,
    )

    # 3. Experts: one grouped call over all expert token blocks.
    expert_out = te_dense.grouped_dense(permuted, expert_weights, group_sizes=group_sizes)

    # 4. Combine: restore token order and merge the top-k contributions.
    return te_permutation.token_combine(expert_out, row_id_map, merging_probs=probs)

output = moe_layer(hidden_states, gate_kernel, kernel)
output.block_until_ready()
# END_MOE_LAYER_JAX
