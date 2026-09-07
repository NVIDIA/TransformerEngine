# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_UNPERMUTE_JAX
from transformer_engine.jax import permutation as te_permutation

# expert_out:   [num_out_tokens, hidden_size], expert-contiguous,
#               produced by grouped_dense (or a grouped MLP).
# row_id_map:   returned by token_dispatch.
# router_probs: [num_tokens, num_experts]; the original (un-permuted) routing
#               probabilities. Provide for top-k routing to weight per-expert
#               contributions in the same fused pass; pass None for top-1.
tokens_out = te_permutation.token_combine(
    expert_out,
    row_id_map,
    merging_probs=router_probs,
)

# tokens_out: [num_tokens, hidden_size], in the original token order
# END_MOE_UNPERMUTE_JAX
