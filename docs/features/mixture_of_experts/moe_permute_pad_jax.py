# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_PERMUTE_PAD_JAX
from transformer_engine.jax import permutation as te_permutation

# tokens:      [num_tokens, hidden_size]
# probs:       [num_tokens, num_experts] routing probabilities
# routing_map: [num_tokens, num_experts] int32 mask
#
# Passing align_size enables the same fused padding. token_dispatch allocates a
# fixed worst-case buffer (so it stays jit-compatible) and reports the aligned
# per-expert counts together with the padding offsets.
padded, permuted_probs, row_id_map, pad_offsets, tokens_per_expert = te_permutation.token_dispatch(
    tokens,
    routing_map,
    num_out_tokens=num_tokens * top_k,
    probs=probs,
    align_size=128,
)

# tokens_per_expert: aligned per-expert counts -> group_sizes for grouped_dense

# ... run the grouped GEMM on `padded`, producing expert_out ...

# Pass pad_offsets so token combine removes the padding it added.
output = te_permutation.token_combine(
    expert_out, row_id_map, merging_probs=probs, pad_offsets=pad_offsets,
)
# END_MOE_PERMUTE_PAD_JAX
