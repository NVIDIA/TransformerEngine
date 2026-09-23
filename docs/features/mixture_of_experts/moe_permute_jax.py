# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_PERMUTE_JAX
import jax.numpy as jnp
from transformer_engine.jax import permutation as te_permutation

# tokens:      [num_tokens, hidden_size]
# routing_map: [num_tokens, num_experts] mask, 1 if token routed to expert
# top_k is the statically configured number of experts selected per token.
# tokens.shape[0] is static while tracing, so this stays valid under ``jit``.
permuted, _, row_id_map, _, group_sizes = te_permutation.token_dispatch(
    tokens,
    routing_map.astype(jnp.int32),
    num_out_tokens=tokens.shape[0] * top_k,
)

# permuted:    [num_out_tokens, hidden_size], expert-contiguous
# group_sizes: [num_experts], per-expert token counts; can be passed directly
#              to ``grouped_dense`` as ``group_sizes``.
# row_id_map:  opaque tensor used by ``token_combine`` to reverse the permutation.
#
# The two ignored outputs are ``permuted_probs`` and ``pad_offsets``:
#   - ``permuted_probs`` (returned only when ``probs=`` is supplied) holds the
#     routing probabilities in expert-contiguous order. Multiply the completed
#     expert outputs by these weights before ``token_combine``; dispatch does
#     not apply them itself. Do not also pass ``merging_probs`` in that case.
#   - ``pad_offsets`` is only used together with ``align_size`` for fused
#     padding to expert-aligned blocks.
# END_MOE_PERMUTE_JAX
