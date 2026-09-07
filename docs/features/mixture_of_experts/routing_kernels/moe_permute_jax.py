# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_PERMUTE_JAX
import jax.numpy as jnp
from transformer_engine.jax import permutation as te_permutation

# tokens:      [num_tokens, hidden_size]
# routing_map: [num_tokens, num_experts] mask, 1 if token routed to expert
# num_out_tokens must be known at trace time when used under ``jit``.
permuted, _, row_id_map, _, group_sizes = te_permutation.token_dispatch(
    tokens,
    routing_map,
    num_out_tokens=int(jnp.sum(routing_map)),
)

# permuted:    [num_out_tokens, hidden_size], expert-contiguous
# group_sizes: [num_experts], per-expert token counts; can be passed directly
#              to ``grouped_dense`` as ``group_sizes``.
# row_id_map:  opaque tensor used by ``token_combine`` to reverse the permutation.
#
# The two ignored outputs are ``permuted_probs`` and ``pad_offsets``:
#   - ``permuted_probs`` (returned only when ``probs=`` is supplied) holds the
#     routing probabilities permuted into expert-contiguous order. It is used
#     for input-side scaling of expert inputs before the grouped GEMM, as an
#     alternative to passing ``merging_probs`` to ``token_combine``.
#   - ``pad_offsets`` is only used together with ``align_size`` for fused
#     padding to expert-aligned blocks.
# END_MOE_PERMUTE_JAX
