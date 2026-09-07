# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_EXPERT_PARALLEL_JAX
from transformer_engine.jax.moe import moe  # experimental

# x:          [num_tokens, hidden_size]
# gate_kernel:[hidden_size, num_experts]        router projection
# wi_0, wi_1: [num_experts, hidden_size, ffn]   expert gate / value projections (SwiGLU)
# wo:         [num_experts, ffn, hidden_size]    expert output projection
#
# moe() runs the whole layer - router, dispatch, grouped expert GEMMs and
# combine - as a single differentiable call. When ep_axis names a mesh axis,
# the dispatch and combine steps become all-to-all collectives over that axis,
# so experts can be sharded across devices (expert parallelism).
output, aux_loss = moe(
    x, gate_kernel, wi_0, wi_1, wo,
    num_experts=8,
    num_experts_per_tok=2,         # top-k
    activation_type="silu",
    score_function="softmax",
    aux_loss_coeff=1e-2,           # load-balancing loss; 0 disables it
    ep_axis="ep",                  # mesh axis for expert parallelism (None = no EP)
)
# output:   [num_tokens, hidden_size]
# aux_loss: scalar load-balancing loss (None when aux_loss_coeff == 0)
# END_MOE_EXPERT_PARALLEL_JAX
