# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_EXPERT_PARALLEL_JAX
from transformer_engine.jax.moe import moe  # experimental

# Run inside a jax.sharding.Mesh with an "ep" axis; x is sharded over it.
# x:           [batch, seq, hidden_size]
# gate_kernel: [hidden_size, num_experts]          router projection
# wi:          [num_experts, hidden_size, 2 * ffn]  gated FC1 (gate and value)
# wo:          [num_experts, ffn, hidden_size]      FC2
output, aux_loss, total_recv_tokens = moe(
    x, gate_kernel, wi, wo,
    num_experts=8,
    num_experts_per_tok=2,        # top-k
    activation_type="silu",
    score_function="softmax",
    aux_loss_coeff=1e-2,          # load-balancing loss; 0 disables it
    ep_axis="ep",                 # mesh axis the experts are sharded over
)
# output:   [batch, seq, hidden_size]
# aux_loss: scalar load-balancing loss (None when aux_loss_coeff == 0)
# END_MOE_EXPERT_PARALLEL_JAX
