# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_GROUPED_LINEAR_JAX
import jax.numpy as jnp
from transformer_engine.jax import dense as te_dense

# x:           [sum(group_sizes), hidden_size], expert-contiguous tokens
# kernel:      [num_experts, hidden_size, ffn_hidden_size], stacked per-expert weights
# bias:        [num_experts, ffn_hidden_size], stacked per-expert biases
# group_sizes: [num_experts] int array; group_sizes[i] is the number of routed
#              tokens for expert i
split_indices = jnp.cumsum(group_sizes)[:-1]
x_by_expert = jnp.split(x, split_indices, axis=0)

# Baseline: one matmul per expert.
loop_out = jnp.concatenate(
    [x_i @ kernel_i + bias_i for x_i, kernel_i, bias_i in zip(x_by_expert, kernel, bias)],
    axis=0,
)

# Transformer Engine: one grouped dense call.
grouped_out = te_dense.grouped_dense(
    x,
    kernel,
    group_sizes=group_sizes,
    bias=bias,
)
# END_GROUPED_LINEAR_JAX
