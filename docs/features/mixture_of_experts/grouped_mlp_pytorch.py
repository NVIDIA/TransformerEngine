# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_GROUPED_MLP_PYTORCH
import transformer_engine.pytorch as te

# FC1 produces gate and value features interleaved in blocks of 32.
expert_mlp = te.ops.Sequential(
    te.ops.GroupedLinear(num_experts, hidden_size, 2 * ffn_hidden_size, split_alignment=256),
    te.ops.ScaledSwiGLU(glu_interleave_size=32),
    te.ops.GroupedLinear(
        num_experts, ffn_hidden_size, hidden_size, scale_bias=True, split_alignment=256
    ),
)

# Inputs from padded dispatch with align_size=256.
expert_out = expert_mlp(
    padded,
    padded_tokens_per_expert,
    permuted_probs,
    padded_tokens_per_expert,
    permuted_probs,
)
# END_GROUPED_MLP_PYTORCH
