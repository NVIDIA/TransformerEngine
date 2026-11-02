# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_PERMUTE_PAD_PYTORCH
from transformer_engine.pytorch import moe_permute_and_pad_with_probs, moe_unpermute

# tokens:      [num_tokens, hidden_size]
# probs:       [num_tokens, num_experts] routing probabilities
# routing_map: [num_tokens, num_experts] int32 mask
#
# Pad each expert's token block up to a multiple of align_size (here 256) so the
# grouped GEMM sees aligned blocks. Permutation and padding happen in one kernel.
tokens_per_expert = routing_map.sum(dim=0)  # [num_experts]
padded, permuted_probs, row_id_map, pad_offsets, padded_tokens_per_expert = (
    moe_permute_and_pad_with_probs(
        tokens,
        probs,
        routing_map,
        tokens_per_expert,
        align_size=256,
    )
)

# padded:                   [sum(padded_tokens_per_expert), hidden_size]
# pad_offsets:              per-expert cumulative padding (None if already aligned)
# padded_tokens_per_expert: aligned per-expert counts -> m_splits for GroupedLinear

# ... run an unweighted expert block on `padded`, producing expert_out ...

# Apply weights once. Skip this multiplication for the scaled grouped MLP,
# which already weights its outputs through ScaledSwiGLU and scale_bias=True.
expert_out = expert_out * permuted_probs[:, None]

# Combine already weighted outputs without merging_probs. Pass pad_offsets to
# remove padding and restore_shape for the original [num_tokens, hidden_size].
output = moe_unpermute(
    expert_out,
    row_id_map,
    restore_shape=tokens.shape,
    pad_offsets=pad_offsets,
)
# END_MOE_PERMUTE_PAD_PYTORCH
