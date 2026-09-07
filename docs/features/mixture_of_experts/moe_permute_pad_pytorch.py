# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_PERMUTE_PAD_PYTORCH
from transformer_engine.pytorch import moe_permute_and_pad_with_probs, moe_unpermute

# tokens:      [num_tokens, hidden_size]
# probs:       [num_tokens, num_experts] routing probabilities
# routing_map: [num_tokens, num_experts] int32 mask
#
# Pad each expert's token block up to a multiple of align_size (here 128) so the
# grouped GEMM sees aligned blocks. Permutation and padding happen in one kernel.
tokens_per_expert = routing_map.sum(dim=0)  # [num_experts]
padded, permuted_probs, row_id_map, pad_offsets, padded_tokens_per_expert = (
    moe_permute_and_pad_with_probs(
        tokens, probs, routing_map, tokens_per_expert, align_size=128,
    )
)

# padded:                   [sum(padded_tokens_per_expert), hidden_size]
# pad_offsets:              per-expert cumulative padding (None if already aligned)
# padded_tokens_per_expert: aligned per-expert counts -> m_splits for GroupedLinear

# ... run the grouped MLP on `padded`, producing expert_out ...

# Apply the permuted routing weights to the completed expert outputs. Since the
# weights are applied here, do not pass them to moe_unpermute as well.
expert_out = expert_out * permuted_probs[:, None]

# Pass pad_offsets so token combine removes the padding it added, and
# restore_shape so the result has the original [num_tokens, hidden_size] shape.
output = moe_unpermute(
    expert_out,
    row_id_map,
    restore_shape=tokens.shape,
    pad_offsets=pad_offsets,
)
# END_MOE_PERMUTE_PAD_PYTORCH
