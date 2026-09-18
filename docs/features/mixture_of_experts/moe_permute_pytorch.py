# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_PERMUTE_PYTORCH
import torch
from transformer_engine.pytorch import moe_permute

# tokens:      [num_tokens, hidden_size]
# routing_map: [num_tokens, num_experts] mask, 1 if token routed to expert
#
# num_out_tokens is the number of rows in the permuted buffer. Reading it from
# the routing map (``int(routing_map.sum())``) triggers a device-to-host sync;
# when the value is known statically (e.g. ``num_tokens * top_k`` for dropless
# routing), prefer passing that constant directly.
permuted, row_id_map = moe_permute(
    tokens,
    routing_map,
    num_out_tokens=int(routing_map.sum()),
)

# permuted:    [num_out_tokens, hidden_size], expert-contiguous
# row_id_map:  opaque tensor used by ``moe_unpermute`` to reverse the
#              permutation after the experts have run.
# END_MOE_PERMUTE_PYTORCH
