# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepEP-backed TokenDispatcher using fused all-to-all and Triton index conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import transformer_engine.pytorch

from fused_a2a import fused_combine, fused_dispatch
from fused_indices_converter import HAVE_TRITON, fused_indices_to_multihot
from te_mixtral import DispatchOutput


@dataclass
class _FusedHandle:
    """Opaque state for FusedTokenRouter between dispatch and combine."""

    deepep_handle: Any
    row_id_map: torch.Tensor
    probs_multihot: torch.Tensor
    recv_shape: torch.Size


class FusedTokenRouter:
    """TokenDispatcher using DeepEP fused communication and Triton index conversion.

    Dispatch flow:
        1. ``fused_dispatch`` -- DeepEP all-to-all sends tokens to expert-owning ranks.
        2. ``fused_indices_to_multihot`` -- Triton kernel converts sparse ``[N, top_k]``
           indices to dense ``[N, num_local_experts]`` mask with differentiable probs.
        3. ``moe_permute(map_type="mask")`` -- TE sorts received tokens by local expert.

    Combine flow:
        1. ``moe_unpermute(map_type="mask")`` -- TE unsorts and applies routing weights.
        2. ``fused_combine`` -- DeepEP reverse all-to-all sends results back.

    Args:
        num_experts: Total number of experts (global, across all EP ranks).
        num_local_experts: Number of experts hosted on this rank.
        hidden_size: Hidden dimension size.
        ep_size: Expert parallel world size.
    """

    def __init__(self, num_experts: int, num_local_experts: int, hidden_size: int, ep_size: int):
        """Initialize the FusedTokenRouter."""
        if fused_dispatch is None or fused_combine is None:
            raise ImportError("deep_ep is required for FusedTokenRouter. Install via: bash install_hybridep.sh")
        if not HAVE_TRITON:
            raise ImportError("Triton is required for FusedTokenRouter. Install via: pip install triton")
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.ep_size = ep_size
        self._ep_group: dist.ProcessGroup | None = None

    def set_ep_group(self, ep_group: dist.ProcessGroup) -> None:
        """Set the expert-parallel process group for communication."""
        self._ep_group = ep_group

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> DispatchOutput:
        """Dispatch tokens to their assigned experts via DeepEP fused all-to-all."""
        assert self._ep_group is not None, "EP group must be set via set_ep_group() before dispatch"

        recv_x, recv_indices, recv_probs, tokens_per_expert, deepep_handle = fused_dispatch(
            hidden_states,
            selected_experts,
            routing_weights.float(),
            self.num_experts,
            self._ep_group,
        )

        multihot_mask, probs_multihot = fused_indices_to_multihot(recv_indices, recv_probs, self.num_local_experts)

        num_out_tokens = int(tokens_per_expert.sum().item())
        permuted_x, row_id_map = transformer_engine.pytorch.moe_permute(
            recv_x, multihot_mask.to(torch.int32), num_out_tokens=num_out_tokens, map_type="mask"
        )

        handle = _FusedHandle(
            deepep_handle=deepep_handle,
            row_id_map=row_id_map,
            probs_multihot=probs_multihot,
            recv_shape=recv_x.shape,
        )

        return DispatchOutput(
            expert_input=permuted_x,
            tokens_per_expert=tokens_per_expert.tolist(),
            handle=handle,
        )

    def combine(self, expert_output: torch.Tensor, handle: _FusedHandle) -> torch.Tensor:
        """Combine expert outputs back to the original token order."""
        unpermuted = transformer_engine.pytorch.moe_unpermute(
            expert_output,
            handle.row_id_map,
            merging_probs=handle.probs_multihot,
            restore_shape=handle.recv_shape,
            map_type="mask",
        )

        combined, _ = fused_combine(unpermuted, self._ep_group, handle.deepep_handle)
        return combined
