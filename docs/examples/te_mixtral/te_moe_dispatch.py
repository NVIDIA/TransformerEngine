# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Token dispatch / combine for the MXFP8 TE-native MoE block.

Wraps the permute/pad/all-to-all/sort-by-expert plumbing that moves tokens
from data-parallel ranks to their owning expert ranks, and reverses the
operation on the way back. The transport is NCCL ``all_to_all_single``;
the all-to-all is just the mechanism — the public API is ``dispatch()``
/ ``combine()``.

Per-expert MoE permute pads to 128 (grouped MXFP8 GEMM M-tile). Both
hidden states *and* per-token routing probabilities are transmitted so
the destination-side ``ScaledSwiGLU(glu_interleave_size=32)`` has its
scales locally — that's what trips the fused
``ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8`` kernel. ``combine()`` does not
re-apply routing weights (already applied inside ScaledSwiGLU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te

# Required so the ``@torch.compile`` helpers below can capture data-dependent
# tensor shapes (e.g. ``repeat_interleave`` with tensor counts) without
# bailing out to Python. Must be set at module level — torch.compile traces
# lazily on the first call, so a scoped setting wouldn't be active.
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True


# Per-expert token-count alignment required by the fused MXFP8 grouped-MLP
# CuTe-DSL kernel (ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8). The kernel
# rejects an input whose per-group token count is not a multiple of 256
# ("Invalid a.shape[0] ... expected to be divisible by 256"). 128 was the
# old value that worked when the fused kernel wasn't firing on B300; with
# the SM10.x gate now passing on B300 we must pad to 256.
_MXFP8_GROUP_ALIGN = 256


@dataclass
class DispatchOutput:
    """Tokens, per-token probs, and split sizes routed to the local experts."""

    expert_input: torch.Tensor
    expert_probs: torch.Tensor
    tokens_per_expert: list[int]
    handle: Any


@dataclass
class _Handle:
    row_id_map: torch.Tensor
    restore_shape: torch.Size
    pad_offsets: torch.Tensor | None
    unsort_indices: torch.Tensor | None = None
    input_split_sizes: list[int] | None = None
    output_split_sizes: list[int] | None = None


class _DifferentiableAllToAll(torch.autograd.Function):
    """``dist.all_to_all_single`` wrapped in autograd (works for 1-D and 2-D)."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.group = group
        total_out = sum(output_split_sizes)
        if input.dim() == 1:
            output = torch.empty(total_out, device=input.device, dtype=input.dtype)
        else:
            output = torch.empty(
                total_out, *input.shape[1:], device=input.device, dtype=input.dtype
            )
        dist.all_to_all_single(
            output, input.contiguous(), output_split_sizes, input_split_sizes, group=group
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        total_in = sum(ctx.input_split_sizes)
        if grad_output.dim() == 1:
            grad_input = torch.empty(total_in, device=grad_output.device, dtype=grad_output.dtype)
        else:
            grad_input = torch.empty(
                total_in, *grad_output.shape[1:], device=grad_output.device, dtype=grad_output.dtype
            )
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            group=ctx.group,
        )
        return grad_input, None, None, None


@torch.compile(fullgraph=True)
def _build_expert_sort_indices(recv_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sort/unsort indices that regroup ``[src*expert]`` tokens by expert.

    After ``all_to_all`` tokens arrive grouped by source rank
    (``[src0_exp0..src0_expL, src1_exp0..src1_expL, ...]``). ``GroupedLinear``
    needs them grouped by expert (``[all_exp0, all_exp1, ...]``).

    Uses only vectorized tensor ops (no ``.item()`` calls or Python-level
    loops) so this is ``torch.compile(fullgraph=True)``-safe.
    """
    ep_size, num_local_experts = recv_counts.shape
    device = recv_counts.device
    num_blocks = ep_size * num_local_experts

    counts_src = recv_counts.reshape(-1).long()
    offsets_src = torch.zeros(num_blocks, dtype=torch.long, device=device)
    offsets_src[1:] = counts_src[:-1].cumsum(0)

    counts_exp = recv_counts.t().contiguous().reshape(-1).long()
    offsets_exp = torch.zeros(num_blocks, dtype=torch.long, device=device)
    offsets_exp[1:] = counts_exp[:-1].cumsum(0)

    total = counts_src.sum()

    s_idx = torch.arange(ep_size, device=device).unsqueeze(1).expand(ep_size, num_local_experts)
    e_idx = (
        torch.arange(num_local_experts, device=device)
        .unsqueeze(0)
        .expand(ep_size, num_local_experts)
    )
    src_to_exp = (e_idx * ep_size + s_idx).reshape(-1)
    shifts = offsets_exp[src_to_exp] - offsets_src
    token_shifts = shifts.repeat_interleave(counts_src)
    src_positions = torch.arange(total, device=device)
    dst_positions = src_positions + token_shifts
    sort_indices = torch.empty(total, dtype=torch.long, device=device)
    sort_indices[dst_positions] = src_positions
    unsort_indices = torch.empty_like(sort_indices)
    unsort_indices[sort_indices] = torch.arange(total, device=device)
    return sort_indices, unsort_indices


class AllToAllTokenDispatcher:
    """NCCL all-to-all dispatcher for the TE-native MXFP8 MoE block.

    Args:
        num_experts: Total global experts.
        num_local_experts: Experts owned by this rank.
        hidden_size: Hidden feature dim.
        ep_size: Expert-parallel world size (1 = single-process).
        pad_align: Per-expert split alignment. Must be a multiple of 128 for
            the grouped MXFP8 GEMM.
    """

    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        ep_size: int,
        pad_align: int = _MXFP8_GROUP_ALIGN,
    ) -> None:
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.ep_size = ep_size
        self.pad_align = pad_align
        self._ep_group: dist.ProcessGroup | None = None

    def set_ep_group(self, ep_group: dist.ProcessGroup) -> None:
        self._ep_group = ep_group

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> DispatchOutput:
        """Permute -> pad -> (all-to-all) -> sort-by-expert.

        ``tokens_per_expert`` is required: the MoE block already computes the
        per-expert token count (for the fused aux loss + the routing tables),
        so the dispatcher takes it as input rather than launching another
        ``torch.bincount`` kernel.
        """
        num_tokens = hidden_states.shape[0]

        # Dense per-expert routing tables required by ``moe_permute_and_pad_with_probs``.
        routing_map = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.bool, device=hidden_states.device
        )
        routing_map.scatter_(1, selected_experts, True)
        routing_probs = torch.zeros(
            num_tokens, self.num_experts, dtype=routing_weights.dtype, device=hidden_states.device
        )
        routing_probs.scatter_(1, selected_experts, routing_weights)

        (
            permuted_hidden,
            permuted_probs,
            row_id_map,
            pad_offsets,
            padded_tokens_per_expert,
        ) = te.moe_permute_and_pad_with_probs(
            hidden_states,
            routing_probs,
            routing_map,
            tokens_per_expert,
            self.pad_align,
        )
        padded_tokens_per_expert = padded_tokens_per_expert.int()

        if self._ep_group is None or self.ep_size == 1:
            handle = _Handle(
                row_id_map=row_id_map,
                restore_shape=hidden_states.shape,
                pad_offsets=pad_offsets,
            )
            return DispatchOutput(
                expert_input=permuted_hidden,
                expert_probs=permuted_probs,
                tokens_per_expert=padded_tokens_per_expert.tolist(),
                handle=handle,
            )

        # EP > 1: ship both tokens and probs across ranks. A single packed
        # all_to_all was tried; the extra ``.contiguous()`` slicing on the
        # receive side cost more than the saved NCCL collective at Mixtral
        # batch=8 / seq=8192 (the probs comm is ~1/2048 of the token comm,
        # so the all_to_all is bandwidth-bound, not latency-bound).
        ep_group = self._ep_group
        send_counts = padded_tokens_per_expert.reshape(self.ep_size, self.num_local_experts)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts.flatten(), send_counts.flatten(), group=ep_group)

        input_split_sizes = send_counts.sum(dim=1).tolist()
        output_split_sizes = recv_counts.sum(dim=1).tolist()
        local_m_splits = recv_counts.sum(dim=0).int().tolist()

        recv_tokens = _DifferentiableAllToAll.apply(
            permuted_hidden, output_split_sizes, input_split_sizes, ep_group
        )
        recv_probs = _DifferentiableAllToAll.apply(
            permuted_probs, output_split_sizes, input_split_sizes, ep_group
        )

        sort_indices, unsort_indices = _build_expert_sort_indices(recv_counts)

        handle = _Handle(
            row_id_map=row_id_map,
            restore_shape=hidden_states.shape,
            pad_offsets=pad_offsets,
            unsort_indices=unsort_indices,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
        )
        return DispatchOutput(
            expert_input=recv_tokens[sort_indices],
            expert_probs=recv_probs[sort_indices],
            tokens_per_expert=local_m_splits,
            handle=handle,
        )

    def combine(self, expert_output: torch.Tensor, handle: _Handle) -> torch.Tensor:
        """Reverse the dispatch. ``ScaledSwiGLU`` already applied per-token probs,
        so ``moe_unpermute`` is called without ``merging_probs``."""
        if handle.unsort_indices is not None:
            combined = _DifferentiableAllToAll.apply(
                expert_output[handle.unsort_indices],
                handle.input_split_sizes,
                handle.output_split_sizes,
                self._ep_group,
            )
        else:
            combined = expert_output

        return te.moe_unpermute(
            combined,
            handle.row_id_map,
            merging_probs=None,
            restore_shape=handle.restore_shape,
            map_type="mask",
            pad_offsets=handle.pad_offsets,
        )
