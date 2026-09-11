# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepSeekV3 MoE block: sigmoid router with aux-loss-free bias, shared +
routed experts."""

from typing import Optional, Union

import torch

import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.router import fused_topk_with_score_function
from transformer_engine.pytorch.permutation import (
    moe_permute_and_pad_with_probs,
    moe_permute_with_probs,
    moe_unpermute,
)
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    get_align_size_for_quantization,
)

__all__ = ["DeepSeekV3MoE"]


_EP_ALIGNMENT = 256
_FUSED_MLP_ROWS = 256
_FUSED_MLP_MARGIN = 1024


def _make_swiglu_mlp(hidden_size, ffn_hidden_size, dtype, device, num_experts=None):
    """Dense SwiGLU MLP, or a grouped one (probs applied inside the activation) per expert.

    The grouped variant fuses into a single CuTe grouped MLP on supported hardware.
    """
    common = {"bias": False, "dtype": dtype, "device": device}
    if num_experts is None:
        return te_ops.Sequential(
            te_ops.Linear(hidden_size, 2 * ffn_hidden_size, **common),
            te_ops.SwiGLU(),
            te_ops.Linear(ffn_hidden_size, hidden_size, **common),
        )
    return te_ops.Sequential(
        te_ops.GroupedLinear(num_experts, hidden_size, 2 * ffn_hidden_size, **common),
        te_ops.ScaledSwiGLU(glu_interleave_size=32),
        te_ops.GroupedLinear(num_experts, ffn_hidden_size, hidden_size, **common),
    )


class DeepSeekV3MoE(torch.nn.Module):
    """
    DeepSeekV3 Mixture-of-Experts block.

    Each token is scored by a sigmoid router with a non-trainable expert bias
    updated by ``update_expert_bias()`` (aux-loss-free load balancing) and,
    optionally, group-limited routing: experts are split into ``num_groups``
    groups, the top ``group_topk`` groups are selected by their summed scores,
    and the final ``topk`` experts are chosen only from those groups. Selected
    tokens run through the routed experts, a SwiGLU MLP shared across experts
    as a grouped GEMM, with the routing probability applied inside the MLP. An
    optional shared expert (dense SwiGLU MLP) is added to every token. On
    hardware that supports it the expert MLP runs as a single fused
    grouped-GEMM kernel.

    Without ``ep_group`` all experts live on the local device. With
    ``ep_group`` the experts are split across the group and tokens are
    exchanged over NCCL; this requires ``ep_bootstrap`` to be called once per
    process before constructing the module, and bfloat16 inputs.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    moe_ffn_hidden_size : int
                         ffn size of each routed expert.
    num_experts : int
                 total number of routed experts.
    topk : int, default = 8
          number of experts per token.
    num_groups : int, optional
                number of expert groups for node-limited routing; requires ``group_topk``.
    group_topk : int, optional
                number of groups each token is limited to; requires ``num_groups``.
    routed_scaling_factor : float, default = 2.5
                           scaling applied to the routing probabilities.
    shared_expert_ffn_hidden_size : int, optional
                                   ffn size of the shared expert; ``None``
                                   disables the shared expert.
    expert_bias_update_rate : float, default = 1e-3
                             step size of the aux-loss-free bias update
                             (see :meth:`update_expert_bias`).
    params_dtype : torch.dtype, optional
                  dtype of module parameters.
    ep_group : ProcessGroup, optional
              expert-parallel process group; enables the NCCL EP path.
    ep_max_tokens_per_rank : int, optional
                            max local tokens per forward (required with EP).
    """

    def __init__(
        self,
        hidden_size: int,
        moe_ffn_hidden_size: int,
        num_experts: int,
        topk: int = 8,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        routed_scaling_factor: float = 2.5,
        shared_expert_ffn_hidden_size: Optional[int] = None,
        expert_bias_update_rate: float = 1e-3,
        params_dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str] = "cuda",
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        ep_max_tokens_per_rank: Optional[int] = None,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if not 1 <= topk <= num_experts:
            raise ValueError("topk must be in [1, num_experts].")
        if (num_groups is None) != (group_topk is None):
            raise ValueError("num_groups and group_topk must be provided together.")
        if num_groups is not None:
            if num_groups <= 0 or num_experts % num_groups != 0:
                raise ValueError("num_groups must be positive and divide num_experts.")
            if not 1 <= group_topk <= num_groups:
                raise ValueError("group_topk must be in [1, num_groups].")
            if topk % group_topk != 0:
                raise ValueError("topk must be divisible by group_topk.")
            if topk // group_topk > num_experts // num_groups:
                raise ValueError("topk per group must not exceed the number of experts per group.")

        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.topk = topk
        self.num_groups = num_groups
        self.group_topk = group_topk
        self.routed_scaling_factor = routed_scaling_factor
        self.expert_bias_update_rate = expert_bias_update_rate

        self.gate = torch.nn.Linear(
            hidden_size, num_experts, bias=False, dtype=dtype, device=device
        )
        self.register_buffer(
            "expert_bias", torch.zeros(num_experts, dtype=torch.float32, device=device)
        )
        self._last_tokens_per_expert: Optional[torch.Tensor] = None

        self.ep_group = ep_group
        self.ep_size = 1 if ep_group is None else torch.distributed.get_world_size(ep_group)
        assert num_experts % self.ep_size == 0
        num_local_experts = num_experts // self.ep_size

        self.experts = _make_swiglu_mlp(
            hidden_size, moe_ffn_hidden_size, dtype, device, num_experts=num_local_experts
        )

        self.shared_expert = None
        if shared_expert_ffn_hidden_size is not None:
            self.shared_expert = _make_swiglu_mlp(
                hidden_size, shared_expert_ffn_hidden_size, dtype, device
            )

        self._ep_buffer_kwargs = None
        if ep_group is not None:
            assert ep_max_tokens_per_rank is not None, "EP requires ep_max_tokens_per_rank."
            cap = self.ep_recv_capacity(
                self.ep_size, ep_max_tokens_per_rank, topk, num_local_experts
            )
            self._ep_buffer_kwargs = {
                "top_k": topk,
                "max_tokens_per_rank": ep_max_tokens_per_rank,
                "hidden_dim": hidden_size,
                "num_local_experts": num_local_experts,
                "recv_capacity_per_rank": cap,
                "alignment": _EP_ALIGNMENT,
            }

    @staticmethod
    def ep_recv_capacity(
        ep_size: int, max_tokens_per_rank: int, topk: int, num_local_experts: int
    ) -> int:
        """Recv rows per rank for ``ep_bootstrap``: worst-case routing plus per-expert
        alignment padding and the fused grouped MLP margin, rounded to its row multiple."""
        cap = ep_size * max_tokens_per_rank * topk
        cap += num_local_experts * _EP_ALIGNMENT + _FUSED_MLP_MARGIN
        return -(-cap // _FUSED_MLP_ROWS) * _FUSED_MLP_ROWS

    def _route(self, logits: torch.Tensor, topk_indices: Optional[torch.Tensor] = None):
        return fused_topk_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=False,
            num_groups=self.num_groups,
            group_topk=self.group_topk,
            scaling_factor=self.routed_scaling_factor,
            score_function="sigmoid",
            expert_bias=self.expert_bias,
            topk_indices=topk_indices,
        )

    def _forward_local(self, tokens: torch.Tensor) -> torch.Tensor:
        probs, routing_map = self._route(self.gate(tokens).float())
        tokens_per_expert = routing_map.sum(dim=0)
        self._last_tokens_per_expert = tokens_per_expert.detach()

        # Quantized grouped GEMMs need every expert's row count aligned.
        align = 1
        if FP8GlobalStateManager.is_fp8_enabled():
            align = get_align_size_for_quantization(FP8GlobalStateManager.get_fp8_recipe())
        if align > 1:
            permuted, permuted_probs, row_id_map, pad_offsets, tokens_per_expert = (
                moe_permute_and_pad_with_probs(tokens, probs, routing_map, tokens_per_expert, align)
            )
        else:
            permuted, permuted_probs, row_id_map = moe_permute_with_probs(
                tokens, probs, routing_map, num_out_tokens=tokens.shape[0] * self.topk
            )
            pad_offsets = None

        # The fused grouped MLP requires the total row count to be a multiple
        # of 128; rows beyond sum(tokens_per_expert) fall outside every group.
        num_rows = permuted.shape[0]
        pad = (-num_rows) % 128
        if pad:
            permuted = torch.nn.functional.pad(permuted, (0, 0, 0, pad))
            permuted_probs = torch.nn.functional.pad(permuted_probs, (0, pad))

        out = self.experts(
            permuted, tokens_per_expert, permuted_probs.to(tokens.dtype), tokens_per_expert
        )
        return moe_unpermute(
            out[:num_rows], row_id_map, restore_shape=tokens.shape, pad_offsets=pad_offsets
        )

    def make_ep_buffer(self, device: Optional[torch.device] = None):
        """EP buffer for this module's routing config. ``forward`` creates one per call unless
        given one; a buffer holds one call's routing state until its backward runs, so reuse
        it only across calls whose backward has completed."""
        from transformer_engine.pytorch.ep import EpBuffer

        assert self._ep_buffer_kwargs is not None, "make_ep_buffer requires ep_group."
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return EpBuffer(**self._ep_buffer_kwargs, device=device)

    def _forward_ep(self, tokens: torch.Tensor, ep_buffer=None) -> torch.Tensor:
        from transformer_engine.pytorch.ep import ep_dispatch, ep_combine

        assert tokens.dtype == torch.bfloat16, "The EP path requires bfloat16 inputs."
        buffer = ep_buffer if ep_buffer is not None else self.make_ep_buffer(tokens.device)
        topk_idx = torch.empty(
            (tokens.shape[0], self.topk), dtype=torch.int64, device=tokens.device
        )
        probs, topk_idx = self._route(self.gate(tokens).float(), topk_indices=topk_idx)
        flat_idx = topk_idx.flatten()
        self._last_tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.long, device=tokens.device
        ).scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        topk_weights = probs.gather(1, topk_idx)

        # NCCL EP zero-fills the alignment padding between experts itself, so
        # the recv/grad buffers can stay uninitialized. The fused grouped MLP
        # reads up to a tile past the last expert, so zero a margin there
        # (offsets stay on device: no host sync).
        cap = buffer.recv_capacity_per_rank
        recv_tokens, recv_weights, tokens_per_expert = ep_dispatch(
            buffer,
            tokens,
            topk_idx,
            topk_weights,
            recv_tokens=torch.empty(
                (cap, self.hidden_size), dtype=tokens.dtype, device=tokens.device
            ),
            recv_topk_weights=torch.empty((cap,), dtype=torch.float32, device=tokens.device),
        )
        grad_out = torch.empty((cap, self.hidden_size), dtype=tokens.dtype, device=tokens.device)
        with torch.no_grad():
            margin = (
                torch.arange(_FUSED_MLP_MARGIN, device=tokens.device) + tokens_per_expert.sum()
            ).clamp_(max=cap - 1)
            for buf in (recv_tokens, recv_weights, grad_out):
                buf.detach().index_fill_(0, margin, 0)
        expert_out = self.experts(
            recv_tokens, tokens_per_expert, recv_weights.to(tokens.dtype), tokens_per_expert
        )
        return ep_combine(buffer, expert_out, num_local_tokens=tokens.shape[0], grad_out=grad_out)

    def forward(self, hidden_states: torch.Tensor, ep_buffer=None) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
                       input of shape ``[..., hidden_size]``.
        ep_buffer : EpBuffer, optional
                   buffer from :meth:`make_ep_buffer` to reuse instead of creating one
                   per call (EP only).
        """
        tokens = hidden_states.reshape(-1, self.hidden_size)
        if self.ep_group is not None:
            out = self._forward_ep(tokens, ep_buffer)
        else:
            out = self._forward_local(tokens)
        if self.shared_expert is not None:
            out = out + self.shared_expert(tokens)
        return out.view_as(hidden_states)

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        """Aux-loss-free bias update from the last forward's routing counts.

        With data/expert parallelism, all-reduce ``_last_tokens_per_expert``
        across ranks before calling (or call on identically-routed ranks).
        """
        counts = self._last_tokens_per_expert
        if counts is None:
            return
        err = counts.float().mean() - counts.float()
        self.expert_bias += self.expert_bias_update_rate * torch.sign(err)
