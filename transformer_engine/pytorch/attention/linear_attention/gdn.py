# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated DeltaNet linear attention.

This module is **experimental** and subject to change.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import torch

from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.jit import no_torch_dynamo

from .base import (
    AlignedTimelineKernelAdapter,
    LinearAttentionBase,
    _needs_eager_linear_attention,
)


class _GDNKernelAdapter(AlignedTimelineKernelAdapter):
    """Adapter from TransformerEngine attention layouts to cuDNN frontend GDN.

    GDN gates the recurrence with one scalar per token and head: a log decay
    ``g`` and a write strength ``beta``.
    """

    variant = "GDN"
    module_name = "GatedDeltaNetAttention"
    op_name = "gated_delta_net"

    def _validate_gates(
        self,
        gates: Dict[str, torch.Tensor],
        query_layer: torch.Tensor,
    ) -> None:
        g, beta = gates["g"], gates["beta"]
        if g.dtype != torch.float32 or beta.dtype != torch.float32:
            raise TypeError(
                "GDN g and beta must have dtype torch.float32 (the kernel-native dtype)."
            )
        expected_gate_shape = (*query_layer.shape[:-2], self.num_q_heads)
        if g.shape != expected_gate_shape or beta.shape != expected_gate_shape:
            raise ValueError(
                "GDN g and beta must both have shape "
                f"{expected_gate_shape}; got {tuple(g.shape)} and {tuple(beta.shape)}."
            )

    def _call_op(
        self,
        op: Callable,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        gates: Dict[str, torch.Tensor],
        cu_seqlens: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return op(
            query_layer,
            key_layer,
            value_layer,
            gates["g"],
            gates["beta"],
            cu_seqlens,
            scale=self.scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        *,
        qkv_format: str,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run GDN and return a TE-layout output, optionally with the final state."""
        return self._run(
            query_layer,
            key_layer,
            value_layer,
            {"g": g, "beta": beta},
            initial_state,
            qkv_format=qkv_format,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


class GatedDeltaNetAttention(LinearAttentionBase):
    """Apply Gated DeltaNet linear attention through the cuDNN frontend.

    This module is **experimental** and subject to change.

    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : Union[int, Tuple[int, int]]
                head size for query/key and value tensors. If ``int``, the same size
                is used for both; if ``Tuple[int, int]``, the first element is the
                query/key head size and the second is the value head size.
    qkv_format : str, default = `sbhd`
               dimension format for query_layer, key_layer and value_layer,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length,
               `b` batch size, `h` the number of heads, `d` head size, and
               `t` the total number of tokens in a batch, with
               ``t = sum(s_i)`` for all sequences in the batch. Gated DeltaNet is
               inherently causal; padded batches must use `qkv_format='thd'` with
               `cu_seqlens` passed to `forward` to exclude padding tokens from the
               recurrence.
    tp_size : int, default = 1
            tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
             tensor parallel process group.
    layer_number : int, default = `None`
                 layer number of the current `GatedDeltaNetAttention` when multiple such
                 modules are concatenated, for instance in consecutive transformer blocks.
    scale : Optional[float], default = `None`
          scale for the Gated DeltaNet recurrence. Defaults to
          ``1.0 / sqrt(kv_channels)`` (or the query/key head size, if
          ``kv_channels`` is a tuple).
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        qkv_format: str = "sbhd",
        tp_size: int = 1,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__(
            num_attention_heads,
            kv_channels,
            qkv_format=qkv_format,
            tp_size=tp_size,
            tp_group=tp_group,
            layer_number=layer_number,
            scale=scale,
        )

        self.gdn_attention = _GDNKernelAdapter(
            self.scale,
            self.num_attention_heads_per_partition,
            self.qk_head_dim,
            self.v_head_dim,
        )

    @no_torch_dynamo(when=_needs_eager_linear_attention)
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        *,
        qkv_format: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        checkpoint_core_attention: bool = False,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply Gated DeltaNet linear attention.

        Parameters
        ----------
        query_layer, key_layer, value_layer : torch.Tensor
            Query, key, and value tensors, in the layout given by `qkv_format`
            (or the module's configured `qkv_format` when omitted).
        g : torch.Tensor
            Per-head log-decay gate, of shape matching Q/K/V's token dimensions
            followed by the number of attention heads on this tensor-parallel
            rank (`num_attention_heads // tp_size`). Required.
        beta : torch.Tensor
            Per-head write-strength gate, of the same shape as `g`. Required.
        qkv_format : Optional[str], default = `None`
            Overrides the module's configured `qkv_format` for this call.
        cu_seqlens : Optional[torch.Tensor], default = `None`
            Cumulative sequence lengths for packed (`thd`) inputs, of shape
            `[batch_size + 1]` and dtype `torch.int32`.
        checkpoint_core_attention : bool, default = `False`
            If true, forward activations for this module are recomputed
            during the backward pass instead of saved.
        initial_state : Optional[torch.Tensor], default = `None`
            Recurrent state to seed the Gated DeltaNet recurrence with, of shape
            `[batch_size, num_attention_heads // tp_size, v_head_dim, qk_head_dim]`
            and dtype `torch.float32`.
        output_final_state : bool, default = `False`
            If true, also return the final recurrent state.
        use_qk_l2norm_in_kernel : bool, default = `False`
            If true, L2-normalize Q and K inside the kernel before the recurrence.
        """
        if g is None or beta is None:
            raise ValueError(
                "GatedDeltaNetAttention requires both g and beta; "
                f"got g={'set' if g is not None else 'None'} and "
                f"beta={'set' if beta is not None else 'None'}."
            )

        gdn_kwargs = {
            "qkv_format": qkv_format if qkv_format is not None else self.qkv_format,
            "cu_seqlens": cu_seqlens,
            "output_final_state": output_final_state,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        }
        with self.prepare_forward_ctx(query_layer, allow_non_contiguous=True) as query_layer:
            return self._dispatch_attention(
                self.gdn_attention,
                (query_layer, key_layer, value_layer, g, beta, initial_state),
                gdn_kwargs,
                checkpoint_core_attention,
            )
