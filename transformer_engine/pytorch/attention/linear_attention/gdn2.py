# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated DeltaNet v2 linear attention.

Gated DeltaNet v2 (GDN-2) generalizes Gated DeltaNet's two scalar gates to three
channel-wise gates: a per-key-channel log decay ``g``, a per-key-channel erase
gate ``beta``, and a per-value-channel write gate ``w``. For state ``S_t``:

    S_t = (I - k_t (beta_t . k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t . v_t)^T,
    o_t = scale * q_t S_t.

Collapsing all three gates to scalars recovers Gated DeltaNet.

Note that the state is written ``[qk_head_dim, v_head_dim]`` above, the natural
orientation for the recurrence. The module's ``initial_state``/final-state
tensors follow the cuDNN frontend's transposed convention instead,
``[batch, heads, v_head_dim, qk_head_dim]``, where the per-key-channel gates
scale the state's columns.

This module is **experimental** and subject to change.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import torch

from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.jit import no_torch_dynamo

from .base import (
    LinearAttentionBase,
    LinearAttentionKernelAdapter,
    _needs_eager_linear_attention,
)


class _GDN2KernelAdapter(LinearAttentionKernelAdapter):
    """Adapter from TransformerEngine attention layouts to cuDNN frontend GDN-2.

    GDN-2's gates are channel-wise: ``g`` and ``beta`` carry one value per
    query/key channel and ``w`` one value per value channel. The kernel reads
    ``beta`` and ``w`` at the Q/K/V dtype, while ``g`` may also be float32.
    """

    variant = "GDN2"
    module_name = "GatedDeltaNet2Attention"
    op_name = "gated_delta_net_v2"

    # Head sizes the cuDNN frontend's only GDN-2 engine (FROST) is built for.
    supported_head_dims = (64, 128)

    def _validate_gates(
        self,
        gates: Dict[str, torch.Tensor],
        query_layer: torch.Tensor,
    ) -> None:
        g, beta, w = gates["g"], gates["beta"], gates["w"]
        if g.dtype not in {torch.float32, query_layer.dtype}:
            raise TypeError(
                "GDN2 g must have dtype torch.float32 or the Q/K/V dtype "
                f"({query_layer.dtype}), got {g.dtype}."
            )
        if beta.dtype != query_layer.dtype or w.dtype != query_layer.dtype:
            raise TypeError(
                "GDN2 beta and w must have the same dtype as Q, K, and V "
                f"({query_layer.dtype}), got {beta.dtype} and {w.dtype}."
            )
        token_dims = tuple(query_layer.shape[:-2])
        expected_key_gate_shape = (*token_dims, self.num_q_heads, self.qk_head_dim)
        expected_value_gate_shape = (*token_dims, self.num_q_heads, self.v_head_dim)
        if g.shape != expected_key_gate_shape or beta.shape != expected_key_gate_shape:
            raise ValueError(
                "GDN2 g and beta must both have shape "
                f"{expected_key_gate_shape} (one value per query/key channel); got "
                f"{tuple(g.shape)} and {tuple(beta.shape)}."
            )
        if w.shape != expected_value_gate_shape:
            raise ValueError(
                "GDN2 w must have shape "
                f"{expected_value_gate_shape} (one value per value channel); got "
                f"{tuple(w.shape)}."
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
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        beta_guard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return op(
            query_layer,
            key_layer,
            value_layer,
            gates["g"],
            gates["beta"],
            gates["w"],
            cu_seqlens,
            scale=self.scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
        )

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        w: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        *,
        qkv_format: str,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        beta_guard: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run GDN-2 and return a TE-layout output, optionally with the final state."""
        return self._run(
            query_layer,
            key_layer,
            value_layer,
            {"g": g, "beta": beta, "w": w},
            initial_state,
            qkv_format=qkv_format,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
        )


class GatedDeltaNet2Attention(LinearAttentionBase):
    """Apply Gated DeltaNet v2 linear attention through the cuDNN frontend.

    Gated DeltaNet v2 replaces Gated DeltaNet's scalar gates with channel-wise
    ones: ``g`` and ``beta`` carry one value per query/key channel and ``w`` one
    value per value channel. The cuDNN frontend serves GDN-2 on Blackwell+
    (SM100/SM103/SM107) only.

    This module is **experimental** and subject to change.

    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : Union[int, Tuple[int, int]]
                head size for query/key and value tensors. If ``int``, the same size
                is used for both; if ``Tuple[int, int]``, the first element is the
                query/key head size and the second is the value head size. Both
                must be 64 or 128.
    qkv_format : str, default = `sbhd`
               dimension format for query_layer, key_layer and value_layer,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length,
               `b` batch size, `h` the number of heads, `d` head size, and
               `t` the total number of tokens in a batch, with
               ``t = sum(s_i)`` for all sequences in the batch. Gated DeltaNet v2
               is inherently causal; padded batches must use `qkv_format='thd'`
               with `cu_seqlens` passed to `forward` to exclude padding tokens
               from the recurrence.
    tp_size : int, default = 1
            tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
             tensor parallel process group.
    layer_number : int, default = `None`
                 layer number of the current `GatedDeltaNet2Attention` when multiple
                 such modules are concatenated, for instance in consecutive
                 transformer blocks.
    scale : Optional[float], default = `None`
          scale for the Gated DeltaNet v2 recurrence. Defaults to
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

        # Reject unsupported head sizes here rather than letting them reach cuDNN
        # frontend engine selection, which reports them as a kernel-level failure.
        for name, head_dim in (
            ("query/key head size", self.qk_head_dim),
            ("value head size", self.v_head_dim),
        ):
            if head_dim not in _GDN2KernelAdapter.supported_head_dims:
                raise ValueError(
                    f"GatedDeltaNet2Attention {name} must be one of "
                    f"{_GDN2KernelAdapter.supported_head_dims}, got {head_dim}. "
                    "kv_channels sets both; pass a tuple to size them separately."
                )

        self.gdn2_attention = _GDN2KernelAdapter(
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
        w: Optional[torch.Tensor] = None,
        *,
        qkv_format: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        checkpoint_core_attention: bool = False,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        beta_guard: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply Gated DeltaNet v2 linear attention.

        Parameters
        ----------
        query_layer, key_layer, value_layer : torch.Tensor
            Query, key, and value tensors, in the layout given by `qkv_format`
            (or the module's configured `qkv_format` when omitted).
        g : torch.Tensor
            Per-key-channel log-decay gate, of shape matching Q/K/V's token
            dimensions followed by
            `[num_attention_heads // tp_size, qk_head_dim]`. Either float32 or
            the Q/K/V dtype. Required.
        beta : torch.Tensor
            Per-key-channel erase gate, of the same shape as `g` and of the
            Q/K/V dtype. Required.
        w : torch.Tensor
            Per-value-channel write gate, of shape matching Q/K/V's token
            dimensions followed by
            `[num_attention_heads // tp_size, v_head_dim]`, of the Q/K/V dtype.
            Required.
        qkv_format : Optional[str], default = `None`
            Overrides the module's configured `qkv_format` for this call.
        cu_seqlens : Optional[torch.Tensor], default = `None`
            Cumulative sequence lengths for packed (`thd`) inputs, of shape
            `[batch_size + 1]` and dtype `torch.int32`.
        checkpoint_core_attention : bool, default = `False`
            If true, forward activations for this module are recomputed
            during the backward pass instead of saved.
        initial_state : Optional[torch.Tensor], default = `None`
            Recurrent state to seed the recurrence with, of shape
            `[batch_size, num_attention_heads // tp_size, v_head_dim, qk_head_dim]`
            and dtype `torch.float32`.
        output_final_state : bool, default = `False`
            If true, also return the final recurrent state.
        use_qk_l2norm_in_kernel : bool, default = `False`
            If true, L2-normalize Q and K inside the kernel before the recurrence.
        use_beta_sigmoid_in_kernel : bool, default = `False`
            If true, `beta` holds raw logits and the kernel applies the sigmoid,
            returning the gradient with respect to those logits.
        allow_neg_eigval : bool, default = `False`
            If true, the fused beta sigmoid is scaled by 2, so the delta-rule
            operator can reach negative eigenvalues. Requires
            `use_beta_sigmoid_in_kernel`.
        beta_guard : bool, default = `False`
            If true, apply the erase-side beta safeguard: rows whose per-channel
            beta contrast would make the erase step expansive are shrunk toward
            the key-weighted mean beta. The backward is straight-through.
            Requires `use_qk_l2norm_in_kernel`, since the guard is defined on
            the normalized key.
        """
        missing = [name for name, gate in (("g", g), ("beta", beta), ("w", w)) if gate is None]
        if missing:
            raise ValueError(
                "GatedDeltaNet2Attention requires all of g, beta, and w; "
                f"got no {', '.join(missing)}."
            )
        if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
            raise ValueError(
                "GatedDeltaNet2Attention allow_neg_eigval requires "
                "use_beta_sigmoid_in_kernel, which owns the sigmoid it scales."
            )
        if beta_guard and not use_qk_l2norm_in_kernel:
            raise ValueError(
                "GatedDeltaNet2Attention beta_guard requires use_qk_l2norm_in_kernel; "
                "the guard is defined on the normalized key."
            )

        gdn2_kwargs = {
            "qkv_format": qkv_format if qkv_format is not None else self.qkv_format,
            "cu_seqlens": cu_seqlens,
            "output_final_state": output_final_state,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
            "allow_neg_eigval": allow_neg_eigval,
            "beta_guard": beta_guard,
        }
        with self.prepare_forward_ctx(query_layer, allow_non_contiguous=True) as query_layer:
            return self._dispatch_attention(
                self.gdn2_attention,
                (query_layer, key_layer, value_layer, g, beta, w, initial_state),
                gdn2_kwargs,
                checkpoint_core_attention,
            )
