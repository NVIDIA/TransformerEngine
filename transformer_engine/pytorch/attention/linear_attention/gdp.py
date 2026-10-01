# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated DeltaProduct linear attention.

Gated DeltaProduct (GDP) generalizes Gated DeltaNet by applying
``n = num_householder`` beta-gated Householder (delta-rule) updates per token
instead of one. The scalar decay ``alpha_t`` acts once, before the token's
updates, and the readout follows the last one. For state ``S_t``:

    S <- alpha_t S_{t-1}
    S <- (I - beta_{t,j} k_{t,j} k_{t,j}^T) S + beta_{t,j} k_{t,j} v_{t,j}^T,
                                                              j = 0 .. n - 1
    S_t = S,  o_t = scale * q_t S_t.

With ``num_householder=1``, the recurrence is mathematically equivalent to Gated
DeltaNet under equivalent gate settings. The extra updates let a single token
move the state further than one rank-one correction can, which raises the
expressivity of the recurrence without lengthening the token timeline.

The decay tensor ``g`` reaches ``alpha_t`` through one of three
parameterizations, so that a model can hand the kernel whatever its gate
projection already produces:

* ``gate_domain='log'`` (default): ``g`` holds ``ln(alpha)``, so
  ``alpha = exp(g)``.
* ``gate_domain='linear'``: ``g`` holds ``alpha`` itself, and the gradient comes
  back with respect to ``alpha``.
* ``safe_gate=True``: ``g`` holds raw pre-activation logits, and
  ``ln(alpha) = -exp(a_log) * softplus(g + dt_bias)`` for the optional per-head
  parameters ``a_log`` and ``dt_bias``. This is the Mamba-style gate, and it is
  numerically safe in the sense that ``alpha`` stays in ``(0, 1]`` for any
  finite logit.

Q and the decay ``g`` therefore live on the real-token timeline, while K, V and
``beta`` live on an expanded timeline carrying ``n`` sub-token rows per token.
This module takes that expansion as an explicit ``num_householder`` axis placed
just after the token dimensions, so K, V and ``beta`` have exactly one more
dimension than their Gated DeltaNet counterparts.

Note that the state is written ``[qk_head_dim, v_head_dim]`` above, the natural
orientation for the recurrence. The module's ``initial_state``/final-state
tensors follow the cuDNN frontend's transposed convention instead,
``[batch, heads, v_head_dim, qk_head_dim]``.

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


class _GDPKernelAdapter(LinearAttentionKernelAdapter):
    """Adapter from TransformerEngine attention layouts to cuDNN frontend GDP.

    GDP gates the recurrence with one scalar per token and head (the log decay
    ``g``) and one scalar per sub-token and head (the write strength ``beta``).
    K, V and ``beta`` carry a ``num_householder`` axis that Q and ``g`` do not.
    """

    variant = "GDP"
    module_name = "GatedDeltaProductAttention"
    op_name = "gated_delta_product"

    # Head sizes the cuDNN frontend's only GDP engine (FROST) is built for.
    supported_head_dims = (64, 128)

    def __init__(
        self,
        scale: float,
        num_q_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        num_householder: int,
    ) -> None:
        super().__init__(scale, num_q_heads, qk_head_dim, v_head_dim)
        self.num_householder = num_householder

    def _validate_qkv_shapes(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_format: str,
    ) -> None:
        """Check Q against K/V, which carry ``num_householder`` rows per Q row."""
        token_rank = 1 if qkv_format == "thd" else 2
        if query_layer.dim() != token_rank + 2:
            raise ValueError(
                f"GDP Q must be a {token_rank + 2}D tensor for qkv_format={qkv_format!r}, "
                f"got {tuple(query_layer.shape)}."
            )
        token_dims = tuple(query_layer.shape[:token_rank])
        for name, tensor in (("K", key_layer), ("V", value_layer)):
            if tensor.dim() != token_rank + 3:
                raise ValueError(
                    f"GDP {name} must be a {token_rank + 3}D tensor for "
                    f"qkv_format={qkv_format!r}: Q's token dimensions, then a "
                    "num_householder axis, then [heads, head_dim]. Got "
                    f"{tuple(tensor.shape)}."
                )
            if tuple(tensor.shape[:token_rank]) != token_dims:
                raise ValueError(
                    f"GDP requires Q and {name} to have the same token dimensions; got "
                    f"{token_dims} and {tuple(tensor.shape[:token_rank])}."
                )
            if tensor.shape[token_rank] != self.num_householder:
                raise ValueError(
                    f"GDP {name} must carry {self.num_householder} sub-token rows per token "
                    f"in its num_householder axis (dimension {token_rank}), got "
                    f"{tensor.shape[token_rank]}."
                )
        if key_layer.shape[-2:] != query_layer.shape[-2:]:
            raise ValueError(
                "GDP requires Q and K to have the same heads and head dimension; got "
                f"{tuple(query_layer.shape[-2:])} and {tuple(key_layer.shape[-2:])}."
            )
        self._validate_head_geometry(query_layer, value_layer)

    def _validate_gates(
        self,
        gates: Dict[str, torch.Tensor],
        query_layer: torch.Tensor,
    ) -> None:
        g, beta = gates["g"], gates["beta"]
        for name, gate in (("g", g), ("beta", beta)):
            if gate.dtype not in {torch.float32, query_layer.dtype}:
                raise TypeError(
                    f"GDP {name} must have dtype torch.float32 or the Q/K/V dtype "
                    f"({query_layer.dtype}), got {gate.dtype}."
                )
        token_dims = tuple(query_layer.shape[:-2])
        expected_decay_shape = (*token_dims, self.num_q_heads)
        # beta gates each Householder update, so it lives on the expanded timeline.
        expected_write_shape = (*token_dims, self.num_householder, self.num_q_heads)
        if g.shape != expected_decay_shape:
            raise ValueError(
                f"GDP g must have shape {expected_decay_shape} (one decay per token "
                f"and head); got {tuple(g.shape)}."
            )
        if beta.shape != expected_write_shape:
            raise ValueError(
                f"GDP beta must have shape {expected_write_shape} (one write strength "
                f"per sub-token and head); got {tuple(beta.shape)}."
            )

    def _validate_safe_gate_params(
        self,
        query_layer: torch.Tensor,
        a_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
    ) -> None:
        """Check the optional per-head safe-gate parameters against Q.

        These are the only kernel inputs that are neither per-token nor
        per-sub-token, so they carry one value per head and skip the THD
        conversion entirely.
        """
        for name, tensor in (("a_log", a_log), ("dt_bias", dt_bias)):
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"GDP {name} must be a torch.Tensor when provided.")
            if tensor.dtype not in {torch.float32, torch.bfloat16, torch.float16}:
                raise TypeError(
                    f"GDP {name} must have dtype torch.float32, torch.bfloat16 or "
                    f"torch.float16, got {tensor.dtype}."
                )
            if tuple(tensor.shape) != (self.num_q_heads,):
                raise ValueError(
                    f"GDP {name} must have shape {(self.num_q_heads,)} (one value per "
                    f"head); got {tuple(tensor.shape)}."
                )
            if tensor.device != query_layer.device:
                raise ValueError(
                    f"GDP {name} must be on the same device as Q "
                    f"({query_layer.device}), got {tensor.device}."
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
        safe_gate: bool = False,
        gate_domain: str = "log",
        a_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The THD conversion flattens the batch into the token dimension but leaves
        # the num_householder axis standing. Folding it into the rows is what puts
        # sub-token j of token t at row t * n + j, which is the timeline the kernel
        # reads.
        key_layer, value_layer, beta = (
            tensor.reshape(-1, *tensor.shape[2:])
            for tensor in (key_layer, value_layer, gates["beta"])
        )
        return op(
            query_layer,
            key_layer,
            value_layer,
            gates["g"],
            beta,
            cu_seqlens,
            self.num_householder,
            scale=self.scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            gate_domain=gate_domain,
            a_log=a_log,
            dt_bias=dt_bias,
        )

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        a_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        *,
        qkv_format: str,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        gate_domain: str = "log",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run GDP and return a TE-layout output, optionally with the final state.

        ``a_log`` and ``dt_bias`` are positional rather than keyword-only because
        they take gradients: TE's reentrant activation checkpoint only returns
        gradients for the positional arguments it forwards, and drops any tensor
        passed inside its keyword dict. Keep them here, alongside
        ``initial_state``, for as long as that holds.
        """
        # Checked here rather than in `_validate_gates`, which only sees the
        # per-token gates, so that a bad shape is reported before the kernel
        # runtime is imported.
        self._validate_safe_gate_params(query_layer, a_log, dt_bias)
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
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            gate_domain=gate_domain,
            a_log=a_log,
            dt_bias=dt_bias,
        )


class GatedDeltaProductAttention(LinearAttentionBase):
    """Apply Gated DeltaProduct linear attention through the cuDNN frontend.

    Gated DeltaProduct applies ``num_householder`` beta-gated Householder
    updates per token, with one scalar decay per token applied before them.
    With ``num_householder=1``, the GDP recurrence is mathematically equivalent
    to GDN under equivalent gate settings. K, V and ``beta`` therefore carry
    ``num_householder`` sub-token rows per token, passed as an explicit axis
    just after the token dimensions. The cuDNN frontend serves GDP on
    Blackwell+ (SM100/SM103/SM107) only.

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
    num_householder : int, default = 1
                    number of beta-gated Householder updates applied per token.
                    Must be positive. K, V and `beta` carry this many rows per
                    token; with `num_householder=1` the recurrence is
                    mathematically equivalent to Gated DeltaNet under equivalent
                    gate settings.
    qkv_format : str, default = `sbhd`
               dimension format for query_layer, key_layer and value_layer,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length,
               `b` batch size, `h` the number of heads, `d` head size, and
               `t` the total number of tokens in a batch, with
               ``t = sum(s_i)`` for all sequences in the batch. The format
               describes the real-token timeline that Q follows; K and V insert
               their `num_householder` axis after `s` (or after `t`). Gated
               DeltaProduct is inherently causal; padded batches must use
               `qkv_format='thd'` with `cu_seqlens` passed to `forward` to
               exclude padding tokens from the recurrence.
    tp_size : int, default = 1
            tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
             tensor parallel process group.
    layer_number : int, default = `None`
                 layer number of the current `GatedDeltaProductAttention` when
                 multiple such modules are concatenated, for instance in
                 consecutive transformer blocks.
    scale : Optional[float], default = `None`
          scale for the Gated DeltaProduct recurrence. Defaults to
          ``1.0 / sqrt(kv_channels)`` (or the query/key head size, if
          ``kv_channels`` is a tuple).
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        num_householder: int = 1,
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

        if not isinstance(num_householder, int) or isinstance(num_householder, bool):
            raise TypeError(
                "GatedDeltaProductAttention num_householder must be an int, "
                f"got {type(num_householder).__name__}."
            )
        if num_householder < 1:
            raise ValueError(
                "GatedDeltaProductAttention num_householder must be positive, "
                f"got {num_householder}."
            )
        self.num_householder = num_householder

        # Reject unsupported head sizes here rather than letting them reach cuDNN
        # frontend engine selection, which reports them as a kernel-level failure.
        for name, head_dim in (
            ("query/key head size", self.qk_head_dim),
            ("value head size", self.v_head_dim),
        ):
            if head_dim not in _GDPKernelAdapter.supported_head_dims:
                raise ValueError(
                    f"GatedDeltaProductAttention {name} must be one of "
                    f"{_GDPKernelAdapter.supported_head_dims}, got {head_dim}. "
                    "kv_channels sets both; pass a tuple to size them separately."
                )

        self.gdp_attention = _GDPKernelAdapter(
            self.scale,
            self.num_attention_heads_per_partition,
            self.qk_head_dim,
            self.v_head_dim,
            self.num_householder,
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
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        gate_domain: str = "log",
        a_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply Gated DeltaProduct linear attention.

        Parameters
        ----------
        query_layer : torch.Tensor
            Query tensor on the real-token timeline, in the layout given by
            `qkv_format` (or the module's configured `qkv_format` when omitted),
            with head dimensions `[num_attention_heads // tp_size, qk_head_dim]`.
        key_layer, value_layer : torch.Tensor
            Key and value tensors on the expanded timeline: `query_layer`'s token
            dimensions, then a `num_householder` axis, then
            `[num_attention_heads // tp_size, qk_head_dim]` for `key_layer` and
            `[num_attention_heads // tp_size, v_head_dim]` for `value_layer`. Row
            `j` of a token's axis is the `j`-th Householder update applied to
            that token, in order.
        g : torch.Tensor
            Per-head decay gate on the real-token timeline, of shape matching
            `query_layer`'s token dimensions followed by the number of attention
            heads on this tensor-parallel rank (`num_attention_heads // tp_size`).
            Either float32 or the Q/K/V dtype. Applied once per token, before that
            token's Householder updates. Holds the log decay `ln(alpha)` by
            default; see `gate_domain` and `safe_gate` for the alternatives.
            Required.
        beta : torch.Tensor
            Per-head write-strength gate on the expanded timeline, of shape
            matching `query_layer`'s token dimensions followed by
            `[num_householder, num_attention_heads // tp_size]`. Either float32 or
            the Q/K/V dtype. Required.
        qkv_format : Optional[str], default = `None`
            Overrides the module's configured `qkv_format` for this call.
        cu_seqlens : Optional[torch.Tensor], default = `None`
            Cumulative sequence lengths for packed (`thd`) inputs, of shape
            `[batch_size + 1]` and dtype `torch.int32`. These count real tokens,
            not the expanded sub-token rows.
        checkpoint_core_attention : bool, default = `False`
            If true, forward activations for this module are recomputed
            during the backward pass instead of saved.
        initial_state : Optional[torch.Tensor], default = `None`
            Recurrent state to seed the Gated DeltaProduct recurrence with, of shape
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
            If true, the fused beta sigmoid is scaled by 2, so each Householder
            operator can reach negative eigenvalues -- a reflection rather than a
            projection. Requires `use_beta_sigmoid_in_kernel`.
        safe_gate : bool, default = `False`
            If true, `g` holds raw pre-activation logits and the kernel derives
            the log decay as `-exp(a_log) * softplus(g + dt_bias)`, returning the
            gradient with respect to those logits. Cannot combine with
            `gate_domain='linear'`.
        gate_domain : str, default = `"log"`
            How `g` parameterizes the decay `alpha`, one of {`log`, `linear`}.
            `log` reads `g` as `ln(alpha)`; `linear` reads `g` as `alpha` itself
            (floored at 1e-10 in the kernel) and returns the gradient with
            respect to `alpha`. Ignored when `safe_gate` is set, which owns the
            transform.
        a_log : Optional[torch.Tensor], default = `None`
            Per-head safe-gate log-amplitude, of shape
            `[num_attention_heads // tp_size]` and dtype float32, bfloat16 or
            float16. `None` means unit amplitude (`exp(a_log) = 1`) and produces
            no gradient. Requires `safe_gate`.
        dt_bias : Optional[torch.Tensor], default = `None`
            Per-head safe-gate bias, of the same shape and dtypes as `a_log`.
            `None` means zero bias and produces no gradient. Requires
            `safe_gate`.
        """
        if g is None or beta is None:
            raise ValueError(
                "GatedDeltaProductAttention requires both g and beta; "
                f"got g={'set' if g is not None else 'None'} and "
                f"beta={'set' if beta is not None else 'None'}."
            )
        if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
            raise ValueError(
                "GatedDeltaProductAttention allow_neg_eigval requires "
                "use_beta_sigmoid_in_kernel, which owns the sigmoid it scales."
            )
        if gate_domain not in {"log", "linear"}:
            raise ValueError(
                "GatedDeltaProductAttention gate_domain must be 'log' or 'linear', "
                f"got {gate_domain!r}."
            )
        if safe_gate and gate_domain == "linear":
            raise ValueError(
                "GatedDeltaProductAttention safe_gate cannot combine with "
                "gate_domain='linear'; the safe gate already maps its logits to a "
                "log decay."
            )
        if not safe_gate:
            unsupported = [
                name
                for name, tensor in (("a_log", a_log), ("dt_bias", dt_bias))
                if tensor is not None
            ]
            if unsupported:
                raise ValueError(
                    f"GatedDeltaProductAttention {' and '.join(unsupported)} "
                    "may only be passed with safe_gate=True; they parameterize the "
                    "safe-gate transform and have no meaning without it."
                )

        gdp_kwargs = {
            "qkv_format": qkv_format if qkv_format is not None else self.qkv_format,
            "cu_seqlens": cu_seqlens,
            "output_final_state": output_final_state,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
            "allow_neg_eigval": allow_neg_eigval,
            "safe_gate": safe_gate,
            "gate_domain": gate_domain,
        }
        with self.prepare_forward_ctx(query_layer, allow_non_contiguous=True) as query_layer:
            # a_log and dt_bias ride in the positional tuple, not gdp_kwargs: under
            # checkpoint_core_attention, TE's reentrant checkpoint returns gradients
            # only for the positional arguments and drops tensors in the keyword dict.
            return self._dispatch_attention(
                self.gdp_attention,
                (query_layer, key_layer, value_layer, g, beta, initial_state, a_log, dt_bias),
                gdp_kwargs,
                checkpoint_core_attention,
            )
