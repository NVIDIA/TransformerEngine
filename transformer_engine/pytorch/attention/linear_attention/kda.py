# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kimi Delta Attention (KDA) linear attention.

KDA generalizes Gated DeltaNet's scalar per-token decay to a per-key-channel
decay vector ``alpha_t = exp(g_t)`` in ``(0, 1]^qk_head_dim``, while keeping
Gated DeltaNet's scalar write strength ``beta_t``. For state ``S_t``:

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,
    o_t = scale * q_t S_t.

The decay is applied before the delta-rule correction, so the erase term reads
the already-decayed state; unlike Gated DeltaNet, where the scalar decay
commutes with the correction, the order is part of the definition. Collapsing
``g`` to a single value per head recovers Gated DeltaNet.

Note that the state is written ``[qk_head_dim, v_head_dim]`` above, the natural
orientation for the recurrence. The module's ``initial_state``/final-state
tensors follow the cuDNN frontend's transposed convention instead,
``[batch, heads, v_head_dim, qk_head_dim]``, where the per-key-channel decay
scales the state's columns.

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


class _KDAKernelAdapter(AlignedTimelineKernelAdapter):
    """Adapter from TransformerEngine attention layouts to cuDNN frontend KDA.

    KDA's decay gate ``g`` carries one value per query/key channel, while its
    write strength ``beta`` stays scalar per token and head. The kernel reads
    ``g`` at float32 or either half precision, and ``beta`` at float32 or the
    Q/K/V dtype.

    The optional safe-gate parameters ``a_log`` and ``dt_bias`` are per-head,
    not per-token, so they travel as kernel arguments rather than gates: the
    THD conversion that the gate tensors go through does not apply to them.
    """

    variant = "KDA"
    module_name = "KimiDeltaAttention"
    op_name = "kimi_delta_attention"

    # Head sizes the cuDNN frontend's KDA engines are built for.
    supported_head_dims = (64, 128)

    # Gate dtypes the op accepts for `g` and the safe-gate parameters.
    supported_gate_dtypes = (torch.float32, torch.bfloat16, torch.float16)

    def _validate_gates(
        self,
        gates: Dict[str, torch.Tensor],
        query_layer: torch.Tensor,
    ) -> None:
        g, beta = gates["g"], gates["beta"]
        if g.dtype not in self.supported_gate_dtypes:
            raise TypeError(
                "KDA g must have dtype torch.float32, torch.bfloat16 or torch.float16, "
                f"got {g.dtype}."
            )
        if beta.dtype not in {torch.float32, query_layer.dtype}:
            raise TypeError(
                "KDA beta must have dtype torch.float32 or the Q/K/V dtype "
                f"({query_layer.dtype}), got {beta.dtype}."
            )
        token_dims = tuple(query_layer.shape[:-2])
        expected_decay_shape = (*token_dims, self.num_q_heads, self.qk_head_dim)
        expected_beta_shape = (*token_dims, self.num_q_heads)
        if g.shape != expected_decay_shape:
            raise ValueError(
                "KDA g must have shape "
                f"{expected_decay_shape} (one value per query/key channel); got "
                f"{tuple(g.shape)}."
            )
        if beta.shape != expected_beta_shape:
            raise ValueError(
                "KDA beta must have shape "
                f"{expected_beta_shape} (one value per token and head); got "
                f"{tuple(beta.shape)}."
            )

    def _validate_safe_gate_params(
        self,
        query_layer: torch.Tensor,
        a_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
    ) -> None:
        """Check the optional per-head safe-gate parameters against Q.

        These are the only kernel inputs that are not per-token, so they skip
        the THD conversion entirely. ``a_log`` is one amplitude per head, while
        ``dt_bias`` biases each query/key channel of each head.
        """
        for name, tensor, expected_shape in (
            ("a_log", a_log, (self.num_q_heads,)),
            ("dt_bias", dt_bias, (self.num_q_heads, self.qk_head_dim)),
        ):
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"KDA {name} must be a torch.Tensor when provided.")
            if tensor.dtype not in self.supported_gate_dtypes:
                raise TypeError(
                    f"KDA {name} must have dtype torch.float32, torch.bfloat16 or "
                    f"torch.float16, got {tensor.dtype}."
                )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"KDA {name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
                )
            if tensor.device != query_layer.device:
                raise ValueError(
                    f"KDA {name} must be on the same device as Q "
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
        gate_lower_bound: Optional[float] = None,
        gate_domain: str = "log",
        a_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
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
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            gate_lower_bound=gate_lower_bound,
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
        gate_lower_bound: Optional[float] = None,
        gate_domain: str = "log",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run KDA and return a TE-layout output, optionally with the final state.

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
            gate_lower_bound=gate_lower_bound,
            gate_domain=gate_domain,
            a_log=a_log,
            dt_bias=dt_bias,
        )


class KimiDeltaAttention(LinearAttentionBase):
    """Apply Kimi Delta Attention (KDA) linear attention through the cuDNN frontend.

    KDA is Gated DeltaNet with a channel-wise forget gate: the decay ``g``
    carries one value per query/key channel, where Gated DeltaNet's carries one
    per head, while the write strength ``beta`` stays scalar. The cuDNN frontend
    serves KDA on Blackwell+ (SM100/SM103/SM107).

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
               ``t = sum(s_i)`` for all sequences in the batch. Kimi Delta
               Attention is inherently causal; padded batches must use
               `qkv_format='thd'` with `cu_seqlens` passed to `forward` to
               exclude padding tokens from the recurrence.
    tp_size : int, default = 1
            tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
             tensor parallel process group.
    layer_number : int, default = `None`
                 layer number of the current `KimiDeltaAttention` when multiple
                 such modules are concatenated, for instance in consecutive
                 transformer blocks.
    scale : Optional[float], default = `None`
          scale for the Kimi Delta Attention recurrence. Defaults to
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
            if head_dim not in _KDAKernelAdapter.supported_head_dims:
                raise ValueError(
                    f"KimiDeltaAttention {name} must be one of "
                    f"{_KDAKernelAdapter.supported_head_dims}, got {head_dim}. "
                    "kv_channels sets both; pass a tuple to size them separately."
                )

        self.kda_attention = _KDAKernelAdapter(
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
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        gate_lower_bound: Optional[float] = None,
        gate_domain: str = "log",
        a_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply Kimi Delta Attention linear attention.

        Parameters
        ----------
        query_layer, key_layer, value_layer : torch.Tensor
            Query, key, and value tensors, in the layout given by `qkv_format`
            (or the module's configured `qkv_format` when omitted).
        g : torch.Tensor
            Per-key-channel decay gate, of shape matching Q/K/V's token
            dimensions followed by
            `[num_attention_heads // tp_size, qk_head_dim]`, and of dtype
            float32, bfloat16 or float16. Log-space by default
            (`alpha = exp(g)`); see `gate_domain` and `safe_gate` for the other
            two parameterizations. Required.
        beta : torch.Tensor
            Per-head write-strength gate, of shape matching Q/K/V's token
            dimensions followed by `[num_attention_heads // tp_size]`, and of
            dtype float32 or the Q/K/V dtype. Required.
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
            If true, L2-normalize Q and K inside the kernel before the
            recurrence, which is the KDA model's feature map.
        use_beta_sigmoid_in_kernel : bool, default = `False`
            If true, `beta` holds raw logits and the kernel applies the sigmoid,
            returning the gradient with respect to those logits.
        allow_neg_eigval : bool, default = `False`
            If true, the fused beta sigmoid is scaled by 2, so the delta-rule
            operator can reach negative eigenvalues. Requires
            `use_beta_sigmoid_in_kernel`.
        safe_gate : bool, default = `False`
            If true, `g` holds raw logits and the kernel applies the safe-gate
            transform `gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`
            to reach the log decay, returning the gradient with respect to
            those logits. Cannot combine with `gate_domain='linear'`.
        gate_lower_bound : Optional[float], default = `None`
            Safe-gate lower bound in log space; the kernel's default is -5.0.
            Requires `safe_gate`.
        gate_domain : str, default = `log`
            `'log'`: `g` holds `ln(alpha)`. `'linear'`: `g` holds `alpha` in
            `(0, 1]` directly, and its gradient comes back with respect to
            `alpha`.
        a_log : Optional[torch.Tensor], default = `None`
            Safe-gate per-head log-amplitude, of shape
            `[num_attention_heads // tp_size]` and dtype float32, bfloat16 or
            float16. Requires `safe_gate`; when omitted the amplitude is 1 and
            no gradient is produced for it.
        dt_bias : Optional[torch.Tensor], default = `None`
            Safe-gate per-channel bias, of shape
            `[num_attention_heads // tp_size, qk_head_dim]` and dtype float32,
            bfloat16 or float16. Requires `safe_gate`; when omitted the bias is
            zero and no gradient is produced for it.
        """
        missing = [name for name, gate in (("g", g), ("beta", beta)) if gate is None]
        if missing:
            raise ValueError(
                f"KimiDeltaAttention requires both g and beta; got no {', '.join(missing)}."
            )
        if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
            raise ValueError(
                "KimiDeltaAttention allow_neg_eigval requires "
                "use_beta_sigmoid_in_kernel, which owns the sigmoid it scales."
            )
        if gate_domain not in {"log", "linear"}:
            raise ValueError(
                "KimiDeltaAttention gate_domain must be 'log' or 'linear', "
                f"got {gate_domain!r}."
            )
        if not safe_gate:
            extra = [
                name
                for name, value in (
                    ("a_log", a_log),
                    ("dt_bias", dt_bias),
                    ("gate_lower_bound", gate_lower_bound),
                )
                if value is not None
            ]
            if extra:
                raise ValueError(
                    f"KimiDeltaAttention {', '.join(extra)} require safe_gate, which owns "
                    "the transform they parameterize."
                )
        elif gate_domain == "linear":
            raise ValueError(
                "KimiDeltaAttention gate_domain='linear' cannot combine with safe_gate; "
                "the safe-gate transform takes raw logits."
            )

        kda_kwargs = {
            "qkv_format": qkv_format if qkv_format is not None else self.qkv_format,
            "cu_seqlens": cu_seqlens,
            "output_final_state": output_final_state,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
            "allow_neg_eigval": allow_neg_eigval,
            "safe_gate": safe_gate,
            "gate_lower_bound": gate_lower_bound,
            "gate_domain": gate_domain,
        }
        with self.prepare_forward_ctx(query_layer, allow_non_contiguous=True) as query_layer:
            # a_log and dt_bias ride in the positional tuple, not kda_kwargs: under
            # checkpoint_core_attention, TE's reentrant checkpoint returns gradients
            # only for the positional arguments and drops tensors in the keyword dict.
            return self._dispatch_attention(
                self.kda_attention,
                (query_layer, key_layer, value_layer, g, beta, initial_state, a_log, dt_bias),
                kda_kwargs,
                checkpoint_core_attention,
            )
