# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Public variant-dispatch API for linear attention."""

import math
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch

from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.distributed import get_distributed_world_size, checkpoint
from transformer_engine.pytorch.jit import no_torch_dynamo

from .gdn import GDNConfig, _GDNLinearAttentionBackend

_LINEAR_ATTENTION_BACKENDS: Dict[Type[Any], Type[torch.nn.Module]] = {
    GDNConfig: _GDNLinearAttentionBackend,
}


def _needs_eager_linear_attention(call: Dict[str, Any]) -> Optional[str]:
    """Why this LinearAttention call has to run outside the graph, or None.

    ``call`` maps :meth:`LinearAttention.forward` parameter names to the arguments
    supplied by the caller, including ``self``.
    """
    if call.get("checkpoint_core_attention", False):
        return "activation checkpointing of the attention"
    return None


class LinearAttention(TransformerEngineBaseModule):
    """Dispatch linear attention to the backend selected by ``variant``.

    ``LinearAttention`` provides stable, algorithm-independent module and forward
    interfaces. A typed variant configuration selects an internal backend and holds its
    static algorithm or kernel choices. Per-call, variant-specific tensors are supplied
    together through the corresponding runtime-input type.

    Parameters
    ----------
    variant : object
             typed configuration that selects the linear-attention backend.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : Union[int, Tuple[int, int]]
                head size for query/key and value tensors. If ``int``, the same size
                is used for both; if ``Tuple[int, int]``, the first element is the
                query/key head size and the second is the value head size.
    qkv_format : str, default = `sbhd`
               dimension format for query_layer, key_layer and value_layer,
               {`sbhd`, `bshd`, `thd`}. `s` stands for sequence length, `b` for batch
               size, `h` for number of heads, `d` for head size, and `t` for total
               tokens, with ``t = sum(s_i)`` for all sequences in the batch.
    attn_mask_type : str, default = `causal`
                    attention mask semantics requested from the selected variant.
    tp_size : int, default = 1
            tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
             tensor parallel process group.
    layer_number : int, default = `None`
                 layer number of the current ``LinearAttention`` when multiple such
                 modules are concatenated, for instance in consecutive transformer blocks.
    scale : Optional[float], default = `None`
           multiplicative query scale. Defaults to ``1.0 / sqrt(qk_head_dim)``.
    name : Optional[str], default = `None`
          module name.
    """

    def __init__(
        self,
        variant: object,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        tp_size: int = 1,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        scale: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        try:
            backend_cls = _LINEAR_ATTENTION_BACKENDS[type(variant)]
        except KeyError as exc:
            raise TypeError(
                f"Unsupported linear-attention configuration: {type(variant).__name__}"
            ) from exc

        super().__init__(name=name)

        self.variant = variant
        self.qkv_format = qkv_format
        attn_mask_type = attn_mask_type.replace(",", "_")
        if attn_mask_type == "causal_padding":
            attn_mask_type = "padding_causal"
        self.attn_mask_type = attn_mask_type

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)

        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}.")
        if num_attention_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"tp_size ({self.tp_size})."
            )

        self.num_attention_heads = num_attention_heads
        self.layer_number = 1 if layer_number is None else layer_number
        self.qk_head_dim = kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        self.v_head_dim = kv_channels if isinstance(kv_channels, int) else kv_channels[1]

        if scale is None:
            scale = 1.0 / math.sqrt(self.qk_head_dim)
        self.scale = scale

        self.backend = backend_cls(
            variant=variant,
            scale=scale,
            num_q_heads=num_attention_heads // self.tp_size,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            attn_mask_type=attn_mask_type,
        )

    def _checkpointed_attention_forward(
        self,
        attention_func: Callable,
        *forward_args: Tuple[torch.Tensor, ...],
        **forward_kwargs: Dict[str, Any],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward method with activation checkpointing."""

        def custom_forward(*input_args, **input_kwargs):
            return attention_func(*input_args, **input_kwargs)

        return checkpoint(
            custom_forward,
            distribute_saved_activations=False,
            get_rng_state_tracker=None,
            tp_group=self.tp_group,
            *forward_args,
            **forward_kwargs,
        )

    @no_torch_dynamo(when=_needs_eager_linear_attention)
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        variant_inputs: object,
        *,
        qkv_format: Optional[str] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        checkpoint_core_attention: bool = False,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the configured linear-attention variant.

        Parameters
        ----------
        query_layer, key_layer, value_layer : torch.Tensor
            Query, key, and value tensors in the layout given by ``qkv_format`` (or the
            module's configured ``qkv_format`` when omitted).
        variant_inputs : object
            Typed per-call inputs required by the selected variant. Differentiable tensors
            contained in this object are unpacked and passed positionally through activation
            checkpointing.
        qkv_format : Optional[str], default = `None`
            Overrides the module's configured ``qkv_format`` for this call.
        cu_seqlens_q : Optional[torch.Tensor], default = `None`
            Cumulative query sequence lengths for packed (``thd``) inputs, with shape
            ``[batch_size + 1]`` and dtype ``torch.int32``.
        cu_seqlens_kv : Optional[torch.Tensor], default = `None`
            Cumulative key/value sequence lengths for packed (``thd``) inputs, with shape
            ``[batch_size + 1]`` and dtype ``torch.int32``. A variant may require query and
            key/value sequence boundaries to be aligned.
        checkpoint_core_attention : bool, default = `False`
            If true, forward activations are recomputed during the backward pass.
        initial_state : Optional[torch.Tensor], default = `None`
            Optional recurrent state whose shape and dtype are defined by the variant.
        output_final_state : bool, default = `False`
            If true, also return the variant's final recurrent state.
        """
        variant_args = self.backend.unpack_variant_inputs(variant_inputs)
        self.backend.validate_runtime_environment()
        backend_kwargs = {
            "qkv_format": qkv_format if qkv_format is not None else self.qkv_format,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "output_final_state": output_final_state,
        }
        forward_args = (
            query_layer,
            key_layer,
            value_layer,
            *variant_args,
            initial_state,
        )

        with self.prepare_forward_ctx(
            query_layer,
            num_gemms=self.backend.num_gemms,
            allow_non_contiguous=True,
        ) as query_layer:
            forward_args = (query_layer, *forward_args[1:])
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(
                    self.backend,
                    *forward_args,
                    **backend_kwargs,
                )
            return self.backend(*forward_args, **backend_kwargs)
