# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear attention (Gated DeltaNet) via the cuDNN frontend."""
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.distributed import get_distributed_world_size, checkpoint
from transformer_engine.pytorch.jit import no_torch_dynamo

from transformer_engine.pytorch.attention.linear_attention.gdn import _GatedDeltaNetAttention


def _needs_eager_linear_attention(call: Dict[str, Any]) -> Optional[str]:
    """Why this LinearAttention call has to run outside the graph, or None.

    `call` maps `LinearAttention.forward`'s parameter names to the arguments
    this call passed, including `self`.
    """
    if call.get("checkpoint_core_attention", False):
        return "activation checkpointing of the attention"
    return None


class LinearAttention(TransformerEngineBaseModule):
    """Apply Gated DeltaNet linear attention through the cuDNN frontend.

    Unlike :class:`~transformer_engine.pytorch.DotProductAttention`, this module does not
    compute a softmax-weighted sum of value vectors. It instead maintains a per-head
    recurrent state that is updated token-by-token with a gated delta rule, so its
    ``forward`` only accepts the inputs the Gated DeltaNet recurrence uses.

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
               `t` the total number of tokens in a batch, with with with
               ``t = sum(s_i)`` for all sequences in the batch.
    attn_mask_type : str, default = `causal`
                    Gated DeltaNet is inherently causal; only `causal` and
                    `padding_causal` are accepted.
    tp_size : int, default = 1
            tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
             tensor parallel process group.
    layer_number : int, default = `None`
                 layer number of the current `LinearAttention` when multiple such
                 modules are concatenated, for instance in consecutive transformer blocks.
    softmax_scale : Optional[float], default = `None`
                  softmax scale for the Gated DeltaNet recurrence. Defaults to
                  ``1.0 / sqrt(kv_channels)`` (or the query/key head size, if
                  ``kv_channels`` is a tuple).
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        tp_size: int = 1,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        self.qkv_format = qkv_format
        attn_mask_type = attn_mask_type.replace(",", "_")
        if attn_mask_type == "causal_padding":
            attn_mask_type = "padding_causal"
        if attn_mask_type not in {"causal", "padding_causal"}:
            raise ValueError(
                "LinearAttention is inherently causal and only supports "
                f"attn_mask_type='causal' or 'padding_causal', got {attn_mask_type!r}."
            )
        self.attn_mask_type = attn_mask_type

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)

        self.num_attention_heads = num_attention_heads
        self.layer_number = 1 if layer_number is None else layer_number

        self.qk_head_dim = kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        self.v_head_dim = kv_channels if isinstance(kv_channels, int) else kv_channels[1]

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

        self.gdn_attention = _GatedDeltaNetAttention(
            softmax_scale,
            num_attention_heads // self.tp_size,
            self.qk_head_dim,
            self.v_head_dim,
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
        g: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        *,
        qkv_format: Optional[str] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
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
            followed by the number of attention heads. Required.
        beta : torch.Tensor
            Per-head write-strength gate, of the same shape as `g`. Required.
        qkv_format : Optional[str], default = `None`
            Overrides the module's configured `qkv_format` for this call.
        cu_seqlens_q : Optional[torch.Tensor], default = `None`
            Cumulative sequence lengths for packed (`thd`) inputs, of shape
            `[batch_size + 1]` and dtype `torch.int32`.
        cu_seqlens_kv : Optional[torch.Tensor], default = `None`
            Gated DeltaNet is self-attention over fully packed sequences, so Q and KV
            must share identical sequence boundaries. Pass `None`, or the same tensor
            object as `cu_seqlens_q`.
        checkpoint_core_attention : bool, default = `False`
            If true, forward activations for this module are recomputed
            during the backward pass instead of saved.
        initial_state : Optional[torch.Tensor], default = `None`
            Recurrent state to seed the Gated DeltaNet recurrence with, of shape
            `[batch_size, num_attention_heads, v_head_dim, qk_head_dim]`.
        output_final_state : bool, default = `False`
            If true, also return the final recurrent state.
        use_qk_l2norm_in_kernel : bool, default = `False`
            If true, L2-normalize Q and K inside the kernel before the recurrence.
        """
        if g is None or beta is None:
            raise ValueError(
                "LinearAttention requires both g and beta; "
                f"got g={'set' if g is not None else 'None'} and "
                f"beta={'set' if beta is not None else 'None'}."
            )
        if FP8GlobalStateManager.is_fp8_enabled() or FP8GlobalStateManager.is_fp8_calibration():
            raise ValueError("LinearAttention does not support FP8 autocast or FP8 calibration.")
        if cu_seqlens_kv is not None and cu_seqlens_kv is not cu_seqlens_q:
            raise ValueError(
                "LinearAttention requires identical Q and KV sequence boundaries. Pass "
                "cu_seqlens_kv=None or pass the same tensor object as cu_seqlens_q."
            )

        gdn_kwargs = {
            "qkv_format": qkv_format if qkv_format is not None else self.qkv_format,
            "cu_seqlens_q": cu_seqlens_q,
            "output_final_state": output_final_state,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        }
        with self.prepare_forward_ctx(
            query_layer,
            num_gemms=3,
            allow_non_contiguous=True,
        ) as query_layer:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(
                    self.gdn_attention,
                    query_layer,
                    key_layer,
                    value_layer,
                    g,
                    beta,
                    initial_state,
                    **gdn_kwargs,
                )
            return self.gdn_attention(
                query_layer,
                key_layer,
                value_layer,
                g,
                beta,
                initial_state,
                **gdn_kwargs,
            )
