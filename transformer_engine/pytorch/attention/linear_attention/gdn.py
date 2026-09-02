# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated DeltaNet linear attention.

This module is **experimental** and subject to change.
"""

import importlib
import math
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.distributed import get_distributed_world_size, checkpoint
from transformer_engine.pytorch.jit import no_torch_dynamo


@lru_cache(maxsize=1)
def _import_gated_delta_net() -> Callable:
    """Import the cuDNN frontend GDN custom op lazily."""
    try:
        ops = importlib.import_module("cudnn.linear_attention.ops")
        return ops.gated_delta_net
    except (AttributeError, ImportError) as exc:
        raise ImportError(
            "GDN attention requires a nvidia-cudnn-frontend installation that provides "
            "cudnn.linear_attention.ops.gated_delta_net and its kernel runtime. Install "
            "cuDNN frontend from the matching source revision with the 'cutedsl' extra."
        ) from exc


def _to_thd(tensor: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """Convert a dense sequence tensor to the THD layout required by cuDNN GDN."""
    if qkv_format == "thd":
        return tensor
    if qkv_format == "sbhd":
        tensor = tensor.transpose(0, 1)
    return tensor.reshape(-1, *tensor.shape[2:])


def _validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    device: torch.device,
    name: str,
) -> None:
    """Validate a cumulative sequence-length tensor for GDN."""
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if cu_seqlens.dim() != 1:
        raise ValueError(f"{name} must have shape [batch_size + 1].")
    if cu_seqlens.numel() < 2:
        raise ValueError(f"{name} must contain at least a start and end offset.")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {cu_seqlens.dtype}.")
    if cu_seqlens.device != device:
        raise ValueError(f"{name} must be on {device}, got {cu_seqlens.device}.")


class _GDNKernelAdapter(torch.nn.Module):
    """Adapter from TransformerEngine attention layouts to cuDNN frontend GDN.

    The cuDNN frontend GDN op is differentiated through ``torch.autograd`` (it registers
    its own backward internally), so this module does not define an explicit backward.
    """

    def __init__(
        self,
        scale: float,
        num_q_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.num_q_heads = num_q_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self._dense_cu_seqlens_key: Optional[Tuple[torch.device, int, int]] = None
        self._dense_cu_seqlens: Optional[torch.Tensor] = None

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
        cu_seqlens_q: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run GDN and return a TE-layout output, optionally with the final state."""
        if qkv_format not in {"thd", "bshd", "sbhd"}:
            raise ValueError(
                "GDN attention only supports qkv_format={'thd', 'bshd', 'sbhd'}, "
                f"got {qkv_format!r}."
            )

        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (query_layer, key_layer, value_layer, g, beta)
        ):
            raise TypeError("GDN Q, K, V, g, and beta must be torch.Tensor instances.")
        if initial_state is not None and not isinstance(initial_state, torch.Tensor):
            raise TypeError("GDN initial_state must be a torch.Tensor when provided.")

        expected_rank = 3 if qkv_format == "thd" else 4
        qkv = (query_layer, key_layer, value_layer)
        if any(tensor.dim() != expected_rank for tensor in qkv):
            raise ValueError(
                f"Q, K, and V must be {expected_rank}D tensors for qkv_format={qkv_format!r}."
            )
        if query_layer.shape != key_layer.shape:
            raise ValueError(
                "GDN requires Q and K to have the same shape; got "
                f"{tuple(query_layer.shape)} and {tuple(key_layer.shape)}."
            )
        if query_layer.shape[:-2] != value_layer.shape[:-2]:
            raise ValueError("GDN requires Q, K, and V to have the same token dimensions.")
        if query_layer.shape[-2] != self.num_q_heads:
            raise ValueError(
                f"GDN Q and K must have {self.num_q_heads} heads, got {query_layer.shape[-2]}."
            )
        # The underlying op supports grouped value heads, but GatedDeltaNetAttention's
        # output contract is fixed at construction time. Integrations that use more V
        # heads must expand Q/K before GatedDeltaNetAttention.
        if value_layer.shape[-2] != self.num_q_heads:
            raise ValueError(
                f"GDN V must have {self.num_q_heads} heads, got {value_layer.shape[-2]}. "
                "GatedDeltaNetAttention requires its output width to match "
                "num_attention_heads * v_head_dim."
            )
        if query_layer.shape[-1] != self.qk_head_dim:
            raise ValueError(
                "GDN Q and K head dimension must match kv_channels; expected "
                f"{self.qk_head_dim}, got {query_layer.shape[-1]}."
            )
        if value_layer.shape[-1] != self.v_head_dim:
            raise ValueError(
                "GDN V head dimension must match kv_channels; expected "
                f"{self.v_head_dim}, got {value_layer.shape[-1]}."
            )

        device = query_layer.device
        if any(not tensor.is_cuda for tensor in qkv):
            raise ValueError("GDN attention only supports CUDA tensors.")
        if any(tensor.device != device for tensor in (*qkv, g, beta)):
            raise ValueError("Q, K, V, g, and beta must be on the same CUDA device.")
        if query_layer.dtype != key_layer.dtype or query_layer.dtype != value_layer.dtype:
            raise TypeError("Q, K, and V must have the same dtype for GDN attention.")
        if query_layer.dtype not in {torch.float16, torch.bfloat16}:
            raise TypeError(
                "GDN Q, K, and V must have dtype float16 or bfloat16 (no cuDNN frontend "
                f"GDN engine supports float32 inputs), got {query_layer.dtype}."
            )
        if g.dtype != torch.float32 or beta.dtype != torch.float32:
            raise TypeError(
                "GDN g and beta must have dtype torch.float32 (the kernel-native dtype)."
            )

        num_output_heads = self.num_q_heads
        expected_gate_shape = (*query_layer.shape[:-2], num_output_heads)
        if g.shape != expected_gate_shape or beta.shape != expected_gate_shape:
            raise ValueError(
                "GDN g and beta must both have shape "
                f"{expected_gate_shape}; got {tuple(g.shape)} and {tuple(beta.shape)}."
            )
        if qkv_format == "thd":
            if cu_seqlens_q is None:
                raise ValueError("cu_seqlens_q is required for GDN with qkv_format='thd'.")
            _validate_cu_seqlens(
                cu_seqlens_q,
                device=device,
                name="cu_seqlens_q",
            )
            cu_seqlens = cu_seqlens_q
        else:
            if cu_seqlens_q is not None:
                raise ValueError(
                    "Dense GDN inputs do not accept cu_seqlens_q. "
                    "Use qkv_format='thd' for packed or ragged batches."
                )
            if qkv_format == "bshd":
                batch_size, sequence_length = query_layer.shape[:2]
            else:
                sequence_length, batch_size = query_layer.shape[:2]
            cache_key = (device, batch_size, sequence_length)
            if self._dense_cu_seqlens_key != cache_key:
                self._dense_cu_seqlens = (
                    torch.arange(batch_size + 1, dtype=torch.int32, device=device) * sequence_length
                )
                self._dense_cu_seqlens_key = cache_key
            cu_seqlens = self._dense_cu_seqlens

        batch_size = cu_seqlens.shape[0] - 1
        expected_state_shape = (
            batch_size,
            num_output_heads,
            value_layer.shape[-1],
            query_layer.shape[-1],
        )
        if initial_state is not None:
            if initial_state.device != device:
                raise ValueError(
                    f"GDN initial_state must be on {device}, got {initial_state.device}."
                )
            if initial_state.dtype != torch.float32:
                raise TypeError(
                    f"GDN initial_state must have dtype torch.float32, got {initial_state.dtype}."
                )
            if initial_state.shape != expected_state_shape:
                raise ValueError(
                    f"GDN initial_state must have shape {expected_state_shape}, "
                    f"got {tuple(initial_state.shape)}."
                )

        q_thd = _to_thd(query_layer, qkv_format)
        k_thd = _to_thd(key_layer, qkv_format)
        v_thd = _to_thd(value_layer, qkv_format)
        g_thd = _to_thd(g, qkv_format)
        beta_thd = _to_thd(beta, qkv_format)
        gated_delta_net = _import_gated_delta_net()
        try:
            output, final_state = gated_delta_net(
                q_thd,
                k_thd,
                v_thd,
                g_thd,
                beta_thd,
                cu_seqlens,
                scale=self.scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        except ImportError as exc:
            raise ImportError(
                "The cuDNN frontend GDN kernel runtime is unavailable. Install cuDNN "
                "frontend from the matching source revision with the 'cutedsl' extra."
            ) from exc

        if qkv_format == "thd":
            output = output.reshape(output.shape[0], -1)
        else:
            output = output.reshape(batch_size, sequence_length, -1)
            if qkv_format == "sbhd":
                output = output.transpose(0, 1).contiguous()

        if output_final_state:
            return output, final_state
        return output


def _needs_eager_gdn_attention(call: Dict[str, Any]) -> Optional[str]:
    """Why this GatedDeltaNetAttention call has to run outside the graph, or None.

    `call` maps `GatedDeltaNetAttention.forward`'s parameter names to the arguments
    this call passed, including `self`.
    """
    if call.get("checkpoint_core_attention", False):
        return "activation checkpointing of the attention"
    return None


class GatedDeltaNetAttention(TransformerEngineBaseModule):
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
                 layer number of the current `GatedDeltaNetAttention` when multiple such
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
                "GatedDeltaNetAttention is inherently causal and only supports "
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

        self.gdn_attention = _GDNKernelAdapter(
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

    @no_torch_dynamo(when=_needs_eager_gdn_attention)
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
                "GatedDeltaNetAttention requires both g and beta; "
                f"got g={'set' if g is not None else 'None'} and "
                f"beta={'set' if beta is not None else 'None'}."
            )
        if FP8GlobalStateManager.is_fp8_enabled() or FP8GlobalStateManager.is_fp8_calibration():
            raise ValueError(
                "GatedDeltaNetAttention does not support FP8 autocast or FP8 calibration."
            )
        if cu_seqlens_kv is not None and cu_seqlens_kv is not cu_seqlens_q:
            raise ValueError(
                "GatedDeltaNetAttention requires identical Q and KV sequence boundaries. Pass "
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
