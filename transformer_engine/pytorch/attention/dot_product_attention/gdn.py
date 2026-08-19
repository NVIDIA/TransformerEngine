# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN frontend Gated DeltaNet attention backend."""

import importlib
from functools import lru_cache
from typing import Callable, Optional, Tuple, Union

import torch


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


class _GatedDeltaNetAttention(torch.nn.Module):
    """Adapter from TransformerEngine attention layouts to cuDNN frontend GDN."""

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
        self._cu_seqlens_validation_cache: list[Tuple[torch.Tensor, int, int, Tuple[int, ...]]] = []

    def _validate_cu_seqlens_values(
        self,
        cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
        name: str,
        total_tokens: int,
    ) -> Tuple[int, ...]:
        """Validate THD offsets, synchronizing only for new or mutated tensors."""
        _validate_cu_seqlens(cu_seqlens, device=device, name=name)
        try:
            tensor_version = cu_seqlens._version  # pylint: disable=protected-access
        except RuntimeError:
            # Tensors created in inference mode have no version counter. They cannot be
            # cached safely because an in-place update would otherwise bypass validation.
            tensor_version = None
        for (
            cached_tensor,
            cached_version,
            cached_total,
            cached_offsets,
        ) in self._cu_seqlens_validation_cache:
            if (
                tensor_version is not None
                and cached_tensor is cu_seqlens
                and cached_version == tensor_version
                and cached_total == total_tokens
            ):
                return cached_offsets

        offsets = tuple(cu_seqlens.detach().cpu().tolist())
        if offsets[0] != 0:
            raise ValueError(f"{name} must start at 0, got {offsets[0]}.")
        for index, (start, end) in enumerate(zip(offsets, offsets[1:])):
            if end < start:
                raise ValueError(
                    f"{name} must be nondecreasing; offsets[{index}]={start} is greater "
                    f"than offsets[{index + 1}]={end}."
                )
        if offsets[-1] != total_tokens:
            raise ValueError(
                f"{name} must end at the flattened token count {total_tokens}, got {offsets[-1]}."
            )

        # Holding a few tiny offset tensors avoids pointer-reuse ambiguity while keeping
        # repeated forwards and CUDA graph replay free of validation synchronizations.
        if tensor_version is not None:
            self._cu_seqlens_validation_cache = [
                entry for entry in self._cu_seqlens_validation_cache if entry[0] is not cu_seqlens
            ]
            self._cu_seqlens_validation_cache.append(
                (cu_seqlens, tensor_version, total_tokens, offsets)
            )
            if len(self._cu_seqlens_validation_cache) > 4:
                self._cu_seqlens_validation_cache.pop(0)
        return offsets

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
        cu_seqlens_kv: Optional[torch.Tensor] = None,
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
        # The underlying op supports grouped value heads, but DPA's output contract is fixed
        # at construction time. Integrations that use more V heads must expand Q/K before DPA.
        if value_layer.shape[-2] != self.num_q_heads:
            raise ValueError(
                f"GDN V must have {self.num_q_heads} heads, got {value_layer.shape[-2]}. "
                "DotProductAttention requires its output width to match "
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
        if query_layer.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError(
                "GDN Q, K, and V must have dtype float16, bfloat16, or float32, "
                f"got {query_layer.dtype}."
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
            cu_seqlens_values = self._validate_cu_seqlens_values(
                cu_seqlens_q,
                device=device,
                name="cu_seqlens_q",
                total_tokens=query_layer.shape[0],
            )
            cu_seqlens = cu_seqlens_q
        else:
            if cu_seqlens_q is not None or cu_seqlens_kv is not None:
                raise ValueError(
                    "Dense GDN inputs do not accept cu_seqlens_q or cu_seqlens_kv. "
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
            cu_seqlens_values = tuple(range(0, batch_size * sequence_length + 1, sequence_length))

        if cu_seqlens_kv is not None:
            cu_seqlens_kv_values = self._validate_cu_seqlens_values(
                cu_seqlens_kv,
                device=device,
                name="cu_seqlens_kv",
                total_tokens=query_layer.shape[0],
            )
            if cu_seqlens_kv_values != cu_seqlens_values:
                raise ValueError(
                    "GDN requires cu_seqlens_q and cu_seqlens_kv to contain identical offsets."
                )

        batch_size = len(cu_seqlens_values) - 1
        expected_state_shape = (
            batch_size,
            num_output_heads,
            query_layer.shape[-1],
            value_layer.shape[-1],
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
