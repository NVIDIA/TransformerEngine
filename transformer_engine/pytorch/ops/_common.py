# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
import math
from typing import Optional

import torch

from transformer_engine_torch import FP8TensorMeta
from ..torch_version import torch_version
from ..quantization import FP8GlobalStateManager
from ..quantized_tensor import QuantizedTensorStorage, Quantizer
from ..tensor import (
    Float8BlockQuantizer,
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
)
from ..tensor.float8_tensor import Float8Tensor
from ..utils import canonicalize_dtype


def get_fused_normalization_quantizer(
    quantizer: Optional[Quantizer],
) -> Optional[Quantizer]:
    """Return a quantizer supported by fused normalization kernels."""
    if isinstance(
        quantizer,
        (
            Float8Quantizer,
            Float8CurrentScalingQuantizer,
            MXFP8Quantizer,
            Float8BlockQuantizer,
            NVFP4Quantizer,
        ),
    ):
        return quantizer
    return None


def validate_or_alloc_output(
    buffer: Optional[torch.Tensor],
    shape: tuple[int, ...] | list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the caller's output buffer, or allocate one if it is None.

    The buffer must be a contiguous, non-grad tensor matching the required
    shape, dtype, and device. Validation reads host-side metadata only. If the
    buffer is reused across iterations, pass ``buffer.detach()`` so autograd does
    not set its ``requires_grad`` (which would trip the non-grad check here on the
    next call).
    """
    shape = tuple(shape)
    if buffer is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if tuple(buffer.shape) != shape:
        raise ValueError(f"Output buffer shape {tuple(buffer.shape)} does not match {shape}.")
    if buffer.dtype != dtype:
        raise ValueError(f"Output buffer dtype {buffer.dtype} does not match {dtype}.")
    if buffer.device != device:
        raise ValueError(f"Output buffer device {buffer.device} does not match {device}.")
    if not buffer.is_contiguous():
        raise ValueError("Output buffer must be contiguous.")
    if buffer.requires_grad:
        raise ValueError("Output buffer must not require gradient.")
    return buffer


def is_quantized_tensor(tensor: torch.Tensor | QuantizedTensorStorage) -> bool:
    """Check if tensor is a quantized tensor"""
    return isinstance(tensor, QuantizedTensorStorage)


def maybe_dequantize(
    tensor: torch.Tensor | QuantizedTensorStorage, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize tensor to given dtype or just convert if not a quantized tensor"""
    if is_quantized_tensor(tensor):
        return tensor.dequantize(dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def maybe_autocast_dtype(
    *,
    device_type: str = "cuda",
    default_dtype: Optional[torch.dtype] = None,
) -> torch.dtype:
    """Get autocast dtype if enabled"""

    if torch_version() >= (2, 4, 3):
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    else:
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return canonicalize_dtype(default_dtype)


def get_fp8_meta_from_fp8_tensor(tensor: Float8Tensor) -> tuple[FP8TensorMeta, int]:
    """Get FP8TensorMeta object and index corresponding to Float8Tensor

    Constructs FP8TensorMeta if needed.

    """

    # Check if tensor already has FP8 metadata
    if tensor._fp8_meta is not None:
        key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=tensor._fp8_meta_forward,
        )
        return tensor._fp8_meta[key], tensor._fp8_meta_index

    # Create FP8TensorMeta class
    fp8_meta = FP8TensorMeta()
    fp8_meta.scale = tensor._scale_inv.reciprocal()
    fp8_meta.amax_history = torch.empty(1, 1, dtype=torch.float32, device=tensor.device)
    fp8_meta.scale_inv = tensor._scale_inv
    return fp8_meta, 0


def get_main_grad_from_param(
    weight_param: torch.nn.Parameter,
    *,
    op_label: str = "",
) -> torch.Tensor:
    """Refresh ``main_grad`` from FSDP (if applicable) and return it.
    Used by Megatron-LM-style wgrad fusion paths
    (``accumulate_into_main_grad=True``) to obtain the buffer the wgrad GEMM
    will write into.
    Raises if the parameter does not have a ``main_grad`` attribute or if it
    is ``None``.
    """
    if hasattr(weight_param, "__fsdp_param__"):
        weight_param.main_grad = weight_param.get_main_grad()
    if not hasattr(weight_param, "main_grad") or weight_param.main_grad is None:
        prefix = f"{op_label} " if op_label else ""
        raise RuntimeError(
            f"{prefix}operation is configured with accumulate_into_main_grad=True, "
            "but weight parameter does not have a valid main_grad attribute"
        )
    return weight_param.main_grad


def get_accumulate_flag_in_param(weight_param: torch.nn.Parameter) -> bool:
    """Return whether the wgrad GEMM should accumulate into ``main_grad``.

    Returns ``False`` (i.e. overwrite) when the parameter has
    ``overwrite_main_grad=True`` (used in Megatron-FSDP), and ``True``
    otherwise.
    """
    return not getattr(weight_param, "overwrite_main_grad", False)


def view_main_grad_as_grouped_buffer(
    main_grad: torch.Tensor,
    num_groups: int,
    weight_shape: tuple[int, ...],
    *,
    label: str = "",
) -> torch.Tensor:
    """Return ``main_grad`` viewed as ``(num_groups, *weight_shape)`` without copy.
    Raises if the numel doesn't match or if the existing stride pattern does
    not allow a zero-copy view to the grouped layout.
    """
    grouped_shape = (num_groups, *weight_shape)
    if tuple(main_grad.shape) == grouped_shape:
        return main_grad
    prefix = f"{label} " if label else "Grouped weight "
    if main_grad.numel() != math.prod(grouped_shape):
        raise RuntimeError(
            f"{prefix}main_grad expected shape {grouped_shape} or matching numel, "
            f"but got shape {tuple(main_grad.shape)}"
        )
    try:
        return main_grad.view(grouped_shape)
    except RuntimeError as e:
        raise RuntimeError(
            f"{prefix}main_grad must be viewable as {grouped_shape} without copy, "
            f"but got shape {tuple(main_grad.shape)} and stride "
            f"{tuple(main_grad.stride())}"
        ) from e


def get_dummy_wgrads_for_params(
    weight_params: list[torch.nn.Parameter],
) -> list[Optional[torch.Tensor]]:
    """Build dummy ``.grad`` placeholders for Megatron-LM wgrad-fusion params.

    For each parameter that exposes ``grad_added_to_main_grad``, set the flag
    to ``True`` and return a dummy wgrad tensor (zeroed if
    ``zero_out_wgrad`` is also set on the parameter). For parameters without
    the flag, the corresponding entry is ``None``.

    The returned list has the same length and order as ``weight_params``.
    """
    from ..module.base import get_dummy_wgrad  # pylint: disable=import-outside-toplevel

    out: list[Optional[torch.Tensor]] = []
    for wp in weight_params:
        if hasattr(wp, "grad_added_to_main_grad"):
            wp.grad_added_to_main_grad = True
            out.append(
                get_dummy_wgrad(
                    list(wp.size()),
                    wp.dtype,
                    zero=getattr(wp, "zero_out_wgrad", False),
                )
            )
        else:
            out.append(None)
    return out
