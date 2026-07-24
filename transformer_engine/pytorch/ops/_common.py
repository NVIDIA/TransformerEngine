# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
import math
from typing import Optional

import torch

import transformer_engine_torch as tex
from transformer_engine_torch import FP8TensorMeta
from ..constants import TE_DType
from ..torch_version import torch_version
from ..quantization import FP8GlobalStateManager
from ..tensor.float8_tensor import Float8Tensor
from ..tensor.grouped_tensor import GroupedTensor
from ..tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from ..tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ..quantized_tensor import QuantizedTensorStorage
from ..utils import canonicalize_dtype, round_up_to_nearest_multiple


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


def grouped_storage_from_grouped_tensor(tensor: GroupedTensor) -> GroupedTensorStorage:
    """Repack a ``GroupedTensor`` into a ``GroupedTensorStorage``.

    ``GroupedTensor`` is a ``torch.Tensor`` subclass, so the CPU offload
    infrastructure's ``prepare_for_saving`` treats it as a plain tensor and
    does not decompose it into its component data tensors. By repacking into
    a ``GroupedTensorStorage`` (not a ``torch.Tensor``), the fuser's
    ``prepare_for_saving`` call correctly decomposes the activation before
    ``save_for_backward``.
    """
    return GroupedTensorStorage(
        shape=tensor.logical_shape,
        dtype=tensor.fake_dtype,
        num_tensors=tensor.num_tensors,
        shapes=tensor.tensor_shapes,
        quantizer=tensor.quantizer,
        data=tensor.rowwise_data,
        columnwise_data=tensor.columnwise_data,
        scale_inv=tensor.scale_inv,
        columnwise_scale_inv=tensor.columnwise_scale_inv,
        amax=tensor.amax,
        columnwise_amax=tensor.columnwise_amax,
        scale=tensor.scale,
        first_dims=tensor.first_dims,
        last_dims=tensor.last_dims,
        tensor_offsets=tensor.tensor_offsets,
        offsets=tensor.offsets,
        scale_inv_offsets=tensor.scale_inv_offsets,
        columnwise_scale_inv_offsets=tensor.columnwise_scale_inv_offsets,
        with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
        row_scaled_nvfp4=tensor.row_scaled_nvfp4,
        nvfp4_use_4over6=tensor.nvfp4_use_4over6,
        nvfp4_e4m3_max=tensor.nvfp4_e4m3_max,
    )


def prepare_prequantized_mxfp8_grouped_input(
    grouped_x: GroupedTensorStorage,
    quantizer: MXFP8Quantizer,
    num_groups: int,
    split_sizes: torch.Tensor,
    dtype: torch.dtype,
    *,
    with_columnwise: bool,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Prepare a rowwise-only MXFP8 grouped input for grouped GEMM (in place).

    Supports inputs that arrive already rowwise-quantized (e.g. FP8 token
    dispatch): the rowwise data is fed to the forward GEMM as-is, while the
    columnwise copy needed by the wgrad GEMM cannot be derived from the
    rowwise data (per-block scales differ per direction), so it is
    manufactured by dequantizing the rowwise data and requantizing
    columnwise-only. Rowwise scales must arrive in compact (unswizzled)
    format; they are converted to the GEMM-swizzled layout afterwards.
    TODO: optimize and fuse the round-trips requant
    """
    if grouped_x.rowwise_data is None:
        raise ValueError("Pre-quantized MXFP8 grouped input is missing rowwise data.")
    if grouped_x._with_gemm_swizzled_scales:
        raise NotImplementedError(
            "Pre-quantized MXFP8 grouped input must have scales in compact format."
        )
    if grouped_x.columnwise_data is not None:
        # Columnwise grouped scales have a per-group layout, so the global
        # single-tensor swizzle below cannot convert them.
        raise NotImplementedError(
            "Pre-quantized MXFP8 grouped input with compact scales must be rowwise-only."
        )
    if grouped_x.quantizer is not None and grouped_x.quantizer.dtype != quantizer.dtype:
        # The forward GEMM consumes the input's rowwise data verbatim while the
        # wgrad GEMM consumes the columnwise copy we manufacture with ``quantizer``.
        # A dtype mismatch would make the two directions disagree numerically.
        raise ValueError(
            f"Pre-quantized MXFP8 grouped input has FP8 dtype {grouped_x.quantizer.dtype}, "
            f"but the op's input quantizer expects {quantizer.dtype}."
        )

    # Manufacture columnwise data for the wgrad GEMM: dequantize the rowwise
    # wire data and requantize columnwise-only.
    if with_columnwise:
        hp_x = tex.group_dequantize(grouped_x, TE_DType[dtype])
        colwise_quantizer = quantizer.copy()
        colwise_quantizer.set_usage(rowwise=False, columnwise=True)
        colwise_quantizer.optimize_for_gemm = True
        colwise_quantizer.internal = True
        colwise_x = tex.group_quantize(
            hp_x.rowwise_data.view(grouped_x.logical_shape),
            colwise_quantizer,
            num_groups,
            split_sizes,
            tensor_offsets=tensor_offsets,
        )
        grouped_x.columnwise_data = colwise_x.columnwise_data
        grouped_x.columnwise_scale_inv = colwise_x.columnwise_scale_inv

    # Convert rowwise scales to the GEMM-swizzled layout. The grouped GEMM
    # reads activation scales as one (total_tokens, cols) matrix, so the
    # single-tensor swizzle applies. Swizzling allocates a new scale buffer;
    # the original compact scales are left untouched.
    total_tokens, cols = grouped_x.logical_shape
    scale_shape = (
        round_up_to_nearest_multiple(total_tokens, 128),
        round_up_to_nearest_multiple(cols // 32, 4),
    )
    tmp = MXFP8Tensor(
        shape=(total_tokens, cols),
        dtype=dtype,
        fp8_dtype=quantizer.dtype,
        rowwise_data=grouped_x.rowwise_data.view(total_tokens, cols),
        rowwise_scale_inv=grouped_x.scale_inv.view(scale_shape),
        columnwise_data=None,
        columnwise_scale_inv=None,
        quantizer=quantizer,
        requires_grad=False,
        with_gemm_swizzled_scales=False,
    )
    tex.swizzle_scales_for_gemm_(tmp)
    grouped_x.scale_inv = tmp._rowwise_scale_inv.view(-1)
    grouped_x._with_gemm_swizzled_scales = True


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
