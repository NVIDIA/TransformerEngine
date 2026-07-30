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
from ..constants import MXFP8_BLOCK_SCALING_SIZE, TE_DType
from ..torch_version import torch_version
from ..quantization import FP8GlobalStateManager
from ..tensor.float8_tensor import Float8Tensor
from ..tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from ..tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ..quantized_tensor import QuantizedTensorStorage
from ..triton.grouped_dbias_dscales import compute_grouped_dbias
from ..utils import canonicalize_dtype


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


def prepare_prequantized_mxfp8_input_for_gemm(
    grouped_x: GroupedTensorStorage,
    quantizer: MXFP8Quantizer,
    num_groups: int,
    split_sizes: torch.Tensor,
    dtype: torch.dtype,
    *,
    with_columnwise: bool,
    with_dbias: bool = False,
    with_dequantized: bool = False,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Make an already-quantized MXFP8 grouped input GEMM-ready (in place).

    For inputs that arrive rowwise-quantized (e.g. FP8 token dispatch), where the
    high-precision tensor no longer exists. The rowwise data feeds the GEMM as-is
    and its scales are swizzled; the columnwise copy cannot be derived from it
    (the two directions scale along perpendicular axes) so it is manufactured by
    dequantize + columnwise-only requantize.

    The input must be rowwise-only with unswizzled scales, and each group's token
    count must be a multiple of 128 so per-group scales start on a swizzle-tile
    boundary.

    TODO: optimize and fuse the round-trips requant

    Parameters
    ----------
    grouped_x : GroupedTensorStorage
        Rowwise-quantized input, updated in place.
    quantizer : MXFP8Quantizer
        The op's input quantizer. Supplies the FP8 dtype for the columnwise copy.
    num_groups : int
        Number of groups.
    split_sizes : torch.Tensor
        Per-group row counts, on device.
    dtype : torch.dtype
        High-precision dtype to dequantize to.
    with_columnwise : bool
        Whether to build the columnwise copy that the wgrad GEMM consumes.
    with_dbias : bool, default = ``False``
        Whether to also produce the per-group bias gradient. When a columnwise
        copy is built the quantize kernel accumulates it in the stage it is
        already running, so it costs no extra pass; otherwise it is reduced from
        the dequantized tensor.
    with_dequantized : bool, default = ``False``
        Whether to dequantize even when no columnwise copy is needed. It must be
        requested here rather than recovered later, since the rowwise scales are
        swizzled before returning and dequantization requires unswizzled ones.
    tensor_offsets : torch.Tensor, optional
        Per-group element offsets for the columnwise requantize.

    Returns
    -------
    torch.Tensor or None
        Per-group bias gradient, when ``with_dbias``.
    torch.Tensor or None
        Dequantized input, when it was materialized (``with_columnwise``, where it
        is a byproduct, or ``with_dequantized``). Used by callers that cannot use
        the fused ``dbias``, e.g. ``scale_bias``, whose dbias/dscales depend on
        the routing probabilities.
    """
    if grouped_x.rowwise_data is None:
        raise ValueError("Pre-quantized MXFP8 grouped input is missing rowwise data.")
    if grouped_x.scale_inv is None:
        raise ValueError("Pre-quantized MXFP8 grouped input is missing rowwise scales.")
    if grouped_x._with_gemm_swizzled_scales:
        raise NotImplementedError("Pre-quantized MXFP8 grouped input must have unswizzled scales.")
    if grouped_x.columnwise_data is not None:
        # Columnwise grouped scales have a per-group layout, so the global
        # single-tensor swizzle below cannot convert them.
        raise NotImplementedError(
            "Pre-quantized MXFP8 grouped input with unswizzled scales must be rowwise-only."
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
    dbias = None
    dequantized = None
    if with_columnwise or with_dbias or with_dequantized:
        dequantized = tex.group_dequantize(grouped_x, TE_DType[dtype]).rowwise_data.view(
            grouped_x.logical_shape
        )
    if with_columnwise:
        colwise_quantizer = quantizer.copy()
        colwise_quantizer.set_usage(rowwise=False, columnwise=True)
        colwise_quantizer.optimize_for_gemm = True
        colwise_quantizer.internal = True
        if with_dbias:
            colwise_x, dbias = tex.bgrad_group_quantize(
                dequantized,
                colwise_quantizer,
                num_groups,
                split_sizes,
                tensor_offsets=tensor_offsets,
            )
        else:
            colwise_x = tex.group_quantize(
                dequantized,
                colwise_quantizer,
                num_groups,
                split_sizes,
                tensor_offsets=tensor_offsets,
            )
        grouped_x.columnwise_data = colwise_x.columnwise_data
        grouped_x.columnwise_scale_inv = colwise_x.columnwise_scale_inv
    elif with_dbias:
        # No columnwise stage to accumulate into (e.g. frozen weights need no
        # wgrad), so reduce the dequantized grad directly.
        dbias = compute_grouped_dbias(
            dequantized, tex.splits_to_offsets(split_sizes, 1), num_groups
        )

    # Convert rowwise scales to the GEMM-swizzled layout. The grouped GEMM
    # reads activation scales as one (total_tokens, cols) matrix, so the
    # single-tensor swizzle applies. Swizzling allocates a new scale buffer;
    # the original unswizzled scales are left untouched.
    # 128-alignment (see docstring) means the scale array needs no padding.
    total_tokens, cols = grouped_x.logical_shape
    if total_tokens % 128 != 0 or cols % 128 != 0:
        raise ValueError(
            "Pre-quantized MXFP8 grouped input requires dims that are multiples of 128, "
            f"but got ({total_tokens}, {cols})."
        )
    scale_shape = (total_tokens, cols // MXFP8_BLOCK_SCALING_SIZE)
    if grouped_x.scale_inv.numel() != math.prod(scale_shape):
        raise ValueError(
            f"Pre-quantized MXFP8 grouped input has {grouped_x.scale_inv.numel()} rowwise "
            f"scales, but expected {math.prod(scale_shape)} for shape {scale_shape}."
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

    return dbias, dequantized


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
