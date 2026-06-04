# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
from collections.abc import Iterable
import functools
import math
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
from typing import Optional

import torch
from packaging.version import Version as PkgVersion

import transformer_engine_torch as tex
from transformer_engine_torch import FP8TensorMeta
from ..torch_version import torch_version
from ..quantization import FP8GlobalStateManager
from ..tensor import NVFP4Quantizer, NVFP4Tensor, NVFP4TensorStorage, Quantizer
from ..tensor.float8_tensor import Float8Tensor
from ..tensor.grouped_tensor import GroupedTensor
from ..quantized_tensor import QuantizedTensorStorage
from ..utils import canonicalize_dtype


@functools.lru_cache(maxsize=None)
def _cudnn_frontend_version_at_least(min_version: str) -> bool:
    """Check cuDNN frontend package version."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion(min_version)
    except PackageNotFoundError:
        return False


def _cudnn_frontend_version_supported() -> bool:
    """Check cuDNN frontend is at least 1.23.0.

    All grouped MLP fused-kernel features require cuDNN frontend >= 1.23.0.
    """
    return _cudnn_frontend_version_at_least("1.23.0")


def _cudnn_frontend_geglu_runtime_params() -> bool:
    """Check cuDNN frontend is at least 1.24.0.

    Runtime-configurable GeGLU parameters (linear_offset, geglu_alpha,
    glu_clamp_max, glu_clamp_min) require cuDNN frontend >= 1.24.0.
    """
    return _cudnn_frontend_version_at_least("1.24.0")


def _cudnn_frontend_supports_grouped_gemm_srelu() -> bool:
    """Check cuDNN frontend min version for grouped GEMM SReLU kernels."""
    return _cudnn_frontend_version_at_least("1.24.0")


def _nvidia_cudnn_frontend_supports_wgrad() -> bool:
    """Check cuDNN FE min version for grouped GEMM wgrad kernel."""
    return _cudnn_frontend_version_supported()


def _group_quantize_for_grouped_mlp(
    tensor: torch.Tensor,
    quantizer: Quantizer,
    num_groups: int,
    split_sizes: Optional[torch.Tensor],
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> GroupedTensor:
    """Quantize into grouped storage."""

    # Typical case: group-quantize
    if num_groups != 1 or not isinstance(quantizer, NVFP4Quantizer):
        return tex.group_quantize(
            tensor,
            quantizer,
            num_groups,
            split_sizes,
            tensor_offsets=tensor_offsets,
        )

    # --------------------------------------------------
    # Special case: single-tensor NVFP4 quantize
    # --------------------------------------------------

    quantized = tex.quantize(tensor, quantizer)
    with_gemm_swizzled_scales = quantized._with_gemm_swizzled_scales
    if quantizer.optimize_for_gemm:
        tex.swizzle_scales_for_gemm_(quantized)
        with_gemm_swizzled_scales = True

    rowwise_data = quantized._rowwise_data
    rowwise_scale = quantized._rowwise_scale_inv
    columnwise_data = quantized._columnwise_data
    columnwise_scale = quantized._columnwise_scale_inv
    amax = quantized._amax_rowwise
    columnwise_amax = quantized._amax_columnwise

    if split_sizes is None:
        split_sizes = torch.full((1,), tensor.shape[0], dtype=torch.int64, device=tensor.device)
    else:
        split_sizes = split_sizes.to(dtype=torch.int64, device=tensor.device)

    m_dim = tensor.shape[0]
    if rowwise_data is not None:
        k_dim = rowwise_data.shape[-1] * 2
    elif columnwise_data is not None:
        k_dim = columnwise_data.shape[0]
    else:
        k_dim = tensor.shape[-1]

    if tensor_offsets is None:
        tensor_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=tensor.device),
                torch.cumsum(split_sizes * k_dim, dim=0),
            ],
        )

    return GroupedTensor(
        shape=(m_dim, k_dim),
        dtype=tensor.dtype,
        quantizer=quantizer,
        num_tensors=1,
        data=rowwise_data.reshape(-1) if rowwise_data is not None else None,
        columnwise_data=columnwise_data.reshape(-1) if columnwise_data is not None else None,
        scale_inv=rowwise_scale.reshape(-1) if rowwise_scale is not None else None,
        columnwise_scale_inv=columnwise_scale.reshape(-1) if columnwise_scale is not None else None,
        amax=amax,
        columnwise_amax=columnwise_amax,
        first_dims=split_sizes,
        tensor_offsets=tensor_offsets,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )


def _nvfp4_amax(
    tensors: GroupedTensor | Iterable[NVFP4TensorStorage],
    *,
    columnwise: bool,
) -> torch.Tensor:
    """Get one NVFP4 amax value per group."""
    grouped_attr = "columnwise_amax" if columnwise else "amax"
    tensor_attr = "_amax_columnwise" if columnwise else "_amax_rowwise"

    if hasattr(tensors, grouped_attr):
        amax = getattr(tensors, grouped_attr)
        if amax is None:
            raise RuntimeError(f"NVFP4 GroupedTensor is missing {grouped_attr}.")
        return amax.view(-1)

    amaxes = [getattr(tensor, tensor_attr) for tensor in tensors]
    if any(amax is None for amax in amaxes):
        raise RuntimeError(f"NVFP4 tensor list is missing {tensor_attr}.")
    return torch.cat([amax.view(-1) for amax in amaxes], dim=0)


def _nvfp4_single_tensor_from_grouped(
    grouped: GroupedTensor,
    quantizer: Optional[NVFP4Quantizer] = None,
    *,
    fp4_dtype: Optional[torch.dtype] = None,
) -> NVFP4Tensor:
    """Build a single NVFP4Tensor view over a one-member grouped storage."""
    if quantizer is None:
        quantizer = grouped.quantizer
    if not isinstance(quantizer, NVFP4Quantizer):
        raise TypeError("Expected an NVFP4 GroupedTensor.")

    shape = tuple(grouped.logical_shape)
    rowwise_data = None
    if grouped.rowwise_data is not None:
        rowwise_data = grouped.rowwise_data.view(quantizer.convert_shape_for_fp4(shape))

    rowwise_scale_inv = None
    if grouped.scale_inv is not None:
        rowwise_scale_inv = grouped.scale_inv.view(quantizer.get_scale_shape(shape, False))

    columnwise_data = None
    if grouped.columnwise_data is not None:
        columnwise_shape = quantizer.get_columnwise_shape(shape)
        columnwise_data = grouped.columnwise_data.view(
            quantizer.convert_shape_for_fp4(columnwise_shape)
        )

    columnwise_scale_inv = None
    if grouped.columnwise_scale_inv is not None:
        columnwise_scale_inv = grouped.columnwise_scale_inv.view(
            quantizer.get_scale_shape(shape, True)
        )

    return NVFP4Tensor(
        shape=shape,
        dtype=grouped.get_dtype(),
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        amax_rowwise=grouped.amax,
        amax_columnwise=grouped.columnwise_amax,
        fp4_dtype=fp4_dtype or quantizer.dtype,
        quantizer=quantizer,
        requires_grad=False,
        with_gemm_swizzled_scales=grouped._with_gemm_swizzled_scales,
    )


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


def is_glu_activation(activation_op) -> bool:
    """Whether an activation consumes a GLU-style doubled input."""
    from .basic import (  # pylint: disable=import-outside-toplevel
        ScaledClampedQGeGLU,
        ScaledSwiGLU,
    )

    return isinstance(activation_op, (ScaledSwiGLU, ScaledClampedQGeGLU))


def validate_grouped_mlp_dims(fc1, activation_op, fc2) -> None:
    """Validate FC1 / activation / FC2 dimensions for fused grouped MLP."""
    from .basic import (  # pylint: disable=import-outside-toplevel
        ScaledSReLU,
    )

    if fc1.in_features % 64 != 0 or fc1.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC1 (num_groups={fc1.num_groups}, "
            f"in_features={fc1.in_features}, out_features={fc1.out_features})."
        )
    if fc2.in_features % 64 != 0 or fc2.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC2 (num_groups={fc2.num_groups}, "
            f"in_features={fc2.in_features}, out_features={fc2.out_features})."
        )
    if is_glu_activation(activation_op):
        expected_fc1_out_features = 2 * fc2.in_features
    elif isinstance(activation_op, ScaledSReLU):
        expected_fc1_out_features = fc2.in_features
    else:
        raise TypeError(f"Unsupported grouped MLP activation ({activation_op.__class__.__name__}).")

    if fc1.out_features != expected_fc1_out_features or fc1.num_groups != fc2.num_groups:
        raise ValueError(
            f"FC1 (num_groups={fc1.num_groups}, in_features={fc1.in_features}, "
            f"out_features={fc1.out_features}) "
            f"and FC2 (num_groups={fc2.num_groups}, in_features={fc2.in_features}, "
            f"out_features={fc2.out_features}) do not match."
        )
    if is_glu_activation(activation_op) and activation_op.glu_interleave_size != 32:
        raise ValueError(
            "Fused kernel requires 32-wide GLU interleaving, "
            f"but got glu_interleave_size={activation_op.glu_interleave_size}."
        )


def fuse_grouped_mlp_ops(
    ops,
    *,
    recipe,
    fused_op_cls,
    activation_op_types=None,
):
    """Sliding-window fusion for GroupedLinear + activation + GroupedLinear.

    Parameters
    ----------
    ops : list of FusibleOperation
        Operations to scan.
    recipe : Recipe or None
        Quantization recipe.
    fused_op_cls : type
        Fused operation class with ``is_supported()`` classmethod and
        constructor accepting ``fc1``, ``activation``, and ``fc2`` keyword args.

    Returns
    -------
    list of FusibleOperation
        Updated operations with matched triples replaced by fused ops.
    """
    from .basic import (  # pylint: disable=import-outside-toplevel
        GroupedLinear,
        ScaledClampedQGeGLU,
        ScaledSwiGLU,
    )

    if not fused_op_cls.is_supported():
        return ops
    if recipe is None or not (recipe.mxfp8() or recipe.nvfp4()):
        return ops
    # NVFP4 fused grouped MLP uses graph-safe grouped quantize, which currently requires RHT.
    if recipe.nvfp4() and recipe.disable_rht:
        return ops
    if activation_op_types is None:
        activation_op_types = (ScaledSwiGLU, ScaledClampedQGeGLU)

    out = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:

        matches_pattern = True
        if not (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], activation_op_types)
            and isinstance(window[2], GroupedLinear)
        ):
            matches_pattern = False
        elif (
            isinstance(window[1], ScaledClampedQGeGLU)
            and not _cudnn_frontend_geglu_runtime_params()
            and (
                abs(window[1]._clamped.alpha - 1.702) > 0.001
                or abs(window[1]._clamped.glu_linear_offset - 1.0) > 0.001
                or abs(window[1]._clamped.limit - 7.0) > 0.001
            )
        ):
            matches_pattern = False
        else:
            try:
                validate_grouped_mlp_dims(window[0], window[1], window[2])
            except (TypeError, ValueError):
                matches_pattern = False

        if matches_pattern:
            op = fused_op_cls(
                fc1=window[0],
                activation=window[1],
                fc2=window[2],
            )
            window = [op]
        else:
            out.extend(window[:-2])
            window = window[-2:]

        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    out.extend(window)
    return out
