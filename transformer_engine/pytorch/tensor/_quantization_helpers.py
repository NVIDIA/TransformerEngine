# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Private helper functions and classes for quantized tensor implementations.

This module contains internal autograd functions and utilities that support
the quantization machinery.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, Any, Dict, Iterable, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


def _resolve_view_shape(input_shape: Iterable[int], shape: Iterable[int]) -> torch.Size:
    """Resolve a requested view shape with PyTorch-compatible semantics.

    The concrete-integer path avoids constructing a temporary meta tensor. If
    either shape contains symbolic dimensions, retain the previous meta-tensor
    path so that PyTorch remains responsible for symbolic shape handling.
    """
    input_shape = tuple(input_shape)
    shape = tuple(shape)
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
        shape = tuple(shape[0])

    # Avoid comparisons that specialize or guard SymInts. The meta fallback is
    # also useful for preserving PyTorch's type checking of non-integer dims.
    if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in (*input_shape, *shape)):
        return torch.empty(input_shape, device="meta").view(shape).shape

    input_numel = 1
    for dim in input_shape:
        input_numel *= dim

    inferred_dim = None
    known_numel = 1
    for index, dim in enumerate(shape):
        if dim == -1:
            if inferred_dim is not None:
                raise RuntimeError("only one dimension can be inferred")
            inferred_dim = index
        elif dim < 0:
            raise RuntimeError(
                f"invalid shape dimension {dim} at index {index} of shape {list(shape)}"
            )
        else:
            known_numel *= dim

    resolved_shape = list(shape)
    if inferred_dim is not None:
        if known_numel == 0:
            if input_numel == 0:
                raise RuntimeError(
                    f"cannot reshape tensor of 0 elements into shape {list(shape)} because "
                    "the unspecified dimension size -1 can be any value and is ambiguous"
                )
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {input_numel}")
        if input_numel % known_numel != 0:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {input_numel}")
        resolved_shape[inferred_dim] = input_numel // known_numel
    elif known_numel != input_numel:
        raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {input_numel}")

    return torch.Size(resolved_shape)


class _QuantizeFunc(torch.autograd.Function):
    """Quantize tensor"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: torch.Tensor,
        quantize_impl: Callable,
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring
        return quantize_impl(tensor)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class _IdentityFunc(torch.autograd.Function):
    """Identity function

    If constructor keyword-arguments are provided, then construct a
    new Float8Tensor using the provided tensor's attributes.

    """

    @staticmethod
    def forward(
        ctx, tensor: QuantizedTensor, init_kwargs: Optional[Dict[str, Any]] = None
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if constructor kwargs are not provided
        if init_kwargs is None:
            return tensor.detach()

        # Construct new tensor if constructor kwargs are provided
        ctx.input_dtype = tensor.dtype
        kwargs = tensor.get_metadata()
        for key, val in init_kwargs.items():
            kwargs[key] = val
        kwargs["device"] = tensor.device
        return type(tensor)(tensor.shape, tensor.dtype, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        grad_input = grad_output
        if grad_input.dtype == ctx.input_dtype:
            grad_input = grad_input.detach()
        else:
            grad_input = grad_input.to(ctx.input_dtype)
        return grad_input, None


def _stride_from_shape(shape: list[int]):
    """Calculate stride from shape for contiguous tensors"""
    if len(shape) == 0:
        return []
    rstride = [1]
    for d in reversed(shape[1:]):
        rstride.append(rstride[-1] * d)
    return list(reversed(rstride))


def safe_quantized_repr(obj, cls_name, extras=None, error=None):
    """Metadata-only repr fallback for quantized tensors whose data cannot be
    materialized for any reason.

    Each attribute access is guarded so that ``__repr__`` never raises.

    Parameters
    ----------
    extras : dict, optional
        Additional plain-Python (non-tensor) attributes to include, e.g.
        ``{"is_2D_scaled": self._is_2D_scaled}``. Values are inserted after
        ``fp8_dtype`` and before ``shape``.
    error : BaseException, optional
        The exception that triggered the fallback. When given, its type and
        message are included in the ``data=`` field so that it is visible *why*
        the data could not be materialized.
    """
    parts = []
    fp8_dtype = getattr(obj, "_fp8_dtype", None)
    if fp8_dtype is not None:
        parts.append(f"fp8_dtype={fp8_dtype}")
    if extras:
        for key, value in extras.items():
            parts.append(f"{key}={value}")
    try:
        parts.append(f"shape={tuple(obj.shape)}")
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        parts.append(f"dtype={obj.dtype}")
    except Exception:  # pylint: disable=broad-except
        pass
    if error is not None:
        parts.append(f"data=<unmaterialized: {type(error).__name__}: {error}>")
    else:
        parts.append("data=<unmaterialized>")
    return f"{cls_name}({', '.join(parts)})"
