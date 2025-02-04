# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

import os
from typing import Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from functools import reduce
from operator import mul as multiply_op

import torch

from .. import cpp_extensions as tex
from ..constants import TE_DType
from ..utils import get_default_init_method
from ..tensor.float8_tensor import Float8Tensor
from ..tensor.mxfp8_tensor import MXFP8Quantizer

_use_cudnn_mxfp8_norm = bool(int(os.getenv("NVTE_CUDNN_MXFP8_NORM", "0")))


def _get_normalization_func(normalization: str, forward: bool):
    fwd_normalization_funcs = {
        "LayerNorm": tex.layernorm_fwd,
        "RMSNorm": tex.rmsnorm_fwd,
    }
    bwd_normalization_funcs = {
        "LayerNorm": tex.layernorm_bwd,
        "RMSNorm": tex.rmsnorm_bwd,
    }

    if forward:
        return fwd_normalization_funcs[normalization]
    return bwd_normalization_funcs[normalization]


def _fix_gathered_fp8_transpose(fp8_tensor: Float8Tensor, tp_size: int) -> Float8Tensor:
    """Reorder FP8 transposes after Userbuffers gather.

    The all-gather is performed in-place in the Float8Tensor's
    row-wise data, and afterwards we need to do a transpose to get the
    correct ordering. This misuses data fields in Float8Tensor and
    should be considered an evil hack. It would be best to move
    transpose logic into CommOverlap::get_buffer.

    Responsibility for fixing: adener, tmoon

    """
    assert isinstance(fp8_tensor, Float8Tensor), "Tensor is not a Float8Tensor"
    assert tp_size > 1, "The tensor transpose cannot be interleaved when TP size is 1"
    assert fp8_tensor._data is not None, "The tensor does not hold any rowwise data"
    assert (
        fp8_tensor._data.shape[0] % tp_size == 0
    ), "Leading dimension of data is not divisble by TP size"

    data = fp8_tensor._data
    batched_size = reduce(multiply_op, data.shape[1:])
    interleaved_shape = [tp_size, data.shape[0] // tp_size, batched_size]
    transposed_shape = [data.shape[0] // tp_size, batched_size * tp_size]
    fp8_tensor._transpose = (
        data.view(interleaved_shape).transpose(0, 1).contiguous().view(transposed_shape)
    )

    fp8_tensor._transpose_invalid = False
    fp8_tensor._data = None

    return fp8_tensor


def apply_normalization(
    inputmat: torch.Tensor,
    ln_out: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    eps: float,
    output_quantizer,
    output_dtype,
    normalization: str,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
):
    """Apply normalization to input."""
    normalization_func = _get_normalization_func(normalization, True)

    inputs = (inputmat, ln_weight) if ln_bias is None else (inputmat, ln_weight, ln_bias)

    split_mxfp8_cast = False
    if not _use_cudnn_mxfp8_norm and isinstance(output_quantizer, MXFP8Quantizer):
        split_mxfp8_cast = True

    output = normalization_func(
        *inputs,
        eps,
        None if split_mxfp8_cast else ln_out,
        None if split_mxfp8_cast else output_quantizer,
        TE_DType[output_dtype] if output_dtype in TE_DType else output_dtype,
        fwd_ln_sm_margin,
        zero_centered_gamma,
    )

    return (
        (output_quantizer.quantize(output[0], out=ln_out), *output[1:])
        if split_mxfp8_cast
        else output
    )


class _NoopCatFunc(torch.autograd.Function):
    """Concatenate tensors, doing a no-op if possible

    See _noop_cat.

    """

    @staticmethod
    def forward(
        ctx: Any,
        dim: int,
        *tensors: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Check first tensor
        if not tensors:
            raise ValueError("Attempted to concatenate 0 tensors")
        num_dims = tensors[0].dim()
        if not -num_dims <= dim < num_dims:
            raise ValueError(
                "Attempted to concatenate tensor "
                f"with shape {list(tensors[0].size())} along dim {dim}"
            )
        dim %= num_dims

        # Check remaining tensors
        out_shape = list(tensors[0].size())
        split_ranges = [(0, tensors[0].size(dim))]
        for tensor in tensors[1:]:
            in_shape = list(tensor.size())
            if (
                len(in_shape) != num_dims
                or in_shape[:dim] != out_shape[:dim]
                or in_shape[dim + 1 :] != out_shape[dim + 1 :]
            ):
                raise ValueError(
                    "Attempted to concatenate tensors with shapes "
                    f"{[list(tensor.size()) for tensor in tensors]} "
                    f"along dim {dim}"
                )
            split_start = out_shape[dim]
            split_end = split_start + in_shape[dim]
            out_shape[dim] = split_end
            split_ranges.append((split_start, split_end))

        # Save state for backward
        ctx.dim = dim
        ctx.split_ranges = split_ranges

        # Out-of-place concatenation if needed
        dtype = tensors[0].dtype
        device = tensors[0].device
        strides = tensors[0].stride()
        data_ptr_stride = strides[dim] * tensors[0].element_size()
        data_ptr = tensors[0].data_ptr() + tensors[0].size(dim) * data_ptr_stride
        for tensor in tensors[1:]:
            if (
                tensor.dtype != dtype
                or tensor.device != device
                or tensor.stride() != strides
                or tensor.data_ptr() != data_ptr
            ):
                return torch.cat(tensors, dim=dim)
            data_ptr += tensor.size(dim) * data_ptr_stride

        # No-op concatenation
        out = tensors[0].new()
        out.set_(
            tensors[0].untyped_storage(),
            tensors[0].storage_offset(),
            out_shape,
            strides,
        )
        out.requires_grad = any(tensor.requires_grad for tensor in tensors)
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        grad_inputs = []
        for split_start, split_end in ctx.split_ranges:
            slices = [slice(None)] * grad_output.dim()
            slices[ctx.dim] = slice(split_start, split_end)
            grad_inputs.append(grad_output[tuple(slices)])
        return None, *grad_inputs


def noop_cat(
    tensors: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate tensors, doing a no-op if possible

    If tensors are already concatenated in memory, a tensor view of
    that memory region will be returned. Otherwise the tensors will be
    concatenated out-of-place, as usual.

    """
    if not tensors:
        raise ValueError("Attempted to concatenate 0 tensors")
    if len(tensors) == 1:
        return tensors[0]
    return _NoopCatFunc.apply(dim, *tensors)


@dataclass
class _ParameterInitMeta:
    """
    Stores essential metadata needed to support deferred parameter initialization.
    """

    init_fn: Optional[Callable] = get_default_init_method()
    get_rng_state_tracker: Optional[Callable] = None
    fp8_meta_index: Optional[int] = None

    def __post_init__(self):
        """Safeguard reference to the parameter's parent module and initialization function."""
        if self.init_fn is None:
            self.init_fn = get_default_init_method()
