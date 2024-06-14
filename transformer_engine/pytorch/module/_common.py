# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass

import torch

from .. import cpp_extensions as tex
from ..export import is_in_onnx_export_mode
from ..fp8 import get_fp8_te_dtype
from ..utils import get_default_init_method


def _get_normalization_func(
    normalization: str, fp8_output: bool, is_grad_enabled: bool, forward: bool
):
    fwd_normalization_funcs = {
        ("LayerNorm", True, True): tex.layernorm_fwd_fp8,
        ("LayerNorm", True, False): tex.layernorm_fwd_fp8_inf,
        ("LayerNorm", False, True): tex.layernorm_fwd_noalloc,
        ("LayerNorm", False, False): tex.layernorm_fwd_inf,
        ("RMSNorm", True, True): tex.rmsnorm_fwd_fp8,
        ("RMSNorm", True, False): tex.rmsnorm_fwd_fp8_inf,
        ("RMSNorm", False, True): tex.rmsnorm_fwd_noalloc,
        ("RMSNorm", False, False): tex.rmsnorm_fwd_inf,
    }
    bwd_normalization_funcs = {
        "LayerNorm": tex.layernorm_bwd,
        "RMSNorm": tex.rmsnorm_bwd,
    }

    if forward:
        return fwd_normalization_funcs[(normalization, fp8_output, is_grad_enabled)]
    assert not fp8_output, "FP8 output is not supported in backward normalization!"
    assert is_grad_enabled, "Gradient has to be enabled to call backward normalization!"
    return bwd_normalization_funcs[normalization]


def _apply_normalization(
    inputmat: torch.Tensor,
    ln_out: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    eps: float,
    fp8_out: bool,
    fp8_meta: Dict[str, Any],
    normalization: str,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
    is_grad_enabled: bool,
):
    normalization_func = _get_normalization_func(normalization, fp8_out, is_grad_enabled, True)

    inputs = (inputmat, ln_weight) if ln_bias is None else (inputmat, ln_weight, ln_bias)
    if fp8_out:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        if is_grad_enabled:
            output_key = "ln_out" if normalization == "LayerNorm" else "rmsnorm_out"
            output_kwarg = {output_key: ln_out}
            output = normalization_func(
                *inputs,
                eps,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                fwd_ln_sm_margin,
                zero_centered_gamma,
                **output_kwarg,
            )
        else:
            return (
                normalization_func(
                    *inputs,
                    eps,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                    fwd_ln_sm_margin,
                    zero_centered_gamma,
                ),
                None,
                None,
            )
    else:
        if is_grad_enabled:
            output = normalization_func(*inputs, ln_out, eps, fwd_ln_sm_margin, zero_centered_gamma)
        else:
            return (
                normalization_func(*inputs, eps, fwd_ln_sm_margin, zero_centered_gamma),
                None,
                None,
            )
    if normalization == "RMSNorm":
        output = (ln_out, None, output[1])
    elif normalization == "LayerNorm":
        output = (ln_out, output[1], output[2])
    return output


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
        grad_inputs = []
        for split_start, split_end in ctx.split_ranges:
            slices = [slice(None)] * grad_output.dim()
            slices[ctx.dim] = slice(split_start, split_end)
            grad_inputs.append(grad_output[tuple(slices)])
        return None, *grad_inputs


def _noop_cat(
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
    if is_in_onnx_export_mode():
        return torch.cat(tensors, dim=dim)
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
