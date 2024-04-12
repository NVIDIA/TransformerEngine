# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass

import torch

from .. import cpp_extensions as tex
from ..fp8 import get_fp8_te_dtype
from ..utils import get_default_init_method

def _get_normalization_func(normalization: str,
                            fp8_output: bool,
                            is_grad_enabled: bool,
                            forward: bool):
    fwd_normalization_funcs = {
            ('LayerNorm', True, True):   tex.layernorm_fwd_fp8,
            ('LayerNorm', True, False):  tex.layernorm_fwd_fp8_inf,
            ('LayerNorm', False, True):  tex.layernorm_fwd_noalloc,
            ('LayerNorm', False, False): tex.layernorm_fwd_inf,
            ('RMSNorm', True, True):     tex.rmsnorm_fwd_fp8,
            ('RMSNorm', True, False):    tex.rmsnorm_fwd_fp8_inf,
            ('RMSNorm', False, True):    tex.rmsnorm_fwd_noalloc,
            ('RMSNorm', False, False):   tex.rmsnorm_fwd_inf,
    }
    bwd_normalization_funcs = {
            'LayerNorm':  tex.layernorm_bwd,
            'RMSNorm':    tex.rmsnorm_bwd,
    }

    if forward:
        return fwd_normalization_funcs[(normalization, fp8_output, is_grad_enabled)]
    assert not fp8_output, "FP8 output is not supported in backward normalization!"
    assert is_grad_enabled, "Gradient has to be enabled to call backward normalization!"
    return bwd_normalization_funcs[normalization]

def _apply_normalization(inputmat:torch.Tensor,
                         ln_out: torch.Tensor,
                         ln_weight: torch.Tensor,
                         ln_bias: Union[torch.Tensor, None],
                         eps: float,
                         fp8_out: bool,
                         fp8_meta: Dict[str, Any],
                         normalization: str,
                         fwd_ln_sm_margin: int,
                         zero_centered_gamma: bool,
                         is_grad_enabled: bool):
    normalization_func = _get_normalization_func(normalization,
                                                 fp8_out,
                                                 is_grad_enabled,
                                                 True)

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
            return normalization_func(
                *inputs,
                eps,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                fwd_ln_sm_margin,
                zero_centered_gamma,
            ), None, None
    else:
        if is_grad_enabled:
            output = normalization_func(
                *inputs, ln_out, eps,
                fwd_ln_sm_margin, zero_centered_gamma
            )
        else:
            return normalization_func(
                    *inputs, eps, fwd_ln_sm_margin, zero_centered_gamma
            ), None, None
    if normalization == "RMSNorm":
        output = (ln_out, None, output[1])
    elif normalization == "LayerNorm":
        output = (ln_out, output[1], output[2])
    return output


class _NoopCatFunc(torch.autograd.Function):
    """No-op concatenate tensors along dim 0

    `full_tensor` is assumed to already be the concatenation of
    `tensors`, i.e. they occupy the same memory with the correct
    offsets.

    """

    @staticmethod
    def forward(
        ctx,
        split_ranges: List[Tuple[int, int]],
        full_tensor: torch.Tensor,
        *tensors: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # pylint: disable=unused-argument
        ctx.split_ranges = split_ranges
        assert not full_tensor.requires_grad, "Concatenated tensor should not require gradient"
        out = full_tensor.new()
        out.set_(
            full_tensor.untyped_storage(),
            full_tensor.storage_offset(),
            full_tensor.size(),
            full_tensor.stride(),
        )
        out.requires_grad = True
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        grads = [
            grad_output[split_start:split_end]
            for split_start, split_end in ctx.split_ranges
        ]
        return None, None, *grads


def _noop_cat(
    tensors: List[torch.Tensor],
    full_tensor: torch.Tensor,
) -> torch.Tensor:
    """Concatenate tensors along dim 0, doing a no-op if possible

    If `full_tensor` is already the concatenation of `tensors`, i.e.
    they occupy the same memory region with the correct offsets, then
    no copies are performed. Otherwise the buffers in all the tensors
    are reallocated so that another call would result in a no-op.

    In the backward pass, gradients to `partial_tensors` will just be
    tensor views.

    """

    # Determine split points
    split_ranges = []
    full_tensor_shape = full_tensor.size()
    offset = 0
    for tensor in tensors:
        tensor_shape = tensor.size()
        if tensor_shape[1:] != full_tensor_shape[1:]:
            raise ValueError(
                f"Attempting to concatenate tensor with shape={list(tensor_shape)} "
                f"into a tensor with shape={list(full_tensor_shape)}"
            )
        split_start = offset
        offset += tensor_shape[0]
        split_end = offset
        split_ranges.append((split_start, split_end))
    if offset != full_tensor_shape[0]:
        raise ValueError(
            f"Attempting to concatenate tensors with total shape[0]={offset} "
            f"into a tensor with shape[0]={full_tensor_shape[0]}"
        )

    # Reallocate buffers if no-op concat isn't possible
    need_to_reallocate = False
    for tensor, (split_start, _) in zip(tensors, split_ranges):
        if tensor.data_ptr() != full_tensor[split_start].data_ptr():
            need_to_reallocate = True
            break
    if need_to_reallocate:
        with torch.no_grad():
            full_tensor.data = torch.cat(tensors)
            for tensor, (split_start, split_end) in zip(tensors, split_ranges):
                tensor.data = full_tensor[split_start:split_end]

    # Perform no-op concat
    return _NoopCatFunc.apply(split_ranges, full_tensor, *tensors)


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
