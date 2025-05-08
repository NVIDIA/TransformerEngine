# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

from typing import Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass

import queue
import torch

from .. import cpp_extensions as tex
from ..constants import TE_DType
from ..utils import get_default_init_method


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

    return normalization_func(
        *inputs,
        eps,
        ln_out,
        output_quantizer,
        TE_DType[output_dtype] if output_dtype in TE_DType else output_dtype,
        fwd_ln_sm_margin,
        zero_centered_gamma,
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


class WeightGradStore:
    """
    A class to manage weight gradient storage and computation in Transformer modules.
    This class enables split backward propagation for better memory efficiency.
    """

    def __init__(self, delay_wgrad_compute=False, ub_bulk_wgrad=False):
        """
        Initialize the WeightGradStore.

        Args:
            delay_wgrad_compute (bool): Whether to delay weight gradient computation
            ub_bulk_wgrad (bool): Whether to enable bulk weight gradient computation
        """
        if delay_wgrad_compute:
            self.context = queue.Queue()
            assert (
                ub_bulk_wgrad is False
            ), "ub_bulk_wgrad is not supported when enabling delay_wgrad_compute"
            self.enabled = delay_wgrad_compute
        else:
            self.context = None
            self.enabled = False

    def delay_wgrad_compute(self):
        """
        Get the current split backward propagation status.

        Returns:
            bool: True if split backward is enabled, False otherwise
        """
        return self.enabled

    def enable_delay_wgrad_compute(self):
        """Enable split backward propagation."""
        self.enabled = True

    def disable_delay_wgrad_compute(self):
        """Disable split backward propagation."""
        self.enabled = False

    def put(self, tensor_list, func):
        """
        Store tensors and computation function for later execution.

        Args:
            tensor_list (list): List of tensors needed for computation
            func (callable): Function to be executed with the tensors
        """
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        self.context.put([tensor_list, func])

    def pop(self):
        """
        Execute the stored computation with the stored tensors.
        Raises an exception if the queue is empty.
        """
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        if self.context.qsize() > 0:
            tensor_list, func = self.context.get()
            return func(*tensor_list), tensor_list
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            raise RuntimeError(f"Pop empty queue. rank {rank}")
        raise RuntimeError("Pop empty queue. No distributed environment detected.")

    def assert_empty(self):
        """
        Assert that the queue is empty.
        Used for debugging and ensuring proper cleanup.
        """
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        rank = torch.distributed.get_rank()
        assert self.context.empty(), f"Queue is not empty. rank {rank}"
