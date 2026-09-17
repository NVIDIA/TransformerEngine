# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

import dataclasses
import queue
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from .. import cpp_extensions as tex
from ..constants import TE_DType
from ..distributed import in_fp8_activation_recompute_phase
from ..dynamo import TensorSpec, is_value_opaque_quantizer
from ..export import is_in_onnx_export_mode
from ..quantization import FP8GlobalStateManager
from ..quantized_tensor import Quantizer
from ..tensor.hybrid_tensor import HybridQuantizer
from ..tensor.nvfp4_tensor import NVFP4Quantizer
from ..utils import get_default_init_method, get_device_compute_capability


def compile_unsupported_quantizer_reason(
    quantizers: Sequence[Optional[Quantizer]],
) -> Optional[str]:
    """Return a fallback reason if a quantizer cannot cross the custom-op boundary."""
    for quantizer in quantizers:
        # Delayed-scaling and unregistered custom-recipe quantizers are not value-opaque.
        if quantizer is not None and not is_value_opaque_quantizer(quantizer):
            return "a quantizer not registered as a torch.compile value-opaque type"
    return None


def set_quantizer_amax_reduction_group(quantizer, amax_reduction_group) -> None:
    """Set the amax reduction group on a quantizer; no-op if it doesn't support it.

    Unwraps ``DebugQuantizer`` to its ``parent_quantizer``, which is the one that
    actually performs the quantization (and thus the amax reduction).
    """
    if quantizer is None:
        return
    # DebugQuantizer delegates quantization to parent_quantizer
    target = getattr(quantizer, "parent_quantizer", quantizer)
    if target is not None and hasattr(target, "with_amax_reduction"):
        target.with_amax_reduction = amax_reduction_group is not None
        target.amax_reduction_group = amax_reduction_group


def set_quantizer_usage_for_wgrad_all_gather(quantizer) -> None:
    """Configure an all-gather output for consumption by wgrad."""
    if quantizer is None:
        return

    parent_quantizer = getattr(quantizer, "parent_quantizer", None)
    target = parent_quantizer if parent_quantizer is not None else quantizer

    # Hybrid currently gathers in high precision, then quantizes the full
    # result, so request the columnwise representation consumed by wgrad.
    if isinstance(target, HybridQuantizer):
        rowwise_usage, columnwise_usage = False, True
    elif quantizer.supports_only_rowwise_all_gather():
        # Per-tensor FP8 gathers rowwise data and synthesizes its transpose.
        rowwise_usage, columnwise_usage = True, False
    else:
        rowwise_usage, columnwise_usage = False, True

    # Preserve wrapper-specific bookkeeping. In particular, DebugQuantizer
    # propagates usage to its parent while keeping its own state synchronized.
    quantizer.set_usage(rowwise=rowwise_usage, columnwise=columnwise_usage)


def can_reconstruct_wgrad_input_from_original(quantizer) -> bool:
    """Whether wgrad input can be reconstructed from a saved original tensor."""
    target = getattr(quantizer, "parent_quantizer", quantizer)
    if target is None:
        target = quantizer
    if isinstance(target, HybridQuantizer):
        if target.columnwise_source == "original":
            return True
        return target.rowwise_quantizer.is_requantization_safe()
    return target.is_requantization_safe()


def update_normalization_output_spec(spec: TensorSpec) -> None:
    """Match the scale layout emitted by the normalization kernel."""
    quantizer = spec.quantizer
    if not isinstance(quantizer, NVFP4Quantizer) or not quantizer.optimize_for_gemm:
        return
    # Normalization does not run the standalone quantizer's post-quantize swizzle.
    rows, cols = spec.shape
    if not (10, 0) <= get_device_compute_capability() <= (11, 0):
        spec.with_gemm_swizzled_scales = False
    elif quantizer.with_rht:
        spec.with_gemm_swizzled_scales = bool(rows % 64 == 0 and cols % 128 == 0)
    else:
        spec.with_gemm_swizzled_scales = bool(
            quantizer.with_2d_quantization
            and not quantizer.row_scaled_nvfp4
            and not quantizer.nvfp4_use_4over6
            and rows % 128 == 0
            and cols % 128 == 0
        )


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

        # Check concat dim
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

        # Tensor properties from first tensor
        dtype = tensors[0].dtype
        device = tensors[0].device
        strides = tensors[0].stride()
        data_ptr_stride = strides[dim] * tensors[0].element_size()

        # Out-of-place concatenation when view tensors have different storage
        # Note: This works around an edge case with the split_quantize
        # function, which might allocate a buffer and construct
        # subviews. However, in order to reduce CPU overheads, these
        # views are configured manually outside of PyTorch. PyTorch
        # doesn't know these views share the same memory, and it
        # blocks us from reconstructing the full tensor because it
        # thinks we are accessing out-of-bounds memory.
        if tensors[0].untyped_storage().nbytes() < out_shape[dim] * data_ptr_stride:
            return torch.cat(tensors, dim=dim)

        # Out-of-place concatenation if tensor properties do not match
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
        out = tensors[0].as_strided(out_shape, strides)
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
    if is_in_onnx_export_mode():
        return torch.cat(tensors, dim=dim)
    return _NoopCatFunc.apply(dim, *tensors)


@dataclasses.dataclass
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


def check_fp8_reduce_and_update(restore_first_module: bool = False) -> bool:
    """Whether this module's backward should reduce and update the FP8 scaling factors.

    Consumes the "first FP8 module" flag, restored when the forward is a
    recomputation so the flag survives for the real forward's owner.
    """
    qstate = FP8GlobalStateManager.quantization_state
    first_fp8_module = qstate.is_first_fp8_module
    result = FP8GlobalStateManager.is_first_fp8_module()
    if restore_first_module or in_fp8_activation_recompute_phase():
        qstate.is_first_fp8_module = first_fp8_module
    return result


def get_output_first_dim_size(input_first_dim_size: int, args: Any) -> int:
    """Compute the output's first dimension size from the input's.

    Sequence parallelism gathers this dimension in column-parallel mode and
    scatters it in row-parallel mode. ``args`` provides ``sequence_parallel``,
    ``parallel_mode``, and ``tp_size``.
    """
    if not args.sequence_parallel:
        return input_first_dim_size
    if args.parallel_mode == "column":
        return input_first_dim_size * args.tp_size
    if args.parallel_mode == "row":
        return input_first_dim_size // args.tp_size
    return input_first_dim_size


def get_input_first_dim_size(output_first_dim_size: int, args: Any) -> int:
    """Recover the input's first dimension size from the output's.

    Inverts the sequence-parallel gather/scatter in
    :func:`get_output_first_dim_size`, including when called with a gradient's shape.
    """
    if not args.sequence_parallel:
        return output_first_dim_size
    if args.parallel_mode == "column":
        return output_first_dim_size // args.tp_size
    if args.parallel_mode == "row":
        return output_first_dim_size * args.tp_size
    return output_first_dim_size


def fake_workspace_valid(workspace: TensorSpec, quantizer: Optional[Quantizer]) -> bool:
    """Spec-level mirror of ``_is_weight_workspace_valid``: the cached workspace
    must already hold every inner buffer the quantizer's current usage needs."""
    if quantizer is None:
        return True
    required = TensorSpec(
        shape=workspace.shape, dtype=workspace.dtype, quantizer=quantizer, device=workspace.device
    ).inner_names()
    return set(required) <= set(workspace.inner_names())
