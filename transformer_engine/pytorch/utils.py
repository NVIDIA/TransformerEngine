# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utility functions for Transformer Engine modules"""
from __future__ import annotations
import functools
import math
import os
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import torch

import transformer_engine.pytorch.cpp_extensions as ext
from . import torch_version
from ..debug.pytorch.debug_quantization import DebugQuantizedTensor


def requires_grad(*tensors: Tuple[Optional[torch.Tensor], ...]) -> None:
    """Check if any of the given tensors require gradient."""
    for tensor in tensors:
        if tensor is not None and tensor.requires_grad:
            return True
    return False


@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()


def clear_tensor_data(*tensors: Tuple[Optional[torch.Tensor], ...]) -> None:
    """
    Trick to deallocate tensor memory when delete operation does not
    release the tensor due to PyTorch override.

    Must be used carefully.
    """
    for t in tensors:
        if t is not None:
            if hasattr(t, "clear"):
                t.clear()
            else:
                t.data = _empty_tensor()
            del t


@functools.lru_cache
def _get_device_compute_capability(device: torch.device) -> Tuple[int, int]:
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def get_device_compute_capability() -> Tuple[int, int]:
    """CUDA compute capability of current GPU"""
    return _get_device_compute_capability(torch.cuda.current_device())


def attention_mask_func(
    attention_scores: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Get attention mask"""
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_default_init_method() -> Callable:
    """Weight initialization method if not provided by user"""
    return init_method_normal(0.023)


def init_method_constant(val: float) -> Callable:
    """Init method to set all tensor elements to a constant value."""
    if val == 1.0:

        def init_(tensor: torch.Tensor) -> Callable:
            return torch.nn.init.ones_(tensor)

    elif val == 0.0:

        def init_(tensor: torch.Tensor) -> Callable:
            return torch.nn.init.zeros_(tensor)

    else:

        def init_(tensor: torch.Tensor) -> Callable:
            return torch.nn.init.constant_(tensor, val)

    return init_


def init_method_normal(sigma: float) -> Callable:
    """Init method based on N(0, sigma)."""

    def init_(tensor: torch.Tensor) -> Callable:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma: float, num_layers: int) -> Callable:
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor: torch.Tensor) -> Callable:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def all_close(a: torch.Tensor, b: torch.Tensor) -> bool:
    """torch.allclose with cpu to not run into OOMs"""
    return torch.allclose(a.cpu(), b.cpu())


def print_rank_0(*args: Any) -> None:
    """print on rank 0"""
    if torch.cuda.current_device() == 0:
        print(*args)


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> None:
    """util function to show some tensor stats"""
    if a.shape != b.shape:
        print_rank_0("Tensors have different shape")
        return
    print_rank_0(a)
    print_rank_0(b)
    max_err = torch.max(torch.abs(a - b))
    max_a = torch.max(a)
    max_b = torch.max(b)
    print_rank_0(f"max err={max_err}, max a={max_a}, max_b={max_b}")


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_dim(
    tensor: torch.Tensor, dim: int, num_partitions: int, contiguous_split_chunks: bool = False
) -> Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    split_size = divide(tensor.size()[dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, split_size, dim=dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


# @klakhani TODO: Consider combining with split_tensor_along_dim() and no_op_cat() and SplitAlongDim
def combine_tensors(
    tensors: List[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Combine tensors along a particular dimension"""

    num_tensors = len(tensors)
    new_shape = list(tensors[0].shape)
    new_shape.insert(dim, num_tensors)
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    if isinstance(tensors[0], Float8Tensor):
        new_stride = list(tensors[0]._data.stride())
        new_stride.insert(dim, int(new_stride[dim - 1] / num_tensors))
        combined_tensor = torch.Tensor().to(device=tensors[0].device, dtype=tensors[0]._data.dtype)
        combined_tensor.set_(
            tensors[0]._data.untyped_storage(),
            tensors[0]._data.storage_offset(),
            new_shape,
            new_stride,
        )
        combined_tensor = Float8Tensor.make_like(tensors[0], data=combined_tensor, shape=new_shape)
    else:
        new_stride = list(tensors[0].stride())
        new_stride.insert(dim, int(new_stride[dim - 1] / num_tensors))
        combined_tensor = torch.Tensor().to(device=tensors[0].device, dtype=tensors[0].dtype)
        combined_tensor.set_(
            tensors[0].untyped_storage(), tensors[0].storage_offset(), new_shape, new_stride
        )

    return combined_tensor


class SplitAlongDim(torch.autograd.Function):
    """
    Split tensor along given dimension
    """

    @staticmethod
    def forward(
        ctx,
        mixed_x_layer: torch.Tensor,
        split_dim: int,
        split_size_or_sections: Union[int, List[int], Tuple[int]],
        squeeze=False,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        ctx.split_dim = split_dim
        ctx.split_size_or_sections = split_size_or_sections
        from transformer_engine.pytorch.float8_tensor import Float8Tensor
        from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase

        if isinstance(mixed_x_layer, Float8TensorBase) and not isinstance(
            mixed_x_layer, Float8Tensor
        ):
            return tuple(
                Float8TensorBase(
                    fp8_scale_inv=mixed_x_layer._scale_inv,
                    fp8_dtype=mixed_x_layer._fp8_dtype,
                    data=x.squeeze(split_dim) if squeeze else x,
                    shape=x.squeeze(split_dim).shape if squeeze else x.shape,
                    quantizer=mixed_x_layer._quantizer,
                )
                for x in torch.split(
                    mixed_x_layer._data,
                    split_size_or_sections=split_size_or_sections,
                    dim=split_dim,
                )
            )
        if isinstance(mixed_x_layer, Float8Tensor):
            return tuple(
                Float8Tensor.make_like(
                    mixed_x_layer,
                    data=x.squeeze(split_dim) if squeeze else x,
                    shape=x.squeeze(split_dim).shape if squeeze else x.shape,
                )
                for x in torch.split(
                    mixed_x_layer._data,
                    split_size_or_sections=split_size_or_sections,
                    dim=split_dim,
                )
            )
        out_list = torch.split(mixed_x_layer, split_size_or_sections, dim=split_dim)
        if squeeze:
            out_list = [x.squeeze(split_dim) for x in out_list]
        return out_list

    @staticmethod
    def backward(ctx, *grad_outputs):
        # pylint: disable=missing-function-docstring
        assert len(grad_outputs) > 0, "No gradients received for backprop!"

        if isinstance(ctx.split_size_or_sections, (list, tuple)):
            split_sizes = ctx.split_size_or_sections
            assert len(grad_outputs) == len(
                split_sizes
            ), "Unequal number of gradients vs split sections for backprop!"
        if isinstance(ctx.split_size_or_sections, int):
            split_sizes = [ctx.split_size_or_sections] * len(grad_outputs)
        dims = len(grad_outputs[0].shape)
        split_dim = (ctx.split_dim + dims) % dims
        from transformer_engine.pytorch.float8_tensor import Float8Tensor

        if isinstance(grad_outputs[0], Float8Tensor):
            noop_ok = True
            strides = grad_outputs[0].stride()
            data_ptr = grad_outputs[0]._data.untyped_storage().data_ptr()
            shape = list(grad_outputs[0].shape)
            for i, tensor in enumerate(grad_outputs):
                shape_i = shape
                shape_i[split_dim] = split_sizes[i]
                offset_size = sum(split_sizes[:i]) * np.prod(shape[split_dim + 1 :])
                if (
                    tensor.stride() != strides
                    or list(tensor.shape) != shape_i
                    or tensor._data.untyped_storage().data_ptr() != data_ptr
                    or tensor.storage_offset() != offset_size
                ):
                    noop_ok = False
                    break
            if noop_ok:
                ret = torch.Tensor().to(
                    device=grad_outputs[0].device, dtype=grad_outputs[0]._data.dtype
                )
                new_shape = list(shape)
                new_shape[split_dim] = sum(split_sizes)
                ret.set_(
                    grad_outputs[0]._data.untyped_storage(),
                    grad_outputs[0]._data.storage_offset(),
                    new_shape,
                    strides,
                )
                return (
                    Float8Tensor.make_like(grad_outputs[0], data=ret, shape=ret.shape),
                    None,
                    None,
                )

            grad_outputs_data = [x._data for x in grad_outputs]
            data = torch.cat(grad_outputs_data, dim=split_dim)
            return (
                Float8Tensor.make_like(grad_outputs[0], data=data, shape=data.shape),
                None,
                None,
                None,
            )
        noop_ok = True
        strides = grad_outputs[0].stride()
        data_ptr = grad_outputs[0].untyped_storage().data_ptr()
        shape = list(grad_outputs[0].shape)
        for i, tensor in enumerate(grad_outputs):
            shape_i = shape
            shape_i[split_dim] = split_sizes[i]
            offset_size = sum(split_sizes[:i]) * np.prod(shape[split_dim + 1 :])
            if (
                tensor.stride() != strides
                or list(tensor.shape) != shape_i
                or tensor.untyped_storage().data_ptr() != data_ptr
                or tensor.storage_offset() != offset_size
            ):
                noop_ok = False
                break
        if noop_ok:
            ret = torch.Tensor().to(device=grad_outputs[0].device, dtype=grad_outputs[0].dtype)
            new_shape = list(shape)
            new_shape[split_dim] = sum(split_sizes)
            ret.set_(
                grad_outputs[0].untyped_storage(),
                grad_outputs[0].storage_offset(),
                new_shape,
                strides,
            )
            return ret, None, None

        return torch.cat(grad_outputs, dim=split_dim), None, None


def validate_ctx_manager(ctx: Callable) -> None:
    """Checks if passed in object can be used as a context manager."""
    try:
        with ctx():
            pass
    except Exception as e:
        raise ValueError("Object must be a valid ctx manager") from e


def validate_rng_states_func(get_rng_tracker: Callable) -> None:
    """Checks if passed in param function has everything
    required for tensor/model and sequence parallel.
    """
    assert callable(get_rng_tracker), "get_rng_tracker is not a valid function"

    rng_tracker = None
    try:
        rng_tracker = get_rng_tracker()
    except Exception as e:
        raise RuntimeError("Cannot call get_rng_tracker function") from e

    assert hasattr(rng_tracker, "get_states") and callable(
        rng_tracker.get_states
    ), "rng_tracker object does not have valid method get_states"
    assert hasattr(rng_tracker, "set_states") and callable(
        rng_tracker.set_states
    ), "rng_tracker object does not have valid method set_states"
    assert hasattr(rng_tracker, "fork") and callable(
        rng_tracker.fork
    ), "rng_tracker object does not have valid method fork"
    validate_ctx_manager(rng_tracker.fork)


def assert_viewless_tensor(tensor: torch.Tensor, extra_msg: Optional[str] = None) -> torch.Tensor:
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set)."""
    if isinstance(tensor, list):
        return [assert_viewless_tensor(t) for t in tensor]
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        f"likely accumulate over iterations). {extra_msg}"
    )
    return tensor


def safely_set_viewless_tensor_data(tensor: torch.Tensor, new_data_tensor: torch.Tensor) -> None:
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    extra_msg = (
        "FYI, tensor._base has shape "
        f"{'--' if tensor._base is None else tensor._base.shape},"
        f"and new_data_tensor has shape {new_data_tensor.shape}."
    )
    assert_viewless_tensor(tensor, extra_msg=extra_msg)
    tensor.data = new_data_tensor


def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype"""
    if tensor is None:
        return None
    if tensor.dtype == dtype:
        return tensor
    with torch.enable_grad():
        return tensor.to(dtype=dtype)


def check_dim_for_fp8_exec(tensor: torch.Tensor) -> bool:
    """Check if tensor dimensions are supported for FP8 TN GEMM"""
    return tensor.dim() == 2 and tensor.size(0) % 8 == 0 and tensor.size(1) % 16 == 0


def assert_dim_for_fp8_exec(*tensors: List[torch.Tensor]) -> None:
    """Assert that tensor or tensors dimensions are supported for FP8 TN GEMM."""

    for tensor in tensors:
        assert math.prod(tensor.shape[:-1]) % 8 == 0 and tensor.shape[-1] % 16 == 0, (
            "FP8 execution requires the product of all dimensions except the last to be divisible"
            " by 8 and the last dimension to be divisible by 16, but got tensor with"
            f" dims={list(tensor.size())}"
        )


def is_bf16_compatible() -> None:
    """Replaces torch.cuda.is_bf16_compatible() with an explicit
    check on device compute capability to enforce sm_80 or higher.
    """
    return torch.cuda.get_device_capability()[0] >= 8


def is_non_tn_fp8_gemm_supported() -> bool:
    """Checks whether the device supports
    non-TN layouts for FP8 GEMMs.
    """
    device_capability = torch.cuda.get_device_capability()
    return (10, 0) <= device_capability < (12, 0) or device_capability >= (13, 0)


@functools.lru_cache(maxsize=None)
def get_cudnn_version() -> Tuple[int, int, int]:
    """Runtime cuDNN version (major, minor, patch)"""
    encoded_version = ext.get_cudnn_version()
    major_version_magnitude = 1000 if encoded_version < 90000 else 10000
    major, encoded_version = divmod(encoded_version, major_version_magnitude)
    minor, patch = divmod(encoded_version, 100)
    return (major, minor, patch)


def canonicalize_device(device: Optional[torch.device | str]) -> torch.device:
    """Canonicalize PyTorch device

    If `None`, then returns the default CUDA device.

    """
    if device is None:
        # Use default CUDA device
        device = torch.get_default_device()
        if device.type != "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def canonicalize_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Canonicalize PyTorch datatype

    If `None`, then returns the default PyTorch datatype.

    """
    if dtype is None:
        # Use default dtype
        dtype = torch.get_default_dtype()
    return dtype


def devices_match(device1: torch.device, device2: torch.device) -> bool:
    """Whether two devices are the same"""
    device1 = torch.device(device1)
    device2 = torch.device(device2)
    if device1.type != device2.type:
        return False
    if device1.type == "cuda":
        index1 = device1.index
        index2 = device2.index
        if index1 == index2:
            return True
        if index1 is None:
            index1 = torch.cuda.current_device()
        if index2 is None:
            index2 = torch.cuda.current_device()
        return index1 == index2
    return device1 == device2


@functools.lru_cache
def get_sm_count() -> int:
    """Returns the number of streaming multiprocessors in the current device."""
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def round_up_to_nearest_multiple(value, multiple):
    """Round up `value` to the next mutiple of `multiple`"""
    if multiple == 0:
        raise ValueError("multiple cannot be zero.")
    return ((value + multiple - 1) // multiple) * multiple


def needs_quantized_gemm(obj, rowwise=True):
    """Used to check if obj will need quantized gemm or normal gemm."""
    if isinstance(obj, DebugQuantizedTensor):
        return type(obj.get_tensor(not rowwise)) not in [  # pylint: disable=unidiomatic-typecheck
            torch.Tensor,
            torch.nn.Parameter,
        ]
    return type(obj) not in [
        torch.Tensor,
        torch.nn.Parameter,
    ]  # pylint: disable=unidiomatic-typecheck


@functools.lru_cache(maxsize=None)
def _nvtx_enabled() -> bool:
    """Check if NVTX range profiling is enabled"""
    return bool(int(os.getenv("NVTE_NVTX_ENABLED", "0")))


# Messages associated with active NVTX ranges
_nvtx_range_messages: list[str] = []


def nvtx_range_push(msg: str) -> None:
    """Push NVTX range onto stack, if NVTX range profiling is enabled

    Set `NVTE_NVTX_ENABLED=1` in the environment to enable NVTX range
    profiling.

    Parameters
    ----------
    msg: str
        Message to associate with range

    """
    if not _nvtx_enabled():
        return
    _nvtx_range_messages.append(msg)
    torch.cuda.nvtx.range_push(msg)


def nvtx_range_pop(msg: Optional[str] = None) -> None:
    """Pop NVTX range from stack, if NVTX range profiling is enabled

    Set `NVTE_NVTX_ENABLED=1` in the environment to enable NVTX range
    profiling.

    Parameters
    ----------
    msg: str, optional
        Message associated with range

    """

    # Return immediately if NVTX range profiling is not enabled
    if not _nvtx_enabled():
        return

    # Update list of NVTX range messages and check for consistency
    if not _nvtx_range_messages:
        raise RuntimeError("Attempted to pop NVTX range from empty stack")
    last_msg = _nvtx_range_messages.pop()
    if msg is not None and msg != last_msg:
        raise ValueError(
            f"Attempted to pop NVTX range from stack with msg={msg}, "
            f"but last range has msg={last_msg}"
        )

    # Pop NVTX range
    torch.cuda.nvtx.range_pop()


def canonicalize_process_group(
    group: Optional[torch.distributed.ProcessGroup],
) -> torch.distributed.ProcessGroup:
    """Convert to PyTorch process group

    If `None`, returns default process group.

    """
    if group is None:
        return torch.distributed.distributed_c10d._get_default_group()
    return group


def torch_get_autocast_gpu_dtype() -> torch.dtype:
    """Get PyTorch autocast GPU dtype."""
    if torch_version() >= (2, 4, 0):
        return torch.get_autocast_dtype("cuda")
    return torch.get_autocast_gpu_dtype()


if torch_version() >= (2, 4, 0):
    gpu_autocast_ctx = functools.partial(torch.amp.autocast, device_type="cuda")
else:
    gpu_autocast_ctx = torch.cuda.amp.autocast
