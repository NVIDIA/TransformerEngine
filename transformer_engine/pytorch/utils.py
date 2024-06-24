# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utility functions for Transformer Engine modules"""
import math
import functools
from typing import Any, Callable, Optional, Tuple
import torch
import transformer_engine.pytorch.cpp_extensions as ext


def requires_grad(*tensors: Tuple[Optional[torch.Tensor], ...]) -> None:
    """Check if any of the given tensors require gradient."""
    for tensor in tensors:
        if tensor is not None and tensor.requires_grad:
            return True
    return False


def clear_tensor_data(*tensors: Tuple[Optional[torch.Tensor], ...]) -> None:
    """
    Trick to deallocate tensor memory when delete operation does not
    release the tensor due to PyTorch override.

    Must be used carefully.
    """
    from .float8_tensor import Float8Tensor

    for t in tensors:
        if t is not None:
            if isinstance(t, Float8Tensor):
                t._data.data = torch.Tensor()
                del t
            else:
                t.data = torch.Tensor()
                del t


def get_device_compute_capability() -> Tuple[int, int]:
    """CUDA compute capability of current GPU"""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.major, props.minor)


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
    with torch.enable_grad():
        return tensor if tensor is None or tensor.dtype == dtype else tensor.to(dtype)


def check_dim_for_fp8_exec(tensor: torch.Tensor) -> bool:
    """Check if tensor dimensions are supported for FP8 TN GEMM"""
    return tensor.dim() == 2 and tensor.size(0) % 8 == 0 and tensor.size(1) % 16 == 0


def assert_dim_for_fp8_exec(tensor: torch.Tensor) -> None:
    """Assert that tensor dimensions are supported for FP8 TN GEMM"""
    # single tensor check so it's clear which tensor is triggering the assertion
    assert tensor.dim() == 2 and tensor.size(0) % 8 == 0 and tensor.size(1) % 16 == 0, (
        "FP8 execution requires 2D input matrices with "
        "height divisible by 8 and width divisible by 16, "
        f"but got tensor with dims={list(tensor.size())}"
    )


def is_bf16_compatible() -> None:
    """Replaces torch.cuda.is_bf16_compatible() with an explicit
    check on device compute capability to enforce sm_80 or higher.
    """
    return torch.cuda.get_device_capability()[0] >= 8


@functools.cache
def get_cudnn_version() -> Tuple[int, int, int]:
    """Runtime cuDNN version (major, minor, patch)"""
    encoded_version = ext.get_cudnn_version()
    major_version_magnitude = 1000 if encoded_version < 90000 else 10000
    major, encoded_version = divmod(encoded_version, major_version_magnitude)
    minor, patch = divmod(encoded_version, 100)
    return (major, minor, patch)
