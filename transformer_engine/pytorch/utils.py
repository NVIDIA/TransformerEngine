# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utility functions for Transformer Engine modules"""
import math
from typing import Any, Callable, Optional, Tuple
import torch


def get_device_compute_capability() -> float:
    """Returns the cuda compute capability of current GPU"""
    major = torch.cuda.get_device_properties(torch.cuda.current_device()).major
    minor = torch.cuda.get_device_properties(torch.cuda.current_device()).minor
    return major + minor / 10


def attention_mask_func(
    attention_scores: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Get attention mask"""
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_default_init_method() -> Callable:
    """Weight initialization method if not provided by user"""
    return init_method_normal(0.023)


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
    assert (
        numerator % denominator == 0
    ), f"{numerator} is not divisible by {denominator}"


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


def assert_viewless_tensor(
    tensor: torch.Tensor, extra_msg: Optional[str] = None
) -> torch.Tensor:
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set)."""
    if isinstance(tensor, list):
        return [assert_viewless_tensor(t) for t in tensor]
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        f"Ensure tensor._base is None before setting tensor.data or storing "
        f"tensor to memory buffer. Otherwise, a memory leak will occur (and "
        f"likely accumulate over iterations). {extra_msg}"
    )
    return tensor


def safely_set_viewless_tensor_data(
    tensor: torch.Tensor, new_data_tensor: torch.Tensor
) -> None:
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    extra_msg = (
        f"FYI, tensor._base has shape "
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
    """For fp8 fprop (TN layout), inputs and weights must be such
       that dim0 is divisible by 8 and dim1 is divisible by 16.
    """
    return not tensor.shape[0] % 8 and not tensor.shape[1] % 16


def assert_dim_for_fp8_exec(tensor: torch.Tensor) -> None:
    """For fp8 fprop (TN layout), inputs and weights must be such
       that dim0 is divisible by 8 and dim1 is divisible by 16.
    """
    # single tensor check so it's clear which tensor is triggering the assertion
    assert check_dim_for_fp8_exec(tensor), (
        "Tensor dimensions are not compatible for FP8 execution: "
        f"({tensor.shape[0]} % 8 != 0, {tensor.shape[1]} % 16 != 0)"
    )
