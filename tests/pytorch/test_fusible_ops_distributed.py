# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import argparse
import functools
import itertools
import os
import pathlib
import subprocess
import sys

import pytest
import torch

import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.ops._common import is_float8_tensor
from transformer_engine.pytorch.utils import is_bf16_compatible
import transformer_engine_torch as tex

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


@functools.cache
def world_group() -> torch.distributed.ProcessGroup:
    """Get NCCL process group, initializing if needed"""
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    group = torch.distributed.init_process_group(
        "nccl",
        init_method="file:///tmp/rdzv",
        world_size=world_size,
        rank=rank,
    )
    return group


def reset_rng(seed: int = 1234) -> None:
    """Reset random number generators"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | Iterable[int],
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device = "cpu",
    test_dtype: torch.dtype = torch.float32,
    test_device: torch.device = "cuda",
    test_is_fp8: bool = False,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values

    The reference tensor is intended for use in plain PyTorch
    operations in high precision. The test tensor is intended for use
    in Transformer Engine operations.

    """

    # Random data
    ref = torch.rand(shape, dtype=ref_dtype, device=ref_device)

    # Make copy of tensor
    if test_is_fp8:
        test = Float8Tensor.to_float8(ref)
    else:
        test = ref.to(device=test_device, dtype=test_dtype)
        if test.data_ptr() == ref.data_ptr():
            test = test.clone()

    # Make sure reference and test tensors represent exact same values
    ref.copy_(test)

    # Return reference and test tensors
    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


def dtype_tols(dtype: torch.dtype | tex.DType) -> dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """

    # Transformer Engine dtypes
    if isinstance(dtype, tex.DType):
        if dtype == tex.DType.kFloat8E4M3:
            return dict(rtol=0.125, atol=0.0675)  # epsilon = 0.0625
        if dtype == tex.DType.kFloat8E5M2:
            return dict(rtol=0.25, atol=0.125)  # epsilon = 0.152
        dtype = {
            tex.DType.kByte: torch.uint8,
            tex.DType.kInt32: torch.int32,
            tex.DType.kFloat32: torch.float32,
            tex.DType.kFloat16: torch.half,
            tex.DType.kBFloat16: torch.bfloat16,
        }[dtype]

    # PyTorch dtypes
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float64:
        return dict(rtol=1e-7, atol=1e-7)
    raise ValueError(f"Unsupported dtype ({dtype})")


def _test_all_reduce(
    *,
    local_size: int = 17,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    fp8: bool = False,
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    in_shape = [world_size, local_size]
    out_shape = [local_size]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=fp8,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=fp8,
    )

    # Plain PyTorch implementation
    y_ref = x_ref.sum(0)
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dx_ref = x_ref.grad[rank]
        x_ref = x_ref[rank]
        x_test = x_test[rank].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    op = te_ops.AllReduce(process_group=process_group)
    y_test = op(x_test)
    y_test.backward(dy_test)

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **dtype_tols(dtype))
    torch.testing.assert_close(dx_test, dx_ref, rtol=0, atol=0)


def _test_all_gather(
    *,
    local_size: int = 13,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    fp8: bool = False,
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    in_shape = [world_size, local_size]
    out_shape = [world_size, world_size * local_size]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=fp8,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=fp8,
    )

    # Plain PyTorch implementation
    y_ref = x_ref.tile((world_size, 1)).reshape(out_shape)
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dx_ref = x_ref.grad[rank]
        x_ref = x_ref[rank]
        x_test = x_test[rank].clone()
        y_ref = y_ref[rank]
        dy_ref = dy_ref[rank]
        dy_test = dy_test[rank].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    op = te_ops.AllGather(process_group=process_group)
    y_test = op(x_test)
    y_test.backward(dy_test)

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, rtol=0, atol=0)
    torch.testing.assert_close(dx_test, dx_ref, **dtype_tols(dtype))


def _test_reduce_scatter(
    *,
    local_size: int = 11,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    fp8: bool = False,
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    in_shape = [world_size, world_size * local_size]
    out_shape = [world_size, local_size]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=fp8,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=fp8,
    )

    # Plain PyTorch implementation
    y_ref = x_ref.sum(0).reshape(out_shape)
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dx_ref = x_ref.grad[rank]
        x_ref = x_ref[rank]
        x_test = x_test[rank].clone()
        y_ref = y_ref[rank]
        dy_ref = dy_ref[rank]
        dy_test = dy_test[rank].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    op = te_ops.ReduceScatter(process_group=process_group)
    y_test = op(x_test)
    y_test.backward(dy_test)

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **dtype_tols(dtype))
    torch.testing.assert_close(dx_test, dx_ref, rtol=0, atol=0)


def _test_basic_linear(
    *,
    local_weight_shape: tuple[int, int] = (16, 16),
    batch_size: int = 16,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    fp8_compute: bool = False,
    fp8_input: bool = False,
    fp8_weight: bool = False,
    fp8_grad_output: bool = False,
    tensor_parallel_mode: str = "column",
    sequence_parallel: bool = False,
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    local_out_features, local_in_features = local_weight_shape
    out_features, in_features = local_out_features, local_in_features
    if tensor_parallel_mode == "column":
        out_features *= world_size
    elif tensor_parallel_mode == "row":
        in_features *= world_size
    in_shape = [batch_size, in_features]
    out_shape = [batch_size, out_features]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=(fp8_compute or fp8_input),
    )
    w_ref, w_test = make_reference_and_test_tensors(
        (out_features, in_features),
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=(fp8_compute or fp8_weight),
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=(fp8_compute or fp8_grad_output),
        requires_grad=False,
    )

    # Plain PyTorch implementation
    y_ref = torch.nn.functional.linear(x_ref, w_ref)
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dw_ref = w_ref.grad
        dx_ref = x_ref.grad
        if tensor_parallel_mode == "column":
            local_out_features = out_features // world_size
            local_slice = slice(
                rank * local_out_features,
                (rank + 1) * local_out_features,
            )
            w_ref = w_ref[local_slice, :]
            dw_ref = dw_ref[local_slice, :]
            w_test = w_test[local_slice, :]
            y_ref = y_ref[..., local_slice]
            dy_ref = dy_ref[..., local_slice]
            dy_test = dy_test[..., local_slice].clone()
        elif tensor_parallel_mode == "row":
            local_in_features = in_features // world_size
            local_slice = slice(
                rank * local_in_features,
                (rank + 1) * local_in_features,
            )
            w_ref = w_ref[:, local_slice]
            dw_ref = dw_ref[:, local_slice]
            w_test = w_test[:, local_slice]
            x_ref = x_ref[..., local_slice]
            dx_ref = dx_ref[..., local_slice]
            x_test = x_test[..., local_slice].clone()
        if sequence_parallel:
            local_batch_size = batch_size // world_size
            local_slice = slice(
                rank * local_batch_size,
                (rank + 1) * local_batch_size,
            )
            if tensor_parallel_mode == "column":
                x_ref = x_ref[local_slice, ...]
                dx_ref = dx_ref[local_slice, ...]
                x_test = x_test[local_slice, ...].clone()
            elif tensor_parallel_mode == "row":
                y_ref = y_ref[local_slice, ...]
                dy_ref = dy_ref[local_slice, ...]
                dy_test = dy_test[local_slice, ...].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    with te.fp8_model_init(enabled=fp8_weight):
        op = te_ops.BasicLinear(
            in_features,
            out_features,
            device=device,
            dtype=dtype,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_group=process_group,
            sequence_parallel=sequence_parallel,
        )
    with torch.no_grad():
        op.weight.copy_(w_test)
        del w_test
    with te.fp8_autocast(enabled=fp8_compute):
        y_test = op(x_test)
    y_test.backward(dy_test)

    # Expected numerical error
    tols = dtype_tols(dtype)
    if dtype == torch.float32:
        tols = dtype_tols(torch.float16)  # TF32 GEMM
    if fp8_compute:
        tols = dtype_tols(
            op.weight._fp8_dtype if is_float8_tensor(op.weight) else tex.DType.kFloat8E4M3
        )

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **tols)
    torch.testing.assert_close(dx_test, dx_ref, **tols)
    torch.testing.assert_close(dw_test, dw_ref, **tols)


def _test_linear(
    *,
    bias: bool = True,
    local_weight_shape: tuple[int, int] = (16, 16),
    batch_size: int = 16,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    fp8_compute: bool = False,
    fp8_input: bool = False,
    fp8_weight: bool = False,
    fp8_grad_output: bool = False,
    tensor_parallel_mode: str = "column",
    sequence_parallel: bool = False,
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    local_out_features, local_in_features = local_weight_shape
    out_features, in_features = local_out_features, local_in_features
    if tensor_parallel_mode == "column":
        out_features *= world_size
    elif tensor_parallel_mode == "row":
        in_features *= world_size
    in_shape = [batch_size, in_features]
    out_shape = [batch_size, out_features]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=(fp8_compute or fp8_input),
    )
    w_ref, w_test = make_reference_and_test_tensors(
        (out_features, in_features),
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=(fp8_compute or fp8_weight),
    )
    b_ref, b_test = None, None
    if bias:
        if tensor_parallel_mode == "row":
            bias_shape = [world_size, out_features]
        else:
            bias_shape = [out_features]
        b_ref, b_test = make_reference_and_test_tensors(
            bias_shape,
            test_dtype=dtype,
            test_device=device,
        )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        test_dtype=dtype,
        test_device=device,
        test_is_fp8=(fp8_compute or fp8_grad_output),
        requires_grad=False,
    )

    # Plain PyTorch implementation
    y_ref = torch.nn.functional.linear(x_ref, w_ref)
    if bias:
        if tensor_parallel_mode == "row":
            y_ref += b_ref.sum(dim=0)
        else:
            y_ref += b_ref
    y_ref.backward(dy_ref)

    # Convert to distributed tensors
    with torch.no_grad():
        dw_ref = w_ref.grad
        db_ref = b_ref.grad if bias else None
        dx_ref = x_ref.grad
        if tensor_parallel_mode == "column":
            local_out_features = out_features // world_size
            local_slice = slice(
                rank * local_out_features,
                (rank + 1) * local_out_features,
            )
            w_ref = w_ref[local_slice, :]
            dw_ref = dw_ref[local_slice, :]
            w_test = w_test[local_slice, :]
            if bias:
                b_ref = b_ref[local_slice]
                db_ref = db_ref[local_slice]
                b_test = b_test[local_slice]
            y_ref = y_ref[..., local_slice]
            dy_ref = dy_ref[..., local_slice]
            dy_test = dy_test[..., local_slice].clone()
        elif tensor_parallel_mode == "row":
            local_in_features = in_features // world_size
            local_slice = slice(
                rank * local_in_features,
                (rank + 1) * local_in_features,
            )
            w_ref = w_ref[:, local_slice]
            dw_ref = dw_ref[:, local_slice]
            w_test = w_test[:, local_slice]
            if bias:
                b_ref = b_ref[rank, :]
                db_ref = db_ref[rank, :]
                b_test = b_test[rank, :]
            x_ref = x_ref[..., local_slice]
            dx_ref = dx_ref[..., local_slice]
            x_test = x_test[..., local_slice].clone()
        if sequence_parallel:
            local_batch_size = batch_size // world_size
            local_slice = slice(
                rank * local_batch_size,
                (rank + 1) * local_batch_size,
            )
            if tensor_parallel_mode == "column":
                x_ref = x_ref[local_slice, ...]
                dx_ref = dx_ref[local_slice, ...]
                x_test = x_test[local_slice, ...].clone()
            elif tensor_parallel_mode == "row":
                y_ref = y_ref[local_slice, ...]
                dy_ref = dy_ref[local_slice, ...]
                dy_test = dy_test[local_slice, ...].clone()
    x_test.requires_grad_()

    # Implementation with fusible operation
    with te.fp8_model_init(enabled=fp8_weight):
        model = te_ops.Sequential(
            te_ops.Linear(
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype,
                tensor_parallel_mode=tensor_parallel_mode,
                tensor_parallel_group=process_group,
                sequence_parallel=sequence_parallel,
            ),
        )
    with torch.no_grad():
        model[0].weight.copy_(w_test)
        if bias:
            model[0].bias.copy_(b_test)
        del w_test
        del b_test
    with te.fp8_autocast(enabled=fp8_compute):
        y_test = model(x_test)
    y_test.backward(dy_test)

    # Expected numerical error
    tols = dtype_tols(dtype)
    if dtype == torch.float32:
        tols = dtype_tols(torch.float16)  # TF32 GEMM
    if fp8_compute:
        tols = dtype_tols(
            model[0].weight._fp8_dtype
            if is_float8_tensor(model[0].weight)
            else tex.DType.kFloat8E4M3
        )

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    dw_test = model[0].weight.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **tols)
    torch.testing.assert_close(dx_test, dx_ref, **tols)
    torch.testing.assert_close(dw_test, dw_ref, **tols)
    if bias:
        db_test = model[0].bias.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(db_test, db_ref, **tols)


def _test_fp8_scale_update(
    *,
    amax_history_len: int = 31,
    amax_compute_algo: str = "max",
    margin: float = 2,
    local_weight_shape: tuple[int, int] = (16, 16),
    batch_size: int = 16,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    tensor_parallel_mode: str = "column",
) -> None:

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    local_out_features, local_in_features = local_weight_shape
    out_features, in_features = local_out_features, local_in_features
    if tensor_parallel_mode == "column":
        out_features *= world_size
    elif tensor_parallel_mode == "row":
        in_features *= world_size
    in_shape = [batch_size, in_features]
    out_shape = [batch_size, out_features]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        test_dtype=dtype,
        test_device=device,
    )
    w_ref, w_test = make_reference_and_test_tensors(
        (out_features, in_features),
        test_dtype=dtype,
        test_device=device,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        test_dtype=dtype,
        test_device=device,
        requires_grad=False,
    )

    def ref_amax_and_scale(
        ref: torch.Tensor,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expected absmax and FP8 scale"""
        amax = ref.abs().amax()
        max_val = {
            "forward": 448.0,
            "backward": 57344.0,
        }[stage]
        scale = (max_val / amax) / (2**margin)
        amax = amax.to(dtype=torch.float32, device="cpu")
        scale = scale.to(dtype=torch.float32, device="cpu")
        return amax, scale

    # Compute expected amaxes and FP8 scales
    x_amax_ref, x_scale_ref = ref_amax_and_scale(x_ref, "forward")
    w_amax_ref, w_scale_ref = ref_amax_and_scale(w_ref, "forward")
    dy_amax_ref, dy_scale_ref = ref_amax_and_scale(dy_ref, "backward")

    # Convert to distributed tensors
    with torch.no_grad():
        if tensor_parallel_mode == "column":
            local_out_features = out_features // world_size
            local_slice = slice(
                rank * local_out_features,
                (rank + 1) * local_out_features,
            )
            w_ref = w_ref[local_slice, :]
            w_test = w_test[local_slice, :]
            dy_ref = dy_ref[..., local_slice]
            dy_test = dy_test[..., local_slice].clone()
        elif tensor_parallel_mode == "row":
            local_in_features = in_features // world_size
            local_slice = slice(
                rank * local_in_features,
                (rank + 1) * local_in_features,
            )
            w_ref = w_ref[:, local_slice]
            w_test = w_test[:, local_slice]
            x_ref = x_ref[..., local_slice]
            x_test = x_test[..., local_slice].clone()
    x_test.requires_grad_()

    # Initialize fusible operation
    op = te_ops.BasicLinear(
        in_features,
        out_features,
        device=device,
        dtype=dtype,
        tensor_parallel_mode=tensor_parallel_mode,
        tensor_parallel_group=process_group,
    )
    with torch.no_grad():
        op.weight.copy_(w_test)
        del w_test

    # Forward and backward pass
    fp8_format = transformer_engine.common.recipe.Format.HYBRID
    recipe = transformer_engine.common.recipe.DelayedScaling(
        margin=margin,
        interval=1,
        fp8_format=fp8_format,
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
    )
    with te.fp8_autocast(fp8_recipe=recipe):
        y_test = op(x_test)
    y_test.backward(dy_test)

    # Check results
    forward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
    backward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=False)
    x_fp8_meta = op.get_fp8_meta("input")[forward_key]
    w_fp8_meta = op.get_fp8_meta("param")[forward_key]
    dy_fp8_meta = op.get_fp8_meta("grad_output")[backward_key]
    x_amax_test = x_fp8_meta.amax_history[-1, 0].to(dtype=torch.float32, device="cpu")
    w_amax_test = w_fp8_meta.amax_history[-1, 0].to(dtype=torch.float32, device="cpu")
    dy_amax_test = dy_fp8_meta.amax_history[-1, 0].to(dtype=torch.float32, device="cpu")
    x_scale_test = x_fp8_meta.scale[0].to(dtype=torch.float32, device="cpu")
    w_scale_test = w_fp8_meta.scale[0].to(dtype=torch.float32, device="cpu")
    dy_scale_test = dy_fp8_meta.scale[0].to(dtype=torch.float32, device="cpu")
    torch.testing.assert_close(x_amax_test, x_amax_ref)
    torch.testing.assert_close(w_amax_test, w_amax_ref)
    torch.testing.assert_close(dy_amax_test, dy_amax_ref)
    torch.testing.assert_close(x_scale_test, x_scale_ref)
    torch.testing.assert_close(w_scale_test, w_scale_ref)
    torch.testing.assert_close(dy_scale_test, dy_scale_ref)


def run_parallel_tests() -> None:
    """Run parallel tests"""

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Collective communication ops
    if rank == 0:
        print(f"Running _test_all_reduce")
    _test_all_reduce()
    if rank == 0:
        print(f"Running _test_all_gather")
    _test_all_gather()
    if rank == 0:
        print(f"Running _test_reduce_scatter")
    _test_reduce_scatter()

    # Basic linear op
    for config in itertools.product(
        (False, True) if fp8_available else (False,),
        ("column", "row"),
        (False, True),
    ):
        if rank == 0:
            print(f"Running _test_basic_linear with {config=}")
        fp8, tensor_parallel_mode, sequence_parallel = config
        _test_basic_linear(
            fp8_compute=fp8,
            fp8_input=fp8,
            fp8_weight=fp8,
            fp8_grad_output=fp8,
            tensor_parallel_mode=tensor_parallel_mode,
            sequence_parallel=sequence_parallel,
        )

    # Linear op
    for config in itertools.product(
        (False, True) if fp8_available else (False,),
        ("column", "row"),
    ):
        if rank == 0:
            print(f"Running _test_linear with {config=}")
        fp8, tensor_parallel_mode = config
        dtype = torch.bfloat16 if is_bf16_compatible() else torch.float32
        _test_linear(
            bias=True,  # bias=False is tested in _test_basic_linear
            dtype=dtype,
            fp8_compute=fp8,
            fp8_input=fp8,
            fp8_weight=fp8,
            fp8_grad_output=fp8,
            tensor_parallel_mode=tensor_parallel_mode,
        )

    # FP8 scale update
    if fp8_available:
        if rank == 0:
            print(f"Running _test_fp8_scale_update")
        _test_fp8_scale_update()


# Parallel job sizes
_world_sizes = [torch.cuda.device_count()]
if 1 not in _world_sizes:
    _world_sizes.append(1)
if torch.cuda.device_count() >= 2 and 2 not in _world_sizes:
    _world_sizes.append(2)


@pytest.mark.parametrize("world_size", _world_sizes)
def test_distributed_fuser_ops(world_size: int) -> None:
    """Launch parallel job that runs parallel tests"""
    python_exe = pathlib.Path(sys.executable).resolve()
    current_file = pathlib.Path(__file__).resolve()
    command = [
        python_exe,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        current_file,
        "--parallel",
    ]
    result = subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="Run parallel tests")
    args = parser.parse_args()
    if args.parallel:
        run_parallel_tests()


if __name__ == "__main__":
    main()
