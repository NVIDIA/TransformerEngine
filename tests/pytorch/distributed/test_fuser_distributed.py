# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import subprocess
import sys

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.fuser as te_fuser
from transformer_engine.pytorch.fuser.ops._common import is_float8_tensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.float8_tensor import Float8Tensor
import transformer_engine_extensions as tex

# Skip tests if there are not enough GPUs
if torch.cuda.device_count() < 2:
    pytest.skip(
        "Distributed tests require at least 2 GPUs "
        f"(found {torch.cuda.device_count()})",
        allow_module_level=True,
    )

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

@functools.cache
def world_sizes() -> list[int]:
    """World sizes for multi-GPU jobs"""
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        return []
    sizes = [2]
    while sizes[-1] * 2 <= num_gpus:
        sizes.append(sizes[-1] * 2)
    return sizes

@functools.cache
def current_file() -> pathlib.Path:
    """Path to current Python file"""
    return pathlib.Path(__file__).resolve()

@functools.cache
def python_exe() -> pathlib.Path:
    """Path to Python executable"""
    return pathlib.Path(sys.executable).resolve()

def init_distributed() -> tuple[torch.distributed.ProcessGroup, int, int]:
    """Initialize NCCL process group"""
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    group = torch.distributed.init_process_group(
        "nccl",
        init_method="file:///tmp/rdzv",
        world_size=world_size,
        rank=rank,
    )
    return group, world_size, rank

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

def _test_unfused_linear() -> None:

    # Initialize process group
    process_group, world_size, rank = init_distributed()

    # Tensor dimensions
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    in_features = int(os.getenv("IN_FEATURES", "16"))
    out_features = int(os.getenv("OUT_FEATURES", "16"))
    in_shape = [batch_size, in_features]
    out_shape = [batch_size, out_features]

    # Tensor-parallel configuration
    tensor_parallel_mode = os.getenv("TENSOR_PARALLEL_MODE", "column")
    sequence_parallel = bool(int(os.getenv("SEQUENCE_PARALLEL", "0")))

    # Compute configuration
    dtype = torch.float32
    device = "cuda"

    # FP8 configuration
    fp8_compute = bool(int(os.getenv("FP8_COMPUTE", "0")))
    fp8_input = bool(int(os.getenv("FP8_INPUT", "0")))
    fp8_weight = bool(int(os.getenv("FP8_WEIGHT", "0")))
    fp8_grad_output = bool(int(os.getenv("FP8_GRAD_OUTPUT", "0")))

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
                (rank+1) * local_out_features,
            )
            w_ref = w_ref[local_slice, :].clone()
            dw_ref = dw_ref[local_slice, :].clone()
            w_test = w_test[local_slice, :].clone()
            y_ref = y_ref[:, local_slice].clone()
            dy_ref = dy_ref[:, local_slice].clone()
            dy_test = dy_test[:, local_slice].clone()
        elif tensor_parallel_mode == "row":
            local_in_features = in_features // world_size
            local_slice = slice(
                rank * local_in_features,
                (rank+1) * local_in_features,
            )
            w_ref = w_ref[:, local_slice].clone()
            dw_ref = dw_ref[:, local_slice].clone()
            w_test = w_test[:, local_slice].clone()
            x_ref = x_ref[:, local_slice].clone()
            dx_ref = dx_ref[:, local_slice].clone()
            x_test = x_test[:, local_slice].clone()
        if sequence_parallel:
            local_batch_size = batch_size // world_size
            local_slice = slice(
                rank * local_batch_size,
                (rank+1) * local_batch_size,
            )
            if tensor_parallel_mode == "column":
                x_ref = x_ref[local_slice, :].clone()
                dx_ref = dx_ref[local_slice, :].clone()
                x_test = x_test[local_slice, :].clone()
            elif tensor_parallel_mode == "row":
                y_ref = y_ref[local_slice, :].clone()
                dy_ref = dy_ref[local_slice, :].clone()
                dy_test = dy_test[local_slice, :].clone()
    x_test.requires_grad_()

    # Implementation with fusable operation
    with te.fp8_model_init(enabled=fp8_weight):
        op = te_fuser.ops.UnfusedLinear(
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
            op.weight._fp8_dtype
            if is_float8_tensor(op.weight)
            else tex.DType.kFloat8E4M3
        )

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **tols)
    torch.testing.assert_close(dx_test, dx_ref, **tols)
    torch.testing.assert_close(dw_test, dw_ref, **tols)


@pytest.mark.skipif(torch.cuda.device_count() <= 1, reason="")
class TestDistributedFuserOps:

    @pytest.mark.parametrize("fp8_compute", (False, True))
    @pytest.mark.parametrize("fp8_input", (False, True))
    @pytest.mark.parametrize("fp8_weight", (False, True))
    @pytest.mark.parametrize("fp8_grad_output", (False, True))
    @pytest.mark.parametrize("world_size", world_sizes())
    @pytest.mark.parametrize("tensor_parallel_mode", ("row", "column"))
    @pytest.mark.parametrize("sequence_parallel", (False, True))
    def test_unfused_linear(
        self,
        *,
        local_weight_shape: tuple[int, int] = (16, 16),
        batch_size: Iterable[int] = 16,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        fp8_compute: bool,
        fp8_input: bool,
        fp8_weight: bool,
        fp8_grad_output: bool,
        world_size: int,
        tensor_parallel_mode: str,
        sequence_parallel: bool,
    ):

        # Tensor dimensions
        local_out_features, local_in_features = local_weight_shape
        out_features, in_features = local_out_features, local_in_features
        if tensor_parallel_mode == "column":
            out_features *= world_size
        elif tensor_parallel_mode == "row":
            in_features *= world_size

        # Skip invalid configurations
        if fp8_compute or fp8_input or fp8_weight or fp8_grad_output:
            if not fp8_available:
                pytest.skip(reason_for_no_fp8)
            if torch.device(device).type != "cuda":
                pytest.skip("FP8 is only supported on CUDA devices")
        if fp8_compute:
            if (
                batch_size % 16 != 0
                or local_in_features % 16 != 0
                or local_out_features % 16 != 0
            ):
                pytest.skip("FP8 GEMMs require dims that are divisible by 16")

        # Launch distributed job
        env = dict(os.environ)
        env["BATCH_SIZE"] = batch_size
        env["IN_FEATURES"] = in_features
        env["OUT_FEATURES"] = out_features
        env["TENSOR_PARALLEL_MODE"] = tensor_parallel_mode
        env["SEQUENCE_PARALLEL"] = int(sequence_parallel)
        env["FP8_COMPUTE"] = int(fp8_compute)
        env["FP8_INPUT"] = int(fp8_input)
        env["FP8_WEIGHT"] = int(fp8_weight)
        env["FP8_GRAD_OUTPUT"] = int(fp8_grad_output)
        env = { key: str(val) for key, val in env.items() }
        command = [
            python_exe(),
            "-m",
            "torch.distributed.launch",
            "--use-env",
            f"--nproc_per_node={world_size}",
            current_file(),
            "--test=unfused_linear",
        ]
        result = subprocess.run(command, stderr=subprocess.STDOUT, env=env)
        if result.returncode != 0:
            print(result.stdout, file=sys.stderr)
            result.check_returncode()


def main() -> None:

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default=None, help="Test name")
    args = parser.parse_args()

    # Execute test if provided
    if args.test is None:
        pass
    elif args.test == "unfused_linear":
        _test_unfused_linear()
    else:
        raise RuntimeError(f"Unrecognized test ({args.test})")

if __name__ == "__main__":
    main()
