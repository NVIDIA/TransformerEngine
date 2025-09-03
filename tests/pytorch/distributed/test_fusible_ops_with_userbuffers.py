# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import argparse
import dataclasses
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
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.ops.fused import (
    UserbuffersBackwardLinear,
    UserbuffersForwardLinear,
)
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.utils import is_bf16_compatible

# Import utility functions
_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import dtype_tols, make_recipe, str_to_dtype

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
quantization_list: list[Optional[str]] = [None]
if fp8_available:
    quantization_list.extend(("fp8_delayed_scaling", "fp8_current_scaling"))
if mxfp8_available:
    quantization_list.append("mxfp8")


# Check if there are multiple GPUs
if torch.cuda.device_count() < 2:
    pytest.skip("Userbuffers requires at least 2 GPUs.")


@dataclasses.dataclass
class ModelConfig:
    """Tensor dimensions in Transformer model"""

    sequence_length: int
    batch_size: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    quantization: Optional[str]

    @property
    def hidden_size(self):
        return self.num_heads * self.head_dim


@functools.cache
def launcher() -> str:
    """Launcher for current parallel job"""
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        return "ompi"
    if "TORCHELASTIC_RUN_ID" in os.environ:
        return "torchrun"
    raise RuntimeError(f"{__file__} must be launched with either `mpirun` or `torchrun`")


@functools.cache
def world_group() -> torch.distributed.ProcessGroup:
    """Get NCCL process group, initializing if needed"""

    # Get launch config from environment
    if launcher() == "ompi":
        # OpenMPI
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
        rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
        local_size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE"))
        local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
    elif launcher() == "torchrun":
        # torchrun
        world_size = int(os.getenv("WORLD_SIZE"))
        rank = int(os.getenv("RANK"))
        local_size = int(os.getenv("LOCAL_WORLD_SIZE"))
        local_rank = int(os.getenv("LOCAL_RANK"))
    else:
        raise RuntimeError("Unexpected launcher ({launcher()})")

    # Construct communicator
    assert local_size == world_size
    torch.cuda.set_device(local_rank)
    group = torch.distributed.init_process_group(
        "nccl",
        init_method="file:///tmp/rdzv",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return group


def reset_rng(seed: int = 1234) -> None:
    """Reset random number generators"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | Iterable[int],
    quantization: Optional[str] = None,
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device = "cpu",
    test_dtype: torch.dtype = torch.float32,
    test_device: torch.device = "cuda",
    test_is_quantized: bool = False,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values

    The reference tensor is intended for use in plain PyTorch
    operations in high precision. The test tensor is intended for use
    in Transformer Engine operations.

    If a quantization scheme is provided, the tensor values are
    quantized so that they are representable.

    """

    # Random reference tensor
    ref = torch.rand(shape, dtype=ref_dtype, device=ref_device)

    # Construct test tensor from reference tensor
    test = ref.to(device=test_device, dtype=test_dtype)
    if quantization is None:
        if test_is_quantized:
            raise ValueError("Quantization scheme not provided")
        if test.data_ptr() == ref.data_ptr():
            test = test.clone()
    elif quantization in ("fp8", "fp8_delayed_scaling"):
        quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=test_device).squeeze(),
            amax=torch.zeros(1, dtype=torch.float32, device=test_device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        test = quantizer(test)
    elif quantization == "fp8_current_scaling":
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=test_device,
        )
        test = quantizer(test)
    elif quantization == "mxfp8":
        test = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)(test)
    else:
        raise ValueError(f"Unsupported quantization scheme ({quantization})")
    if isinstance(test, QuantizedTensor) and not test_is_quantized:
        test = test.dequantize()

    # Make sure reference and test tensors match each other
    ref.copy_(test)

    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


def _test_linear(
    *,
    model_config: ModelConfig,
    bias: bool = False,
    device: torch.device = "cuda",
    tensor_parallel_mode: str = "column",
    sequence_parallel: bool = True,
    weight_requires_grad: bool = True,
) -> None:
    dtype = model_config.dtype
    quantization = model_config.quantization
    quantized_compute = quantization is not None

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Tensor dimensions
    out_features = model_config.hidden_size
    in_features = model_config.hidden_size
    batch_size = model_config.sequence_length * model_config.batch_size
    in_shape = [batch_size, in_features]
    out_shape = [batch_size, out_features]

    # Random data
    reset_rng()
    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
    )
    w_ref, w_test = make_reference_and_test_tensors(
        (out_features, in_features),
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
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
        quantization=quantization,
        test_dtype=dtype,
        test_device=device,
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
    recipe = make_recipe(quantization)
    with te.fp8_model_init(enabled=quantized_compute, recipe=recipe):
        ops = []
        linear_op = None
        bias_op = None
        if tensor_parallel_mode == "column":
            userbuffers_options = {}
            if not weight_requires_grad:
                userbuffers_options["comm_name"] = "fc1"
            else:
                userbuffers_options["comm_name"] = "qkv"
            linear_op = te_ops.BasicLinear(
                in_features,
                out_features,
                device=device,
                dtype=dtype,
                tensor_parallel_mode=tensor_parallel_mode,
                tensor_parallel_group=process_group,
                sequence_parallel=sequence_parallel,
                userbuffers_options=userbuffers_options,
            )
            ops.append(linear_op)
            if bias:
                bias_op = te_ops.Bias(
                    out_features // world_size,
                    device=device,
                    dtype=dtype,
                )
                ops.append(bias_op)
        elif tensor_parallel_mode == "row":
            userbuffers_options = dict(comm_name="proj")
            linear_op = te_ops.BasicLinear(
                in_features // world_size,
                out_features,
                device=device,
                dtype=dtype,
                userbuffers_options=userbuffers_options,
            )
            ops.append(linear_op)
            if bias:
                bias_op = te_ops.Bias(out_features, device=device, dtype=dtype)
                ops.append(bias_op)
            ops.append(te_ops.ReduceScatter(process_group))
        model = te_ops.Sequential(*ops)
    with torch.no_grad():
        linear_op.weight.copy_(w_test)
        linear_op.weight.requires_grad_(requires_grad=weight_requires_grad)
        if bias:
            bias_op.bias.copy_(b_test)
        del w_test
        del b_test
    with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
        y_test = model(x_test)
    y_test.backward(dy_test)

    # Check that forward operations have been fused
    forward_ops = model._module_groups[0]._forward_ops
    backward_ops = model._module_groups[0]._backward_ops
    assert len(forward_ops) == 1
    assert len(backward_ops) == 1
    assert isinstance(forward_ops[0][0], UserbuffersForwardLinear)
    assert isinstance(backward_ops[0][0], UserbuffersBackwardLinear)

    # Expected numerical error
    tols = dtype_tols(dtype)
    if dtype == torch.float32:
        tols = dtype_tols(torch.float16)  # TF32 GEMM
    if quantized_compute:
        tols = dtype_tols(
            model[0].weight._fp8_dtype
            if isinstance(model[0].weight, Float8Tensor)
            else tex.DType.kFloat8E4M3
        )

    # Check results
    y_test = y_test.to(dtype=torch.float64, device="cpu")
    dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
    torch.testing.assert_close(y_test, y_ref, **tols)
    torch.testing.assert_close(dx_test, dx_ref, **tols)
    if weight_requires_grad:
        dw_test = linear_op.weight.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(dw_test, dw_ref, **tols)
    if bias:
        db_test = bias_op.bias.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(db_test, db_ref, **tols)


def run_parallel_tests(model_config: ModelConfig) -> None:
    """Run parallel tests"""

    # Distributed process group
    process_group = world_group()
    rank = torch.distributed.get_rank(process_group)
    world_size = torch.distributed.get_world_size(process_group)

    # Linear op
    for test_config in itertools.product(
        (False, True),  # bias
        ("column", "row"),  # tensor_parallel_mode
        (True, False),  # weight_requires_grad
    ):
        if rank == 0:
            print(f"Running _test_linear with {test_config=}")
        bias, tensor_parallel_mode, weight_requires_grad = test_config
        _test_linear(
            model_config=model_config,
            bias=bias,
            tensor_parallel_mode=tensor_parallel_mode,
            weight_requires_grad=weight_requires_grad,
        )


# Parallel job sizes
_world_sizes = []
if torch.cuda.device_count() > 1:
    _world_sizes.append(torch.cuda.device_count())


@pytest.mark.parametrize("world_size", _world_sizes)
@pytest.mark.parametrize("quantization", quantization_list)
def test_fuser_ops_with_userbuffers(
    *,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
    quantization: Optional[str],
) -> None:
    """Launch parallel job and run tests"""

    # Parallel job launcher
    command = []
    if tex.ubuf_built_with_mpi():
        python_exe = pathlib.Path(sys.executable).resolve()
        command.extend(("mpirun", "-np", str(world_size), "--oversubscribe", "--quiet", python_exe))
    else:
        command.extend(("torchrun", f"--nproc_per_node={world_size}"))

    # Script invocation
    command.extend(
        (
            _current_file,
            "--parallel",
            "--batch-size",
            str(world_size),
            "--num-heads",
            str(world_size),
            "--dtype",
            str(dtype),
        )
    )
    if quantization is not None:
        command.extend(("--quantization", quantization))

    # Environment
    env = dict(os.environ)
    if not tex.device_supports_multicast():
        env["UB_SKIPMC"] = "1"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["PYTORCH_JIT"] = "0"
    env["NVTE_TORCH_COMPILE"] = "0"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    # Launch parallel job
    result = subprocess.run(command, check=True, env=env)


def main() -> None:

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="Run parallel tests")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--quantization", type=str, default=None)
    args = parser.parse_args()

    # Run parallel tests if needed
    if args.parallel:

        # Model config
        model_config = ModelConfig(
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dtype=str_to_dtype(args.dtype),
            quantization=args.quantization,
        )

        # Initialize Userbuffers
        group = world_group()  # Initialize NCCL
        bootstrap_backend = "mpi" if launcher() == "ompi" else "nccl"
        userbuffer_configs = {
            "fc1_dgrad": {
                "method": "ring_exchange",
                "fp8_buf": False,
            },  # Overlap dgrad RS with dgrad GEMM
        }
        te.module.base.initialize_ub(
            [
                model_config.sequence_length * model_config.batch_size,
                model_config.num_heads * model_config.head_dim,
            ],
            torch.distributed.get_world_size(group),
            quantization_modes=[
                (
                    te.module.base.UserBufferQuantizationMode.FP8
                    if model_config.quantization is not None
                    else te.module.base.UserBufferQuantizationMode.NONE
                )
            ],
            dtype=model_config.dtype,
            bootstrap_backend=bootstrap_backend,
            ub_cfgs=userbuffer_configs,
        )

        # Run tests
        run_parallel_tests(model_config)

        # Clean up
        te.module.base.destroy_ub()


if __name__ == "__main__":
    main()
