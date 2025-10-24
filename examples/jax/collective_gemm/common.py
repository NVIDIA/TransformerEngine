# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared functions for the comm_overlap tests"""

import jax.numpy as jnp
import numpy as np


# Add this after your existing imports
def dtype_tols(dtype, rtol=None, atol=None):
    """Expected numerical tolerance for a data type."""
    # Return immediately if tolerances are fully specified
    if rtol is not None and atol is not None:
        return {"rtol": rtol, "atol": atol}

    # Default tolerances for common dtypes
    if dtype in [jnp.float32, "float32"]:
        return {"rtol": 1e-5, "atol": 1e-8}
    elif dtype in [jnp.float16, "float16"]:
        return {"rtol": 1e-3, "atol": 1e-6}
    elif dtype in [jnp.bfloat16, "bfloat16"]:
        return {"rtol": 1e-2, "atol": 1e-5}
    else:
        return {"rtol": 1e-5, "atol": 1e-8}


def assert_allclose(
    actual,
    desired,
    rtol=None,
    atol=None,
    dtype=None,
    **kwargs,
):
    """Check if two tensors are close."""
    # Infer data type if needed
    if dtype is None:
        if isinstance(actual, float):
            dtype = "float32"
        else:
            dtype = actual.dtype

    # Determine tolerances
    tols = {}
    if rtol is None or atol is None:
        tols = dtype_tols(dtype)
    if rtol is not None:
        tols["rtol"] = rtol
    if atol is not None:
        tols["atol"] = atol

    # Cast tensors to fp32
    if not isinstance(actual, float):
        actual = actual.astype(jnp.float32)
    if not isinstance(desired, float):
        desired = desired.astype(jnp.float32)

    # Check if tensors are close
    np.testing.assert_allclose(actual, desired, **tols, **kwargs)


def assert_allclose_print_index(ref_output, gathered_output, rtol=1e-5, atol=1e-8):
    if not jnp.allclose(ref_output, gathered_output, rtol=rtol, atol=atol):
        diff = jnp.abs(ref_output - gathered_output)
        mask = diff > (atol + rtol * jnp.abs(gathered_output))
        print(mask.astype(int))
        print(jnp.where(mask, diff, 0))


# Shared constants for all tests
DP_AXIS = "data"
TPSP_AXIS = "tensor_sequence"
PARAMS_KEY = "params"

# Shared functions for distributed testing
import argparse
import jax
from jax.experimental import mesh_utils
from transformer_engine.jax.cpp_extensions.gemm import collective_gemm_bootstrap

# Global flag to track if distributed has been initialized
_distributed_initialized = False


def _is_distributed_initialized():
    """Check if JAX distributed has been initialized."""
    return _distributed_initialized


def _initialize_distributed(args):
    """Initialize JAX distributed with custom arguments."""
    global _distributed_initialized

    # Check if already initialized
    if _distributed_initialized:
        return

    if args.coordinator_address is None or args.num_processes is None or args.process_id is None:
        raise ValueError(
            "All distributed initialization arguments are required: "
            "--coordinator-address, --num-processes, --process-id"
        )
    if args.local_device_ids is None:
        assert (
            args.num_devices_per_process is not None
        ), "Either local_device_ids or num_devices_per_process must be provided"
        # Calculate device range for this process
        # Single process single device: each process gets one unique device
        # Single process multiple devices: each process gets a unique range of devices
        start_device = args.process_id * args.num_devices_per_process
        device_range = range(start_device, start_device + args.num_devices_per_process)
        global_device_ids_for_this_process = ",".join(map(str, device_range))
    else:
        # Use explicitly provided global device IDs
        global_device_ids_for_this_process = args.local_device_ids
        args.num_devices_per_process = len(args.local_device_ids.split(","))

    assert args.num_devices_per_process == 1, "Only single process single GPU is supported!"

    print(
        f"Initializing JAX distributed with coordinator={args.coordinator_address}, "
        f"num_processes={args.num_processes}, process_id={args.process_id}"
    )
    # Note: "local_device_ids" is a JAX term meaning "global CUDA devices managed by this process"
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=global_device_ids_for_this_process,
    )

    _distributed_initialized = True
    jax.clear_caches()
    jax.config.update(
        "jax_use_shardy_partitioner", False
    )  # CollectiveGEMM does not work with Shardy yet

    assert jax.local_device_count() == 1, (
        f"[{args.process_id}|{args.num_devices_per_process}] Expected 1 GPU per process, found"
        f" {jax.local_device_count()}"
    )

    devices_per_process = 1
    num_total_devices = args.num_processes

    print(
        f"Initializing CGEMM communicator with num_total_devices={num_total_devices},"
        f" devices_per_process={devices_per_process}, process_id={args.process_id}"
    )

    collective_gemm_bootstrap(
        num_total_devices=num_total_devices,
        num_devices_per_process=devices_per_process,
        process_id=args.process_id,
        tensor_parallel_size=args.tensor_parallel_size,
    )


def _get_dp_and_tp_sizes(args):
    num_gpu = args.num_processes * args.num_devices_per_process
    if args.tensor_parallel_size is None:
        num_gpu_dp = 2 if args.enable_data_parallel else 1
        assert (
            num_gpu > 1 and num_gpu % num_gpu_dp == 0
        ), "Number of GPUs must be greater than 1 and divisible by number of data parallel GPUs"
        num_gpu_tp = num_gpu // num_gpu_dp
    else:
        num_gpu_tp = args.tensor_parallel_size
        assert (
            num_gpu > 1 and num_gpu % num_gpu_tp == 0
        ), "Number of GPUs must be greater than 1 and divisible by number of data parallel GPUs"
        num_gpu_dp = num_gpu // num_gpu_tp
    return num_gpu_dp, num_gpu_tp


def _create_mesh(args):
    """Create mesh configuration with proper validation."""
    num_gpu = args.num_processes * args.num_devices_per_process
    assert num_gpu == len(jax.devices()), "Number of GPUs must be equal to number of devices"
    num_gpu_dp, num_gpu_tp = _get_dp_and_tp_sizes(args)

    print(f"Using {num_gpu_dp}x{num_gpu_tp} mesh ({num_gpu_dp * num_gpu_tp} total GPUs)")

    device_mesh = mesh_utils.create_device_mesh((num_gpu_dp, num_gpu_tp))
    mesh = jax.sharding.Mesh(devices=device_mesh, axis_names=(DP_AXIS, TPSP_AXIS))
    return mesh


def cgemm_parser(description="Collective GEMM test on multi-GPU with tensor parallelism"):
    """Create common argument parser for all collective GEMM tests."""
    parser = argparse.ArgumentParser(description=description)

    # Distributed initialization arguments
    parser.add_argument(
        "--coordinator-address",
        type=str,
        default=None,
        help="Coordinator address for distributed initialization",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes for distributed initialization",
    )
    parser.add_argument(
        "--process-id", type=int, default=None, help="Process ID for distributed initialization"
    )
    parser.add_argument(
        "--local-device-ids",
        type=str,
        default=None,
        help="Local device IDs for distributed initialization (comma-separated)",
    )
    parser.add_argument(
        "--num-devices-per-process", type=int, default=1, help="Number of devices per process"
    )

    # Test configuration arguments
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None, help="Tensor parallel size"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length for testing")
    parser.add_argument("--hidden-in", type=int, default=4096, help="Input hidden dimension")
    parser.add_argument("--hidden-out", type=int, default=8192, help="Output hidden dimension")
    parser.add_argument(
        "--collective-type",
        type=str,
        default="all_gather",
        choices=["all_gather", "reduce_scatter"],
        help="Type of collective operation",
    )
    parser.add_argument(
        "--fp8-recipe", type=str, default="DelayedScaling", help="FP8 recipe to use"
    )
    parser.add_argument(
        "--enable-data-parallel", action="store_true", help="Enable data parallelism"
    )
    parser.add_argument(
        "--enable-result-check", action="store_true", default=True, help="Enable result checking"
    )

    return parser
