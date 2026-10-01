# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Measure native/CuTeDSL grouped requantization with optional BF16 output.

Requires an editable SM100+ build with NVTE_WITH_CUTEDSL=1. Both backends use
preallocated buffers through the same C API. CUDA graph replay measures GPU
execution without Python dispatch or allocation. First-call time is separate.
"""

import argparse
import ctypes as ct
import json
from importlib.metadata import version
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import tvm_ffi
import transformer_engine.pytorch  # pylint: disable=unused-import
from transformer_engine.common import _get_shared_object_file


class Shape(ct.Structure):
    """NVTEShape ABI."""

    _fields_ = [("data", ct.c_size_t * 15), ("ndim", ct.c_size_t)]

    def __init__(self, dims):
        super().__init__((ct.c_size_t * 15)(*dims), len(dims))


class BasicTensor(ct.Structure):
    """NVTEBasicTensor ABI."""

    _fields_ = [("data_ptr", ct.c_void_p), ("dtype", ct.c_int), ("shape", Shape)]


class CommonAPI:
    """Preallocate common tensor descriptors outside all timed regions."""

    def __init__(self):
        self.path = _get_shared_object_file("core")
        self.lib = ct.CDLL(str(self.path))
        signatures = {
            "nvte_create_grouped_tensor": (ct.c_void_p, [ct.c_int, ct.c_size_t, Shape]),
            "nvte_create_tensor": (ct.c_void_p, [ct.c_int]),
            "nvte_create_quantization_config": (ct.c_void_p, []),
            "nvte_set_grouped_tensor_param": (
                None,
                [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t],
            ),
            "nvte_set_tensor_param_v2": (None, [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t]),
            "nvte_set_quantization_config_attribute": (
                None,
                [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t],
            ),
            "nvte_group_requantize": (None, [ct.c_void_p] * 5),
            "nvte_set_cutedsl_backend": (None, [ct.c_int]),
            "nvte_is_cutedsl_backend_built": (ct.c_int, []),
            "nvte_destroy_tensor": (None, [ct.c_void_p]),
            "nvte_destroy_grouped_tensor": (None, [ct.c_void_p]),
            "nvte_destroy_quantization_config": (None, [ct.c_void_p]),
        }
        for name, (restype, argtypes) in signatures.items():
            func = getattr(self.lib, name)
            func.restype, func.argtypes = restype, argtypes
        self.handles = []
        self.config = self.lib.nvte_create_quantization_config()
        fast = ct.c_bool(True)
        self.lib.nvte_set_quantization_config_attribute(
            self.config, 7, ct.byref(fast), ct.sizeof(fast)
        )

    def tensor(self, params, rows=None, hidden=None, groups=None, swizzled=False):
        """Create a descriptor over caller-owned torch allocations."""
        grouped = groups is not None
        handle = (
            self.lib.nvte_create_grouped_tensor(1, groups, Shape((rows, hidden)))
            if grouped
            else self.lib.nvte_create_tensor(1)
        )
        setter = (
            self.lib.nvte_set_grouped_tensor_param if grouped else self.lib.nvte_set_tensor_param_v2
        )
        self.handles.append((handle, grouped))
        for param, tensor, dtype, dims in params:
            value = BasicTensor(tensor.data_ptr(), dtype, Shape(dims))
            setter(handle, param, ct.byref(value), ct.sizeof(value))
        if swizzled:
            value = ct.c_bool(True)
            setter(handle, 10 if grouped else 7, ct.byref(value), ct.sizeof(value))
        return handle

    def close(self):
        for handle, grouped in self.handles:
            destroy = (
                self.lib.nvte_destroy_grouped_tensor if grouped else self.lib.nvte_destroy_tensor
            )
            destroy(handle)
        self.handles.clear()
        self.lib.nvte_destroy_quantization_config(self.config)


def timing(fn, iterations, repeats):
    """Return event-batch means after five calls and three graph warmups."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(iterations):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def run_case(args, rows, hidden, return_dequantized):
    """Check all outputs exactly, identify dispatch, then time each backend."""
    groups = 4
    if rows % (groups * 128) or hidden % 128:
        raise ValueError("Shapes must have four uniform 128-aligned groups")
    torch.manual_seed(42)
    # Random E4M3 data with independent compact E8M0 scales, decoded exactly in BF16.
    data = (torch.randn(rows, hidden, device="cuda") * 8).to(torch.float8_e4m3fn)
    compact = torch.randint(100, 155, (rows, hidden // 32), device="cuda", dtype=torch.uint8)
    decoded_ref = (data.float() * torch.exp2(compact.float() - 127).repeat_interleave(32, 1)).to(
        torch.bfloat16
    )
    counts = torch.full((groups,), rows // groups, device="cuda", dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))) * hidden
    dst = torch.empty(rows * hidden, device="cuda", dtype=torch.uint8)
    row_sf = torch.empty_like(compact).flatten()
    col_sf = torch.empty_like(row_sf)
    decoded = torch.empty_like(decoded_ref)
    api = CommonAPI()
    if not api.lib.nvte_is_cutedsl_backend_built():
        raise RuntimeError("Rebuild TE with NVTE_WITH_CUTEDSL=1")
    metadata = [(7, counts, 3, (groups,)), (9, offsets, 3, (groups + 1,))]
    inp = api.tensor(
        [(0, data, 7, (data.numel(),)), (4, compact, 9, (compact.numel(),))] + metadata,
        rows,
        hidden,
        groups,
    )
    out = api.tensor(
        [
            (0, data, 7, (data.numel(),)),
            (1, dst, 7, (dst.numel(),)),
            (4, row_sf, 9, (row_sf.numel(),)),
            (5, col_sf, 9, (col_sf.numel(),)),
        ]
        + metadata,
        rows,
        hidden,
        groups,
        True,
    )
    dequant = api.tensor([(0, decoded, 6, (rows, hidden))]) if return_dequantized else None
    stream = torch.cuda.current_stream().cuda_stream
    call = lambda: api.lib.nvte_group_requantize(inp, out, dequant, api.config, stream)
    result = {
        "rows": rows,
        "hidden": hidden,
        "groups": groups,
        "return_dequantized": return_dequantized,
    }
    reference = None
    torch.cuda.synchronize()
    for enabled in (False, True):
        backend = "cutedsl" if enabled else "cuda"
        api.lib.nvte_set_cutedsl_backend(enabled)
        begin = time.perf_counter()
        call()
        torch.cuda.synchronize()
        first_seconds = time.perf_counter() - begin
        outputs = (dst, row_sf, col_sf) + ((decoded,) if return_dequantized else ())
        if reference is None:
            reference = tuple(output.clone() for output in outputs)
        else:
            for actual, expected in zip(outputs, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if return_dequantized:
            torch.testing.assert_close(decoded, decoded_ref, rtol=0, atol=0)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
            call()
            torch.cuda.synchronize()
        names = sorted(
            set(
                event.name
                for event in profile.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            )
        )
        if enabled:
            # A failed compilation falls back to CUDA; do not label that as CuTeDSL performance.
            registered = (
                f"cutedsl_mxfp8_requant_Float8E4M3_{rows}_{hidden}_{groups}_0_1_1_"
                f"{int(return_dequantized)}_100_"
                f"{torch.cuda.get_device_properties(0).multi_processor_count}"
            )
            if tvm_ffi.get_global_func(registered, allow_missing=True) is None:
                raise RuntimeError(f"CuTeDSL specialization was not registered: {registered}")
        result[backend] = {
            "first_call_seconds": first_seconds,
            "kernels": names,
            **timing(call, args.iterations, args.repeats),
        }
    result["speedup"] = result["cuda"]["median_us"] / result["cutedsl"]["median_us"]
    api.close()
    return result


def main():
    """Write reproducible per-batch timings and environment metadata as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["1024x256", "8192x1024", "16384x8192"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mxfp8_requantize_results.json"))
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    results = {
        "environment": {
            "gpu": props.name,
            "visible_gpu_count": torch.cuda.device_count(),
            "tested_gpu_count": 1,
            "compute_capability": [props.major, props.minor],
            "sm_count": props.multi_processor_count,
            "torch": torch.__version__,
            "cutlass_dsl": version("nvidia-cutlass-dsl"),
            "tvm_ffi": version("apache-tvm-ffi"),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": torch.cuda.nccl.version(),
            "library": str(_get_shared_object_file("core")),
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).splitlines()[0],
            "NVTE": {key: value for key, value in os.environ.items() if key.startswith("NVTE_")},
        },
        "command": sys.argv,
        "method": {
            "input": "row-major E4M3 from N(0, 8), compact E8M0 codes uniform in [100, 154]",
            "output": "row-major E4M3, GEMM-swizzled scales, optional row-major BF16",
            "recipe": "MXFP8 1D scaling, use_fast_math=True (BF16 intermediate)",
            "warmup": "5 eager calls, 3 graph replays of iterations calls",
            "timing": (
                "median of event batch means; synchronization before timing and after each batch"
            ),
            "excluded": "allocation, compilation, graph capture, Python and C API dispatch",
        },
        "iterations": args.iterations,
        "repeats": args.repeats,
        "cases": [],
    }
    # CUDA graph capture requires a non-default stream; C API calls use it too.
    with torch.cuda.stream(torch.cuda.Stream()):
        for shape in args.shapes:
            rows, hidden = map(int, shape.split("x"))
            for dequantized in (False, True):
                case = run_case(args, rows, hidden, dequantized)
                results["cases"].append(case)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
