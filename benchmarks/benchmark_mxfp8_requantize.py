# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare TE requantization with cuDNN Frontend's Mxfp8ColRequant on SM100+.

Calls the common C API directly so tensor allocation and Python tensor mutation
are excluded. cuDNN input/output layout conversion is reported separately.
The external kernel is loaded from an installed cuDNN Frontend package or
--cudnn-source; no external source is vendored into TE.
"""

import argparse
import ctypes as ct
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import transformer_engine.pytorch  # noqa: F401
from transformer_engine.common import _get_shared_object_file
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import triton
import triton.language as tl
from requantize_mxfp8_experimental import CompactRequantize, CompactTmaRequantize
from transformer_engine.common.CuTeDSL.cast.mxfp8.requantize_mxfp8 import GroupedRequantize

EXTERNAL_MODULE = (
    "cudnn.moe_ep._megamoe_backend.cutedsl_src.kernel_src.rubin.training.mega.fwd_glu."
    "glu_mxfp8_col_requant"
)


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
            "nvte_grouped_requantize": (None, [ct.c_void_p] * 4),
            "nvte_group_requantize": (None, [ct.c_void_p] * 6),
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


@triton.jit
def swizzle_input(src, dst, H: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    """Compact rowwise scales -> dispatch-pool / TE GEMM scales."""
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = x // (H // 32), x % (H // 32)
    offset = ((row // 128) * (H // 128) + col // 4) * 512
    offset += (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
    tl.store(dst + offset, tl.load(src + x, x < SIZE, other=0), x < SIZE)


def swizzle_reference(scales):
    """Native torch reference for the 128x4 scale atom."""
    rows, cols = scales.shape
    return (
        scales.reshape(rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


def quantize_reference(values, columnwise=False):
    """Power-of-two scaling, round-to-nearest-even E4M3, independent of TE."""
    rows, hidden = values.shape
    blocks = (
        values.float().reshape(rows // 32, 32, hidden)
        if columnwise
        else values.float().reshape(rows, hidden // 32, 32)
    )
    axis = 1 if columnwise else 2
    amax = torch.where(torch.isnan(blocks), 0.0, blocks.abs()).amax(axis)
    exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
    exponent = torch.where(amax == 0, -127, exponent)
    inverse = torch.exp2(-exponent).clamp(max=torch.finfo(torch.float32).max)
    data = (
        (blocks * inverse.unsqueeze(axis))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape(rows, hidden)
    )
    return data, (exponent + 127).to(torch.uint8)


def timing(fn, iterations, repeats):
    """CUDA graph replay removes Python launch gaps; median of event batches."""
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


def kernel_names(fn):
    """Record the GPU launches, independently of the timing batches."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        fn()
        torch.cuda.synchronize()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


def benchmark_case(args, external, api, rows, hidden, sizes):
    """Validate all layouts before collecting any timing."""
    groups, live = len(sizes), sum(sizes)
    counts = torch.tensor(sizes, device="cuda", dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))) * hidden
    stream = torch.cuda.current_stream().cuda_stream
    # Normal BF16 activations with blockwise power-of-two variation.
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    powers = torch.randint(-8, 9, (rows, hidden // 32), device="cuda")
    source *= torch.exp2(powers.float()).repeat_interleave(32, dim=1)
    if args.pattern == "zeros":
        source.zero_()
    data, compact = quantize_reference(source)
    if args.pattern in ("special", "special_mixed"):
        data.view(torch.uint8).fill_(56)  # exactly 1.0 in E4M3
        codes = torch.tensor([0, 1, 2, 100, 127, 245, 254, 255], device="cuda", dtype=torch.uint8)
        compact.copy_(codes[(torch.arange(rows, device="cuda") // 32) % len(codes), None])
        data.view(torch.uint8)[64:96].fill_(126)  # 448 * 2^-125
        data.view(torch.uint8)[96:128].fill_(128)  # negative zero
        data.view(torch.uint8)[128:160].fill_(127)  # all-NaN columns
        data.view(torch.uint8)[160:192].fill_(255)  # negative NaN
        if args.pattern == "special_mixed":
            compact.copy_(codes[torch.arange(rows, device="cuda") % len(codes), None])
    decoded_scales = torch.exp2(compact.float() - 127)
    decoded_scales = torch.where(compact == 255, float("nan"), decoded_scales)
    dequant = data.float() * decoded_scales.repeat_interleave(32, 1)
    dequant = dequant.to(torch.bfloat16)
    scales = torch.full_like(compact, 0xA5).flatten()
    live_scales = live * hidden // 32
    swizzle = lambda: swizzle_input[(triton.cdiv(live_scales, 1024),)](
        compact, scales, hidden, live_scales, 1024
    )
    swizzle()
    assert torch.equal(scales[:live_scales], swizzle_reference(compact)[:live_scales])
    assert (scales[live_scales:] == 0xA5).all()
    te_data = torch.full((rows * hidden,), 0xA5, dtype=torch.uint8, device="cuda")
    te_col_sf, te_row_sf = torch.full_like(scales, 0xA5), torch.full_like(scales, 0xA5)
    metadata = [(7, counts, 3, (groups,)), (9, offsets, 3, (groups + 1,))]
    inp = api.tensor(
        [(0, data, 7, (rows * hidden,)), (4, compact, 9, (compact.numel(),))] + metadata,
        rows,
        hidden,
        groups,
    )
    out = api.tensor(
        [
            (0, data, 7, (rows * hidden,)),
            (1, te_data, 7, (rows * hidden,)),
            (4, te_row_sf, 9, (scales.numel(),)),
            (5, te_col_sf, 9, (scales.numel(),)),
        ]
        + metadata,
        rows,
        hidden,
        groups,
        True,
    )
    te = lambda: api.lib.nvte_grouped_requantize(inp, out, api.config, stream)
    te()
    ref_data, ref_sf, start = [], [], 0
    for size in sizes:
        if size:
            q, sf = quantize_reference(dequant[start : start + size], True)
            ref_data.append(q.contiguous().view(torch.uint8).flatten())
            ref_sf.append(swizzle_reference(sf.T.contiguous()))
        start += size
    ref_data, ref_sf = torch.cat(ref_data), torch.cat(ref_sf)
    torch.cuda.synchronize()
    assert torch.equal(te_data[: live * hidden], ref_data), "TE payload differs from reference"
    assert torch.equal(te_col_sf[: live * hidden // 32], ref_sf), "TE column scales differ"
    assert torch.equal(te_row_sf[: live * hidden // 32], scales[: live * hidden // 32])
    assert (te_data[live * hidden :] == 0xA5).all(), "TE touched capacity tail"
    assert (te_col_sf[live_scales:] == 0xA5).all(), "TE touched column scale tail"
    assert (te_row_sf[live_scales:] == 0xA5).all(), "TE touched row scale tail"

    results = {
        "rows": rows,
        "hidden": hidden,
        "sizes": sizes,
        "timings": {},
        "compile_seconds": {},
        "configs": {},
        "validation": {"te": "passed"},
    }
    results["timings"]["te_grouped"] = timing(te, args.iterations, args.repeats)
    if args.profile:
        results["kernel_names"] = {"te_grouped": kernel_names(te)}
    if args.cutedsl_dispatch:
        api.lib.nvte_set_cutedsl_backend.argtypes = [ct.c_bool]
        api.lib.nvte_set_cutedsl_backend.restype = None
        api.lib.nvte_set_cutedsl_backend(True)
        te_data.fill_(0xA5)
        te_col_sf.fill_(0xA5)
        te_row_sf.fill_(0xA5)
        begin = time.perf_counter()
        te()
        torch.cuda.synchronize()
        results["compile_seconds"]["te_cutedsl"] = time.perf_counter() - begin
        assert torch.equal(te_data[: live * hidden], ref_data), "CuTeDSL dispatch data"
        assert torch.equal(te_col_sf[:live_scales], ref_sf), "CuTeDSL dispatch column scales"
        assert torch.equal(te_row_sf[:live_scales], scales[:live_scales])
        assert (te_data[live * hidden :] == 0xA5).all()
        assert (te_col_sf[live_scales:] == 0xA5).all()
        assert (te_row_sf[live_scales:] == 0xA5).all()
        results["validation"]["te_cutedsl"] = "passed"
        results["timings"]["te_cutedsl"] = timing(te, args.iterations, args.repeats)
        if args.profile:
            results["kernel_names"]["te_cutedsl"] = kernel_names(te)
        api.lib.nvte_set_cutedsl_backend(False)
    # Earlier dense fused API, still used for an optional BF16 output.
    dense_in = api.tensor([(0, data, 7, (rows, hidden)), (4, compact, 9, (rows, hidden // 32))])
    dense_out = api.tensor(
        [
            (0, data, 7, (rows, hidden)),
            (1, te_data, 7, (rows, hidden)),
            (4, te_row_sf, 9, (scales.numel(),)),
            (5, te_col_sf, 9, (scales.numel(),)),
        ]
    )
    off = api.tensor([(0, offsets, 3, (groups + 1,))])
    dense = lambda: api.lib.nvte_group_requantize(
        dense_in, dense_out, off, None, api.config, stream
    )
    dense()
    torch.cuda.synchronize()
    assert torch.equal(te_data[: live * hidden], ref_data)
    assert torch.equal(te_col_sf[: live * hidden // 32], ref_sf)
    results["timings"]["te_dense"] = timing(dense, args.iterations, args.repeats)
    results["timings"]["input_swizzle"] = timing(swizzle, args.iterations, args.repeats)
    if args.adapted:
        adapted_dst = torch.full_like(te_data, 0xA5).view(torch.float8_e4m3fn).view(rows, hidden)
        adapted_row, adapted_col = torch.full_like(scales, 0xA5), torch.full_like(scales, 0xA5)
        adapted_args = tuple(
            from_dlpack(x, assumed_align=16)
            for x in (data, compact, offsets, adapted_dst, adapted_row, adapted_col)
        ) + (cuda.CUstream(stream),)
        kernel = GroupedRequantize(
            hidden, groups, rows, sm_count=torch.cuda.get_device_properties(0).multi_processor_count
        )
        begin = time.perf_counter()
        adapted_compiled = cute.compile(kernel, *adapted_args)
        results["compile_seconds"]["adapted"] = time.perf_counter() - begin
        adapted_fn = lambda: adapted_compiled(*adapted_args)
        adapted_fn()
        torch.cuda.synchronize()
        assert torch.equal(
            adapted_dst.view(torch.uint8).flatten()[: ref_data.numel()], ref_data
        ), "adapted data"
        assert torch.equal(adapted_col[: ref_sf.numel()], ref_sf), "adapted column scales"
        assert torch.equal(adapted_row[:live_scales], scales[:live_scales]), "adapted row scales"
        assert (adapted_dst.view(torch.uint8).flatten()[ref_data.numel() :] == 0xA5).all()
        assert (adapted_col[live_scales:] == 0xA5).all()
        assert (adapted_row[live_scales:] == 0xA5).all()
        results["timings"]["adapted"] = timing(adapted_fn, args.iterations, args.repeats)
        results["configs"]["adapted"] = {
            "tile_hidden": kernel.TILE_HID,
            "stages": kernel.NumStages,
            "grid": kernel.grid,
            "threads": kernel.ThreadsPerCta,
            "smem_bytes": kernel.smem_bytes,
        }
        if args.profile:
            results["kernel_names"]["adapted"] = kernel_names(adapted_fn)
        if args.adapted_column_only:
            column_args = adapted_args[:4] + (None,) + adapted_args[5:]
            column_kernel = GroupedRequantize(
                hidden,
                groups,
                rows,
                rowwise_output=False,
                sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
            )
            column_compiled = cute.compile(column_kernel, *column_args)
            column_fn = lambda: column_compiled(*column_args)
            adapted_dst.view(torch.uint8).fill_(0xA5)
            adapted_col.fill_(0xA5)
            column_fn()
            torch.cuda.synchronize()
            assert torch.equal(
                adapted_dst.view(torch.uint8).flatten()[: ref_data.numel()], ref_data
            )
            assert torch.equal(adapted_col[: ref_sf.numel()], ref_sf)
            assert (adapted_dst.view(torch.uint8).flatten()[ref_data.numel() :] == 0xA5).all()
            assert (adapted_col[live_scales:] == 0xA5).all()
            results["timings"]["adapted_column_only"] = timing(
                column_fn, args.iterations, args.repeats
            )
    for token_blocks, hidden_blocks in ((1, 1), (4, 1), (4, 2)) if args.compact_experiment else ():
        label = f"compact_{token_blocks}x{hidden_blocks}"
        dst = torch.empty_like(data)
        dst.view(torch.uint8).fill_(0xA5)
        row_sf, col_sf = torch.empty_like(scales), torch.empty_like(scales)
        cute_args = tuple(
            from_dlpack(x, assumed_align=16)
            for x in (data, compact.flatten(), offsets, dst, row_sf, col_sf)
        ) + (cuda.CUstream(stream),)
        begin = time.perf_counter()
        compiled = cute.compile(
            CompactRequantize(hidden, groups, token_blocks, hidden_blocks), *cute_args
        )
        results["compile_seconds"][label] = time.perf_counter() - begin
        fn = lambda: compiled(*cute_args)
        fn()
        torch.cuda.synchronize()
        assert torch.equal(dst.view(torch.uint8).flatten()[: ref_data.numel()], ref_data), label
        assert torch.equal(col_sf[: ref_sf.numel()], ref_sf), label
        assert torch.equal(row_sf[: live * hidden // 32], scales[: live * hidden // 32]), label
        results["timings"][label] = timing(fn, args.iterations, args.repeats)
        if args.profile:
            results["kernel_names"][label] = kernel_names(fn)
    if args.compact_experiment:
        sf_pair = torch.full((compact.numel() * 2,), 0xA5, device="cuda", dtype=torch.uint8)
        sf_pair[: compact.numel()].copy_(compact.flatten())
        dst, col_sf = torch.empty_like(data), torch.empty_like(scales)
        cute_args = tuple(
            from_dlpack(x, assumed_align=16) for x in (data, sf_pair, counts, dst, col_sf)
        ) + (cuda.CUstream(stream),)
        kernel = CompactTmaRequantize(hidden, groups, rows, "mxfp8_e4m3", scaled_cvt=False)
        compiled = cute.compile(kernel, *cute_args)
        fn = lambda: compiled(*cute_args)
        fn()
        torch.cuda.synchronize()
        assert torch.equal(
            dst.view(torch.uint8).flatten()[: ref_data.numel()], ref_data
        ), "compact_tma"
        assert torch.equal(col_sf[: ref_sf.numel()], ref_sf), "compact_tma"
        assert torch.equal(
            sf_pair[compact.numel() : compact.numel() + live * hidden // 32],
            scales[: live * hidden // 32],
        ), "compact_tma"
        results["timings"]["compact_tma"] = timing(fn, args.iterations, args.repeats)
    for k_major in (False, True):
        label = "cudnn_transposed" if k_major else "cudnn_row_major"
        dst = torch.empty_like(data)
        dst.view(torch.uint8).fill_(0xA5)
        dst_sf = torch.full_like(scales, 0xA5)
        cute_args = tuple(
            from_dlpack(x, assumed_align=16) for x in (data, scales, counts, dst, dst_sf)
        ) + (cuda.CUstream(stream),)
        kernel = external.Mxfp8ColRequant(
            hidden, groups, rows, "mxfp8_e4m3", scaled_cvt=False, dst_k_major=k_major
        )
        begin = time.perf_counter()
        compiled = cute.compile(kernel, *cute_args)
        results["compile_seconds"][label] = time.perf_counter() - begin
        fn = lambda: compiled(*cute_args)
        fn()
        torch.cuda.synchronize()
        if k_major:
            normalized = []
            start = 0
            for size in sizes:
                if size:
                    normalized.append(
                        dst.view(torch.uint8)
                        .reshape(hidden, rows)[:, start : start + size]
                        .T.contiguous()
                        .flatten()
                    )
                start += size
            normalized = torch.cat(normalized)
        else:
            normalized = []
            start = 0
            for size in sizes:
                if size:
                    normalized.append(
                        dst.view(torch.uint8)[start : start + size].contiguous().flatten()
                    )
                start += size
            normalized = torch.cat(normalized)
        payload_mismatches = int((normalized != ref_data).sum())
        scale_mismatches = int((dst_sf[: ref_sf.numel()] != ref_sf).sum())
        results["validation"][label] = {
            "payload_mismatches": payload_mismatches,
            "scale_mismatches": scale_mismatches,
        }
        if args.pattern not in ("special", "special_mixed"):
            assert payload_mismatches == 0, f"{label} payload differs"
            assert scale_mismatches == 0, f"{label} scales differ"
        results["timings"][label] = timing(fn, args.iterations, args.repeats)
        if args.profile:
            results["kernel_names"][label] = kernel_names(fn)
        if k_major:
            assert (dst.view(torch.uint8).reshape(hidden, rows)[:, live:] == 0xA5).all()
        else:
            assert (dst.view(torch.uint8)[live:] == 0xA5).all()
        assert (dst_sf[live_scales:] == 0xA5).all()
        results["configs"][label] = {
            "tile_hidden": kernel.TILE_HID,
            "stages": kernel.NumStages,
            "grid": kernel.grid,
            "threads": kernel.ThreadsPerCta,
            "smem_bytes": kernel.smem_bytes,
        }
        if not k_major:

            def compatible():
                swizzle()
                fn()

            results["timings"]["cudnn_compatible"] = timing(
                compatible, args.iterations, args.repeats
            )
            if args.tune:
                candidates = []
                for tile in (128, 256, 512):
                    if hidden % tile:
                        continue
                    for stages in (1, 2):
                        tuned_type = type(
                            "TunedRequant",
                            (external.Mxfp8ColRequant,),
                            {
                                "TileHidPortable": tile,
                                "NSTAGE": stages,
                            },
                        )
                        tasks = live // 128 * (hidden // tile)
                        for grid in sorted({min(tasks, g) for g in (152, 304, 608, 1216, 2432)}):
                            tuned = tuned_type(
                                hidden,
                                groups,
                                rows,
                                "mxfp8_e4m3",
                                num_persistent_ctas=grid,
                                scaled_cvt=False,
                            )
                            begin = time.perf_counter()
                            run = cute.compile(tuned, *cute_args)
                            compile_seconds = time.perf_counter() - begin
                            tuned_fn = lambda: run(*cute_args)
                            tuned_fn()
                            torch.cuda.synchronize()
                            assert torch.equal(
                                dst.view(torch.uint8).flatten()[: ref_data.numel()], ref_data
                            ), "tuned payload"
                            assert torch.equal(dst_sf[: ref_sf.numel()], ref_sf), "tuned scales"
                            measure = timing(tuned_fn, args.iterations, args.repeats)

                            def tuned_compatible():
                                swizzle()
                                tuned_fn()

                            full = timing(tuned_compatible, args.iterations, args.repeats)
                            candidates.append(
                                {
                                    "tile": tuned.TILE_HID,
                                    "stages": stages,
                                    "grid": grid,
                                    "raw": measure,
                                    "compatible": full,
                                    "compile_seconds": compile_seconds,
                                }
                            )
                results["tuning"] = candidates
                results["best_tuned"] = min(candidates, key=lambda c: c["compatible"]["median_us"])
    # Effective logical traffic: payload read/write plus input, row-output and
    # column-output scales. This excludes metadata and does not count redundant
    # TMA scale reads or claim to measure physical DRAM traffic.
    logical_bytes = live * hidden * (2 + 3 / 32)
    results["effective_logical_bytes"] = int(logical_bytes)
    results["effective_bandwidth_tb_s"] = {
        label: logical_bytes / measure["median_us"] / 1e6
        for label, measure in results["timings"].items()
        if label in ("te_grouped", "te_cutedsl", "adapted", "te_dense")
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["4096,4096,8", "32768,4096,64"],
        help="Comma-separated rows,hidden,groups triples",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--occupancy", type=float, default=1.0)
    parser.add_argument("--imbalance", choices=("uniform", "uneven"), default="uniform")
    parser.add_argument("--cudnn-source", type=Path)
    parser.add_argument("--compact-experiment", action="store_true")
    parser.add_argument("--adapted", action="store_true")
    parser.add_argument("--adapted-column-only", action="store_true")
    parser.add_argument("--cutedsl-dispatch", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--pattern", choices=("normal", "zeros", "special", "special_mixed"), default="normal"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--splits", help="Explicit comma-separated group rows for a single shape")
    parser.add_argument("--output", type=Path, default=Path("/tmp/mxfp8_requantize.json"))
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1 or not 0 < args.occupancy <= 1:
        parser.error("iterations/repeats must be positive; occupancy must be in (0,1]")
    if args.splits and len(args.shapes) != 1:
        parser.error("--splits requires exactly one --shapes entry")
    external = importlib.import_module(EXTERNAL_MODULE)
    if args.cudnn_source:
        spec = importlib.util.spec_from_file_location(
            EXTERNAL_MODULE + "_benchmark", args.cudnn_source
        )
        external = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(external)
    api = CommonAPI()
    # Every C API / CuTeDSL launch and graph capture must use the same
    # non-default stream. A cached default stream would escape capture.
    torch.cuda.set_stream(torch.cuda.Stream())
    torch.manual_seed(1234)
    source = Path(external.__file__)
    report = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cutlass": cutlass.__version__,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "te_library": str(api.path),
            "external_source": str(source),
            "external_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "te_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "envvars": {
                key: value
                for key, value in os.environ.items()
                if key.startswith("NVTE_") or key in ("CUDA_VISIBLE_DEVICES", "CUDA_CACHE_DISABLE")
            },
            "gpu_config": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,driver_version,power.limit",
                    "--format=csv",
                ],
                text=True,
            ).strip(),
        },
        "method": vars(args) | {"output": str(args.output), "cudnn_source": str(args.cudnn_source)},
        "cases": [],
    }
    try:
        for shape in args.shapes:
            rows, hidden, groups = map(int, shape.split(","))
            assert rows % 128 == 0 and hidden % 128 == 0
            blocks = int(rows // 128 * args.occupancy)
            if args.splits:
                sizes = list(map(int, args.splits.split(",")))
                assert len(sizes) == groups and 0 < sum(sizes) <= rows
                assert all(size >= 0 and size % 128 == 0 for size in sizes)
            elif args.imbalance == "uniform":
                sizes = [(blocks // groups + (g < blocks % groups)) * 128 for g in range(groups)]
            else:
                weights = [1 / (g + 1) for g in range(groups)]
                sizes = [int(blocks * w / sum(weights)) for w in weights]
                for g in range(blocks - sum(sizes)):
                    sizes[g % groups] += 1
                sizes = [s * 128 for s in sizes]
            assert sum(sizes) > 0, "The requested occupancy leaves no live rows"
            result = benchmark_case(args, external, api, rows, hidden, sizes)
            report["cases"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        api.close()


if __name__ == "__main__":
    main()
