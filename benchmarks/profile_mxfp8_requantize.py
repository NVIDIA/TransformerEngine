# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Profile one adapted requantization launch with Nsight Compute.

Use --profile-from-start off to exclude compilation and five warmup launches.
Input distribution matches benchmark_mxfp8_requantize.py's normal pattern.
"""

import argparse

import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from transformer_engine.common.CuTeDSL.cast.mxfp8.requantize_mxfp8 import GroupedRequantize
from benchmark_mxfp8_requantize import quantize_reference


def main():
    """Warm the compiled specialization and mark one launch for profiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--groups", type=int, default=256)
    args = parser.parse_args()
    rows, hidden, groups = args.rows, args.hidden, args.groups
    if groups <= 0 or rows % groups or (rows // groups) % 128:
        parser.error("Uniform groups must each have a positive 128-aligned row count")
    torch.manual_seed(1234)
    torch.cuda.set_stream(torch.cuda.Stream())
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    powers = torch.randint(-8, 9, (rows, hidden // 32), device="cuda")
    source *= torch.exp2(powers.float()).repeat_interleave(32, dim=1)
    data, scales = quantize_reference(source)
    offsets = torch.arange(groups + 1, device="cuda", dtype=torch.int64) * (rows // groups) * hidden
    dst = torch.empty_like(data)
    row_scales = torch.empty_like(scales).flatten()
    column_scales = torch.empty_like(row_scales)
    tensors = (data, scales, offsets, dst, row_scales, column_scales)
    kernel_args = tuple(from_dlpack(tensor, assumed_align=16) for tensor in tensors) + (
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    kernel = GroupedRequantize(
        hidden,
        groups,
        rows,
        sm_count=torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
    )
    compiled = cute.compile(kernel, *kernel_args)
    for _ in range(5):
        compiled(*kernel_args)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    compiled(*kernel_args)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
