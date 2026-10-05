# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuBLAS workspace ownership and numerical checks for independent CUDA streams."""

import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm, get_cublas_workspace


@pytest.mark.parametrize("ub,grouped_gemm", [(False, False), (True, False), (False, True)])
def test_workspace_is_reused_only_on_its_stream(ub, grouped_gemm):
    """A workspace remains reusable without aliasing another stream's active scratch."""
    device = torch.cuda.current_device()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = []
    for stream in streams:
        with torch.cuda.stream(stream):
            first = get_cublas_workspace(device, ub, grouped_gemm)
            assert get_cublas_workspace(device, ub, grouped_gemm) is first
            workspaces.append(first if grouped_gemm else [first])
    assert len(workspaces[0]) == len(workspaces[1])
    assert {w.data_ptr() for w in workspaces[0]}.isdisjoint(w.data_ptr() for w in workspaces[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_concurrent_gemms_match_exact_reference(dtype):
    """Large reductions exercise cuBLAS algorithms that use scratch for partial sums."""
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    reduction, hidden, experts = 16384, 4096, 128
    inputs = [
        torch.full((reduction, hidden), value, dtype=dtype, device="cuda") for value in (1.0, 2.0)
    ]
    gradients = [
        torch.full((reduction, experts), value, dtype=dtype, device="cuda") for value in (1.0, 3.0)
    ]
    ready = torch.cuda.Event()
    ready.record()
    for stream in streams:
        stream.wait_event(ready)

    outputs = []
    for _ in range(8):
        for stream, inp, grad in zip(streams, inputs, gradients):
            with torch.cuda.stream(stream):
                out, *_ = general_gemm(inp, grad, dtype, layout="NT", grad=True)
                outputs.append(out)
    for stream in streams:
        stream.synchronize()
    # Both operands and the result are exactly representable in either dtype.
    for index, out in enumerate(outputs):
        expected = torch.full_like(out, reduction * (1 if index % 2 == 0 else 6))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
