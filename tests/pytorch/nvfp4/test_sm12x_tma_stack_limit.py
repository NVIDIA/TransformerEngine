# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression test: TMA cast kernels must not raise the CUDA stack limit on SM120/SM121.

On sm_12x, ptxas lowers a cp.async.bulk.tensor load with a .shared::cluster destination to a
run-time branch whose remote-CTA side is a driver syscall. The first launch of a kernel that
contains it raises the context's per-thread stack limit (1 KB -> ~14.6 KB), and the driver keeps
the matching local memory reserved for the lifetime of the context. The limit is per context and
never shrinks, so each check runs in a fresh process.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

import transformer_engine.pytorch as te

nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

_CHILD = textwrap.dedent(
    """
    import ctypes
    import sys

    import torch
    import transformer_engine.pytorch as te

    libcuda = ctypes.CDLL("libcuda.so.1")


    def stack_limit():
        torch.cuda.synchronize()
        value = ctypes.c_size_t()
        assert libcuda.cuCtxGetLimit(ctypes.byref(value), 0) == 0  # CU_LIMIT_STACK_SIZE
        return value.value


    x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
    quantizer = te.NVFP4Quantizer(
        rowwise=True, columnwise=True, with_2d_quantization=sys.argv[1] == "2d"
    )
    before = stack_limit()
    quantizer(x)
    print(before, stack_limit(), flush=True)
    """
)


def _is_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


@pytest.mark.skipif(not _is_sm12x(), reason="Only SM120/SM121 lower cluster TMA loads to a syscall")
@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
@pytest.mark.parametrize("scaling", ["1d", "2d"])
def test_nvfp4_cast_keeps_stack_limit(scaling: str) -> None:
    """The first NVFP4 cast in a process must not raise the per-thread stack limit."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, scaling], capture_output=True, text=True, check=False
    )
    fields = result.stdout.split()
    assert len(fields) >= 2, f"child failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
    before, after = (int(v) for v in fields[-2:])
    assert after == before, (
        f"The first {scaling} NVFP4 cast raised the per-thread stack limit from {before} to"
        f" {after} bytes"
    )
