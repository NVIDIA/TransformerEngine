# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Check both NVFP4 amax outputs, including reuse with a decreasing maximum."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer


@pytest.mark.parametrize("direction", ["row", "column", "both"])
@pytest.mark.parametrize("shape", [(128, 128), (160, 192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_2d_quantization", [False, True])
def test_nvfp4_amax_matches_input_after_reuse(direction, shape, dtype, with_2d_quantization):
    """The second amax must reset and track the same input maximum as the first."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    q = NVFP4Quantizer(
        rowwise=direction != "column",
        columnwise=direction != "row",
        with_2d_quantization=with_2d_quantization,
    )
    q.optimize_for_gemm = False
    out = q.make_empty(shape, dtype=dtype)
    # The maximum decreases, reaches zero and rises again; stale atomic-max state fails.
    for magnitude in (16.0, 0.5, 0.0, 2.0):
        x = torch.full(shape, -magnitude, device="cuda", dtype=dtype)
        q.update_quantized(x, out)
        expected = x.float().abs().amax().reshape(1)
        for name in ("_amax_rowwise", "_amax_columnwise"):
            actual = getattr(out, name)
            if actual is not None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
