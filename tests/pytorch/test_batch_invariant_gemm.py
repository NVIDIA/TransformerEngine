# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""A row's output must not depend on which other rows were in the batch.

The general GEMM path lets cuBLASLt pick a kernel from the full problem shape, so
the same row can produce different bits depending on the batch it arrives in.
``batch_invariant_gemm`` fixes the tile geometry and the reduction order so the
result is a function of the row and the weight only.
"""

import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.batch_invariant_gemm import (
    batch_invariant_gemm,
    is_supported,
)

M, N, K = 256, 192, 128
SLICES = [(0, 1), (0, 7), (5, 29), (128, 256), (0, 256)]


@pytest.fixture(scope="module")
def operands():
    if not torch.cuda.is_available():
        pytest.skip("batch_invariant_gemm requires a GPU")
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    return a, b


def test_rows_are_bitwise_stable_across_batch_composition(operands):
    a, b = operands
    full = batch_invariant_gemm(a, b)
    for lo, hi in SLICES:
        part = batch_invariant_gemm(a[lo:hi].contiguous(), b)
        assert torch.equal(part, full[lo:hi]), (
            f"rows {lo}:{hi} changed when the batch was sliced"
        )


def test_matches_torch_reference(operands):
    a, b = operands
    got = batch_invariant_gemm(a, b)
    ref = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(got, ref, rtol=0, atol=1.0)


def test_out_parameter_is_written(operands):
    a, b = operands
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    assert batch_invariant_gemm(a, b, out=out) is out
    assert torch.equal(out, batch_invariant_gemm(a, b))


def test_unsupported_combinations_raise():
    if not torch.cuda.is_available():
        pytest.skip("batch_invariant_gemm requires a GPU")
    a = torch.randn(8, 16, dtype=torch.float16, device="cuda")
    b = torch.randn(8, 16, dtype=torch.float16, device="cuda")
    assert not is_supported(a, b)
    with pytest.raises(ValueError, match="bfloat16"):
        batch_invariant_gemm(a, b)
