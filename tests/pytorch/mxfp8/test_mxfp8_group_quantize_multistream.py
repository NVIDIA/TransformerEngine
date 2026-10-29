# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression test for NVIDIA/TransformerEngine#3630.

Grouped MXFP8 quantization stored its per-tensor TMA descriptor storage in a
file-global ``static __device__`` buffer. Because that buffer is shared by every launch of the
kernel, two ``group_quantize`` calls running on different CUDA streams raced on the same
descriptors: one stream's ``update_tma_descriptors`` overwrote tensor maps the other stream was
still consuming, producing incorrect output.

The fix allocates a private ``TensorMapStorage`` per invocation with ``cudaMallocAsync`` and
passes it to both the descriptor-update and consumer kernels, so concurrent invocations never
share descriptor state. This test exercises concurrent multi-stream grouped quantization and
checks every result against a single-stream reference.
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)

# VARYING_LAST_DIM layout: 2D input is [rows, sum(last_dims)] and each group is a column slice.
NUM_GROUPS = 8
ROWS = 4096
COLS_PER_GROUP = 1024


def _make_input(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        ROWS, NUM_GROUPS * COLS_PER_GROUP, device="cuda", dtype=torch.bfloat16, generator=g
    )


def _last_dims():
    return torch.full((NUM_GROUPS,), COLS_PER_GROUP, dtype=torch.int64, device="cuda")


def _quantize(x):
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    return tex.group_quantize(x, quantizer, NUM_GROUPS, None, _last_dims())


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("iteration", range(8))
def test_group_quantize_multistream_matches_reference(iteration):
    """Concurrent multi-stream grouped quantize must equal the single-stream reference."""
    x1 = _make_input(iteration)
    x2 = _make_input(1_000_000 + iteration)

    # Deterministic single-stream references.
    ref1 = _quantize(x1)
    ref2 = _quantize(x2)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    with torch.cuda.stream(s1):
        out1 = _quantize(x1)
    with torch.cuda.stream(s2):
        out2 = _quantize(x2)

    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)

    assert torch.equal(out1.rowwise_data, ref1.rowwise_data)
    assert torch.equal(out1.scale_inv, ref1.scale_inv)
    assert torch.equal(out2.rowwise_data, ref2.rowwise_data)
    assert torch.equal(out2.scale_inv, ref2.scale_inv)
