# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
"""Device offset validation runs in children because CUDA errors poison a context."""

import subprocess
import sys

import pytest
import torch
import transformer_engine.pytorch
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor


def _invalid_offsets(case, graph_replay, hidden):
    capacity = 256
    invalid = {
        "origin": [hidden, 128 * hidden, capacity * hidden],
        "negative": [0, -hidden, capacity * hidden],
        "decreasing": [0, 128 * hidden, 64 * hidden],
        "capacity": [0, 128 * hidden, (capacity + 128) * hidden],
        "fractional_row": [0, 128 * hidden + 1, capacity * hidden],
        "unaligned_start": [0, 63 * hidden, capacity * hidden],
    }[case]
    counts = torch.tensor([128, 128], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([0, 128 * hidden, capacity * hidden], device="cuda", dtype=torch.int64)
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3, rowwise=True, columnwise=True,
        block_scaling_dim=1, force_pow_2_scales=True,
    )
    grouped = GroupedTensor(
        (capacity, hidden), torch.bfloat16, num_tensors=2, quantizer=quantizer,
        data=torch.zeros(capacity * hidden, device="cuda", dtype=torch.uint8),
        scale_inv=torch.ones(capacity * (hidden // 128), device="cuda"), first_dims=counts,
        tensor_offsets=offsets, with_gemm_swizzled_scales=True,
    )
    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            tex.group_requantize_inplace(
                grouped, quantizer, 2, counts, tex.DType.kBFloat16, offsets
            )
    offsets.copy_(torch.tensor(invalid, device="cuda", dtype=torch.int64))
    if graph_replay:
        graph.replay()
    else:
        tex.group_requantize_inplace(
            grouped, quantizer, 2, counts, tex.DType.kBFloat16, offsets
        )
    torch.cuda.synchronize()
    raise AssertionError("Invalid device offsets were not rejected")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="EP block128 requant requires Hopper or newer",
)
@pytest.mark.parametrize("case", [
    "origin", "negative", "decreasing", "capacity", "fractional_row", "unaligned_start"
])
@pytest.mark.parametrize("graph_replay", [False, True])
@pytest.mark.parametrize("hidden", [128, 384])
def test_invalid_device_offsets(case, graph_replay, hidden):
    child = subprocess.run(
        [sys.executable, __file__, case, str(int(graph_replay)), str(hidden)],
        capture_output=True, text=True, timeout=60,
    )
    assert child.returncode != 0
    assert "Block scaling grouped requant invalid offsets" in child.stdout + child.stderr


if __name__ == "__main__":
    _invalid_offsets(sys.argv[1], bool(int(sys.argv[2])), int(sys.argv[3]))
