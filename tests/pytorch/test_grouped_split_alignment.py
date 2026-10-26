# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Device-side alignment checks in graph-safe grouped offset scans."""

import os
import subprocess
import sys

import pytest
import torch

import transformer_engine.pytorch  # Load the common library before the extension.
import transformer_engine_torch as tex


STRIDES = [1, 1, 128, 256, 64, 512, 2, 4, 8]


@pytest.mark.parametrize("num_outputs,alignment", [(1, 128), (4, 128), (6, 256), (9, 256)])
def test_multi_offsets_alignment_graph_replay(num_outputs, alignment):
    """Every scaled output follows dynamic aligned splits, including a second kernel chunk."""
    splits = torch.tensor([256, 256], device="cuda", dtype=torch.int64)
    device = torch.device("cuda")
    strides = STRIDES[:num_outputs]
    leading_zeros = [False] + [True] * (num_outputs - 1)
    dtypes = [torch.int32] + [torch.int64] * (num_outputs - 1)

    def scan():
        return tex.splits_to_offsets_multi(
            splits,
            device,
            strides=strides,
            include_leading_zero=leading_zeros,
            dtypes=dtypes,
            bulk_allocate=num_outputs > 1,
            split_alignment=alignment,
        )

    scan()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, outputs = scan()

    sizes_to_replay = ([256, 256], [0, 512])
    if alignment == 128:
        sizes_to_replay += ([128, 384],)
    for sizes in sizes_to_replay:
        splits.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int64))
        graph.replay()
        for output, stride, leading_zero, dtype in zip(outputs, strides, leading_zeros, dtypes):
            expected = [sizes[0], sum(sizes)]
            if leading_zero:
                expected.insert(0, 0)
            torch.testing.assert_close(
                output, torch.tensor(expected, device="cuda", dtype=dtype) * stride
            )


@pytest.mark.parametrize(
    "num_outputs,alignment,num_splits,bad_index,bad_value",
    [(4, 128, 2, 0, 64), (6, 256, 2, 0, 128), (6, 256, 2, 0, -256), (6, 256, 257, 256, 128)],
)
def test_multi_offsets_alignment_asserts_on_replay(
    num_outputs, alignment, num_splits, bad_index, bad_value
):
    """Invalid graph replays trap on device without poisoning the pytest CUDA context."""
    script = """
import os
import torch
import transformer_engine.pytorch
import transformer_engine_torch as tex

num_outputs = int(os.environ['NUM_OUTPUTS'])
alignment = int(os.environ['ALIGNMENT'])
num_splits = int(os.environ['NUM_SPLITS'])
bad_index = int(os.environ['BAD_INDEX'])
bad_value = int(os.environ['BAD_VALUE'])
splits = torch.full((num_splits,), 256, device='cuda', dtype=torch.int64)
kwargs = dict(strides=[1] * num_outputs, include_leading_zero=[True] * num_outputs,
              dtypes=[torch.int64] * num_outputs, bulk_allocate=True,
              split_alignment=alignment)
def scan():
    return tex.splits_to_offsets_multi(splits, torch.device('cuda'), **kwargs)
scan()
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    captured = scan()
splits[bad_index] = bad_value
graph.replay()
torch.cuda.synchronize()
"""
    env = os.environ.copy()
    env.update(
        NUM_OUTPUTS=str(num_outputs),
        ALIGNMENT=str(alignment),
        NUM_SPLITS=str(num_splits),
        BAD_INDEX=str(bad_index),
        BAD_VALUE=str(bad_value),
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode != 0
    assert "split_alignment" in result.stdout + result.stderr
