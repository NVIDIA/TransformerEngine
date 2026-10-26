# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Device-side split checks used by graph-safe grouped quantization."""

import os
import subprocess
import sys

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer


available, unavailable_reason = te.is_mxfp8_available(return_reason=True)


def _quantizer():
    return MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=False)


@pytest.mark.skipif(not available, reason=unavailable_reason)
def test_group_quantize_split_alignment_graph_replay():
    """Changing device splits between replays keeps aligned calls graph-safe."""
    x = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    splits = torch.tensor([256, 256], device="cuda", dtype=torch.int64)
    quantizer = _quantizer()

    tex.group_quantize(x, quantizer, 2, splits, split_alignment=128)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = tex.group_quantize(x, quantizer, 2, splits, split_alignment=128)

    for sizes in ([256, 256], [128, 384], [0, 512]):
        splits.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int64))
        graph.replay()
        torch.cuda.synchronize()
    assert output is not None


@pytest.mark.skipif(not available, reason=unavailable_reason)
@pytest.mark.parametrize("bad_metadata", ["splits", "offsets"])
def test_group_quantize_split_alignment_asserts_on_replay(bad_metadata):
    """An invalid replay fails on device without poisoning the pytest CUDA context."""
    script = """
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

x = torch.randn(512, 128, device='cuda', dtype=torch.bfloat16)
splits = torch.tensor([256, 256], device='cuda', dtype=torch.int64)
offsets = torch.tensor([0, 256 * 128, 512 * 128], device='cuda', dtype=torch.int64)
quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=False)
tex.group_quantize(x, quantizer, 2, splits, tensor_offsets=offsets, split_alignment=256)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = tex.group_quantize(
        x, quantizer, 2, splits, tensor_offsets=offsets, split_alignment=256
    )
if BAD_METADATA == 'splits':
    splits.copy_(torch.tensor([128, 384], device='cuda', dtype=torch.int64))
else:
    offsets.copy_(torch.tensor([0, 128 * 128, 512 * 128], device='cuda', dtype=torch.int64))
graph.replay()
torch.cuda.synchronize()
""".replace(
        "BAD_METADATA", repr(bad_metadata)
    )
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode != 0
    assert "split_alignment" in result.stdout + result.stderr


def test_grouped_offset_scan_alignment_graph_replay():
    """The offset scan validates dynamic splits without an additional kernel."""
    splits = torch.tensor([256, 256], device="cuda", dtype=torch.int64)
    device = torch.device("cuda")

    def scan():
        return tex.splits_to_offsets_multi(
            splits,
            device,
            strides=[1],
            include_leading_zero=[True],
            dtypes=[torch.int64],
            split_alignment=128,
        )

    scan()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, outputs = scan()

    for sizes, expected in (([256, 256], [0, 256, 512]), ([128, 384], [0, 128, 512])):
        splits.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int64))
        graph.replay()
        torch.testing.assert_close(
            outputs[0], torch.tensor(expected, device="cuda", dtype=torch.int64)
        )


def test_grouped_offset_scan_alignment_asserts_on_replay():
    """The shared grouped-linear offset kernel traps on an invalid graph replay."""
    script = """
import torch
import transformer_engine.pytorch
import transformer_engine_torch as tex

splits = torch.tensor([256, 256], device='cuda', dtype=torch.int64)
device = torch.device('cuda')
def scan():
    return tex.splits_to_offsets_multi(
        splits, device, strides=[1], include_leading_zero=[True],
        dtypes=[torch.int64], split_alignment=256
    )
scan()
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    outputs = scan()
splits.copy_(torch.tensor([128, 384], device='cuda', dtype=torch.int64))
graph.replay()
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode != 0
    assert "split_alignment" in result.stdout + result.stderr
