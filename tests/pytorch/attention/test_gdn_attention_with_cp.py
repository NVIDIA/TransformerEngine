# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Contiguous GDN CP: host boundaries, common arithmetic, and multi-GPU numerics."""

import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import pytest
import torch

from test_gdn_attention import _GDN_AVAILABLE
import transformer_engine
from transformer_engine.common.linear_attention import build_linear_attention_partition
import transformer_engine_torch as tex


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_partition_intersections(size):
    """Recover all document intervals without dropping tokens or crossing documents."""
    rng = random.Random(301)
    for _ in range(100):
        lengths = [rng.randrange(0, 20) for _ in range(rng.randrange(1, 10))]
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        partitions = [build_linear_attention_partition(cu, size, rank) for rank in range(size)]
        for doc, (start, end) in enumerate(zip(cu, cu[1:])):
            fragments = [f for p in partitions for f in p.fragments if f.document == doc]
            assert sum(f.end - f.start for f in fragments) == end - start
            assert all(start <= f.start < f.end <= end for f in fragments)
            assert all(a.end == b.start for a, b in zip(fragments, fragments[1:]))
        for rank, partition in enumerate(partitions):
            assert (
                partition.local_cu_seqlens[-1]
                == (rank + 1) * cu[-1] // size - rank * cu[-1] // size
            )
            assert len(partition.incoming) <= 1
            assert len(partition.outgoing) <= 1
            for f, forward, backward in zip(
                partition.fragments, partition.forward_descriptors, partition.backward_descriptors
            ):
                assert forward[2] == rank - f.first_rank
                assert backward[2] == f.last_rank - rank
                # Every referenced recipe belongs to this same document.
                for predecessor in range(f.first_rank, rank):
                    p = partitions[predecessor]
                    if p.fragments:
                        assert p.fragments[p.outgoing[0]].document == f.document
                for successor in range(rank + 1, f.last_rank + 1):
                    p = partitions[successor]
                    # An empty rank contributes an identity recipe.
                    if p.fragments:
                        assert p.fragments[p.incoming[0]].document == f.document


@pytest.mark.parametrize("documents", [[1, 2], [0], [0, 2, 1]])
def test_invalid_documents(documents):
    """Reject malformed global offsets before computing communication descriptors."""
    with pytest.raises(ValueError):
        build_linear_attention_partition(documents, 4, 0)


@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("length", [0, 1, 5, 9])
def test_dense_partition_intersections(size, length):
    """Dense rows retain their document identity, including uneven and empty ranks."""
    partitions = [
        build_linear_attention_partition([0, length, 2 * length], size, rank, dense=True)
        for rank in range(size)
    ]
    for doc in (0, 1):
        spans = [f for p in partitions for f in p.fragments if f.document == doc]
        assert sum(f.end - f.start for f in spans) == length
        assert all(a.end == b.start for a, b in zip(spans, spans[1:]))
        if length:
            assert spans[0].start == doc * length
            assert spans[-1].end == (doc + 1) * length
    for rank, partition in enumerate(partitions):
        local_length = (rank + 1) * length // size - rank * length // size
        assert partition.local_cu_seqlens[-1] == 2 * local_length
        assert partition.slots == 2
        assert len(partition.fragments) == (2 if local_length else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("key_dim", [3, 64, 128, 257])
def test_common_affine_prefix(key_dim):
    """Nonsymmetric transitions, reversed order, zero prefixes, and independent seeds."""
    torch.manual_seed(22)
    current = torch.cuda.current_device()
    # Also exercise the binding's device guard: allocating on another GPU must
    # neither launch on the current GPU nor change the caller's current device.
    device = torch.device(
        "cuda", (current + 1) % torch.cuda.device_count() if key_dim == 3 else current
    )
    seed = torch.randn(3, 2, 5, key_dim, device=device)
    additive = torch.randn(6, 2, 5, key_dim, device=device)
    transitions = torch.randn(6, 2, key_dim, key_dim, device=device) / key_dim**0.5
    rows = [
        [0, 0, 0, 1],
        [1, 0, 3, 1],
        [2, 5, 3, -2],
        [-1, 0, 0, 1],
        [-1, 0, 3, 1],
        [-1, 5, 3, -2],
    ]
    descriptors = torch.tensor(rows, device=device, dtype=torch.int32)
    actual = tex.linear_attention_affine_prefix(seed, additive, transitions, descriptors)
    assert torch.cuda.current_device() == current
    expected = []
    for source, first, count, stride in rows:
        x = (
            torch.zeros_like(seed[0], dtype=torch.float64)
            if source == -1
            else seed[source].double()
        )
        for step in range(count):
            index = first + step * stride
            x = x @ transitions[index].double() + additive[index].double()
        expected.append(x)
    torch.testing.assert_close(actual.double(), torch.stack(expected), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("size", [2, 4])
def test_gdn_with_cp(size):
    """Run genuine NCCL CP with all token and state gradients checked numerically."""
    if not _GDN_AVAILABLE or torch.cuda.device_count() < size:
        if os.getenv("NVTE_GDN_CP_TEST_REQUIRED", "0") == "1":
            pytest.fail(f"Required GDN CP test needs FE GDN and {size} CUDA GPUs")
        pytest.skip(f"GDN CP test requires FE GDN and {size} CUDA GPUs")
    runner = Path(__file__).with_name("run_gdn_with_cp.py")
    env = os.environ.copy()
    # torchrun starts a script, so its sys.path[0] is the tests directory.
    # Preserve the TE package that pytest imported, including source worktrees.
    package_root = str(Path(transformer_engine.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join((package_root, env.get("PYTHONPATH", "")))
    with tempfile.TemporaryDirectory(prefix="te-gdn-cp-") as log_dir:
        log_path = Path(log_dir) / "worker.log"
        print(f"CP={size} progress log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={size}",
                    str(runner),
                ],
                check=False,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
                env=env,
            )
        output = log_path.read_text()
        assert result.returncode == 0, output
        print(output)
