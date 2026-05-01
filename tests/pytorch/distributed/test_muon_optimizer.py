# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for distributed Muon optimizer."""

import os
import subprocess
from pathlib import Path

import pytest
import torch

from transformer_engine.pytorch.optimizers.muon import MuonOptimizer

MULTI_GPU_AVAILABLE = torch.cuda.device_count() >= 2
requires_multi_gpu = pytest.mark.skipif(
    not MULTI_GPU_AVAILABLE,
    reason="Muon optimizer distributed tests require at least 2 GPUs.",
)

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS = torch.cuda.device_count()
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def _run_test(dtype: str, partition_dim: int, weight_decay_mode: str) -> None:
    test_path = TEST_ROOT / "run_muon_optimizer.py"
    test_cmd = LAUNCH_CMD + [
        str(test_path),
        f"--dtype={dtype}",
        f"--partition-dim={partition_dim}",
        f"--weight-decay-mode={weight_decay_mode}",
    ]
    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False, timeout=300)
    if (
        result.returncode != 0
        or "MUON OPTIMIZER CHECK FAILED" in result.stderr.decode()
        or "MUON OPTIMIZER CHECK PASSED" not in result.stdout.decode()
    ):
        raise AssertionError(
            "Muon optimizer test failed.\n"
            f"stdout: {result.stdout.decode()}\n"
            f"stderr: {result.stderr.decode()}"
        )


@requires_multi_gpu
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("partition_dim", [0, 1])
def test_muon_optimizer_matches_reference(dtype: str, partition_dim: int) -> None:
    """Compare distributed Muon updates with a full-matrix reference."""
    _run_test(dtype, partition_dim, "decoupled")


@requires_multi_gpu
def test_muon_optimizer_l2_weight_decay() -> None:
    """Exercise the L2 weight decay branch against the same reference."""
    _run_test("float32", 1, "l2")


def test_muon_optimizer_requires_explicit_process_group() -> None:
    """Muon should not silently fall back to the world process group."""
    param = torch.nn.Parameter(torch.empty(2, 2))
    with pytest.raises(ValueError, match="explicit NCCL tensor-parallel process_group"):
        MuonOptimizer([param], process_group=None, partition_dim=0)


def test_muon_optimizer_resolves_partition_dim_per_parameter() -> None:
    """TE tensor-parallel metadata should provide per-parameter partition dims."""
    param = torch.empty(2, 2)
    param.partition_dim = 0

    assert MuonOptimizer._resolve_partition_dim(param, None) == 0

    param_without_metadata = torch.empty(2, 2)
    assert MuonOptimizer._resolve_partition_dim(param_without_metadata, 1) == 1

    with pytest.raises(ValueError, match="Conflicting partition_dim"):
        MuonOptimizer._resolve_partition_dim(param, 1)

    param.partition_dim = -1
    with pytest.raises(ValueError, match="Non-parallel parameters are not supported"):
        MuonOptimizer._resolve_partition_dim(param, None)
