# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""pytest entry for the NVTE_TP_INVARIANT_MODE distributed correctness test.

Launches ``run_tp_invariant.py`` under torchrun with per-case CLI args and asserts
the subprocess returns 0. The actual bitwise checks live in ``run_tp_invariant.py``.

Test matrix (per tp_size):
  parallel_mode   ∈ {row, column}
  sequence_parallel ∈ {False, True}
  expect_bitwise  ∈ {True (with_tp_invariant, patches on),
                     False (without_tp_invariant, patches off)}
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch

if torch.cuda.device_count() < 2:
    pytest.skip(
        "TP-invariant test requires at least 2 GPUs.",
        allow_module_level=True,
    )


TEST_ROOT = Path(__file__).parent.resolve()


def _tp_sizes():
    """TP sizes worth exercising on this node (2 and, if available, 4)."""
    n = torch.cuda.device_count()
    sizes = [2]
    if n >= 4:
        sizes.append(4)
    return sizes


@pytest.mark.parametrize("tp_size", _tp_sizes())
@pytest.mark.parametrize("parallel_mode", ["row", "column"])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("expect_bitwise", [True, False],
                         ids=["with_tp_invariant", "without_tp_invariant"])
def test_tp_invariant(tp_size, parallel_mode, sequence_parallel, expect_bitwise):
    """One TP-invariant correctness check per parameter combination.

    expect_bitwise=True  → NVTE_TP_INVARIANT_MODE=1, TP=N must equal TP=1 bit-for-bit.
    expect_bitwise=False → NVTE_TP_INVARIANT_MODE=0, TP=N must DIFFER from TP=1
                           (without_tp_invariant, guards against trivial-pass)."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={tp_size}",
        str(TEST_ROOT / "run_tp_invariant.py"),
        "--check-type", "linear",
        "--parallel-mode", parallel_mode,
        "--expect-bitwise", str(int(expect_bitwise)),
    ]
    if sequence_parallel:
        cmd.append("--sequence-parallel")

    result = subprocess.run(cmd, env=os.environ, check=False)
    assert result.returncode == 0, (
        f"run_tp_invariant.py failed: tp_size={tp_size}, parallel_mode={parallel_mode}, "
        f"sequence_parallel={sequence_parallel}, expect_bitwise={expect_bitwise} "
        f"(returncode={result.returncode})"
    )


@pytest.mark.parametrize("tp_size", _tp_sizes())
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_tp_invariant_deinterleave(tp_size, sequence_parallel):
    """LayerNormLinear column-parallel + partition_stride=2 (SwiGLU FC1 layout): dgrad bitwise
    matches TP=1 reference. Uses MLM's golden stride=2 sharding to construct per-rank
    interleaved weight; verifies our deinterleave correctly inverts it."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={tp_size}",
        str(TEST_ROOT / "run_tp_invariant.py"),
        "--check-type", "deinterleave",
    ]
    if sequence_parallel:
        cmd.append("--sequence-parallel")
    result = subprocess.run(cmd, env=os.environ, check=False)
    assert result.returncode == 0, (
        f"deinterleave failed: tp_size={tp_size}, sequence_parallel={sequence_parallel}"
    )
