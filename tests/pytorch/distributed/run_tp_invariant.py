#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVTE_TP_INVARIANT_MODE distributed test body. Launched via test_tp_invariant.py.

One invocation runs ONE (parallel_mode, sequence_parallel, expect_bitwise)
combination per pytest parametrize axis:

  expect_bitwise=True  (NVTE_TP_INVARIANT_MODE=1, with_tp_invariant):
                       TP=N == TP=1 bit-for-bit.
  expect_bitwise=False (NVTE_TP_INVARIANT_MODE=0, without_tp_invariant):
                       TP=N != TP=1 (stock TP isn't bitwise; guards
                       with_tp_invariant against trivial-pass).

Wgrad is intentionally not compared — patches gate only the forward
(row-parallel) and dgrad (column-parallel) paths.

Reuses TestDistributedLinearBase from run_numerics_exact for sharding + gather.

LayerNormLinear with partition_stride=2 (SwiGLU FC1 deinterleave) is covered
by ``_check_tp_invariance_deinterleave`` — positive only (no without_tp_invariant
variant), because stock TP dgrad uses the same products as TP=1 just in a
different accumulation order, so bit-difference is layout-dependent (flaky).
"""

import argparse
import datetime
import os
import sys

import run_numerics_exact as rne
import torch
import torch.distributed as dist
from run_numerics_exact import (
    TestDistributedLayerNormLinearBase,
    TestDistributedLinearBase,
    dist_print,
)

BATCH, HIDDEN, OUT = 16, 256, 128
DTYPE = torch.bfloat16


def _run_linear(parallel_mode, sequence_parallel):
    """Run TP=1 reference and TP=N for given config; return relevant output pair.

    For parallel_mode='row' returns (y_ref, y_tp). For 'column' returns
    (dgrad_ref, dgrad_tp). Both are full (gathered) tensors regardless of
    sharding, suitable for direct bitwise comparison.
    """
    x, w, bias, gradient = TestDistributedLinearBase._prepare_data(
        BATCH, HIDDEN, OUT, use_bias=False, seed=42, dtype=DTYPE,
    )
    y_ref, dgrad_ref, _, _ = TestDistributedLinearBase.run_linear(
        x, w, bias, gradient,
        parallel_mode=None, sequence_parallel=False,
        tp_group=None, tp_size=1, rank=0,
    )
    y_tp, dgrad_tp, _, _ = TestDistributedLinearBase.run_linear(
        x, w, bias, gradient,
        parallel_mode=parallel_mode, sequence_parallel=sequence_parallel,
        tp_group=rne.NCCL_WORLD, tp_size=rne.WORLD_SIZE, rank=rne.WORLD_RANK,
    )
    if parallel_mode == "row":
        return y_ref, y_tp
    return dgrad_ref, dgrad_tp


def _check_tp_invariance(parallel_mode, sequence_parallel, expect_bitwise):
    """Run one check; assert bitwise (with_tp_invariant) or non-bitwise (without_tp_invariant)."""
    os.environ["NVTE_TP_INVARIANT_MODE"] = "1" if expect_bitwise else "0"
    ref, tp = _run_linear(parallel_mode, sequence_parallel)

    if rne.WORLD_RANK != 0:
        return

    kind = "fwd" if parallel_mode == "row" else "dgrad"
    label = f"{parallel_mode}-parallel {kind} sp={int(sequence_parallel)}"

    if expect_bitwise:
        torch.testing.assert_close(
            tp, ref, atol=0, rtol=0,
            msg=f"{label} not bitwise under NVTE_TP_INVARIANT_MODE=1",
        )
        dist_print(f"[with_tp_invariant   ] {label}: TP=1 ≡ TP={rne.WORLD_SIZE} bitwise")
    else:
        assert not torch.equal(tp, ref), (
            f"without_tp_invariant: {label} unexpectedly bitwise under NVTE_TP_INVARIANT_MODE=0"
        )
        dist_print(f"[without_tp_invariant] {label}: TP=1 ≠ TP={rne.WORLD_SIZE} (as expected)")


def _check_tp_invariance_deinterleave(sequence_parallel):
    """LayerNormLinear column-parallel + partition_stride=2 (SwiGLU FC1) TP-invariance.

    Uses MLM's golden stride=2 sharding (added to ``TestDistributedLayerNormLinearBase``)
    to construct per-rank interleaved weight; verifies our deinterleave correctly inverts
    it so TP=N dgrad bitwise matches the TP=1 reference."""
    os.environ["NVTE_TP_INVARIANT_MODE"] = "1"
    x, w, _, g = TestDistributedLinearBase._prepare_data(
        BATCH, HIDDEN, OUT, use_bias=False, seed=42, dtype=DTYPE,
    )
    _, _, dgrad_ref, _, _ = TestDistributedLayerNormLinearBase.run_layernorm_linear(
        x, w, None, g, parallel_mode=None, sequence_parallel=False,
        tp_group=None, tp_size=1, rank=0, partition_stride=1,
    )
    _, _, dgrad_tp, _, _ = TestDistributedLayerNormLinearBase.run_layernorm_linear(
        x, w, None, g, parallel_mode="column", sequence_parallel=sequence_parallel,
        tp_group=rne.NCCL_WORLD, tp_size=rne.WORLD_SIZE, rank=rne.WORLD_RANK,
        partition_stride=2,
    )
    if rne.WORLD_RANK != 0:
        return
    label = f"LN-Linear stride=2 sp={int(sequence_parallel)}"
    torch.testing.assert_close(dgrad_tp, dgrad_ref, atol=0, rtol=0,
                               msg=f"{label}: not TP-invariant")
    dist_print(f"{label}: TP=1 ≡ TP={rne.WORLD_SIZE} bitwise via deinterleave")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-type", choices=["linear", "deinterleave"], default="linear")
    parser.add_argument("--parallel-mode", choices=["row", "column"])
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--expect-bitwise", type=int, choices=[0, 1])
    args = parser.parse_args()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    assert world_size <= torch.cuda.device_count()

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://",
        timeout=datetime.timedelta(seconds=60),
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    rne.WORLD_RANK, rne.WORLD_SIZE = rank, world_size
    rne.NCCL_WORLD = dist.new_group(backend="nccl")

    if args.check_type == "linear":
        assert args.parallel_mode is not None, "--parallel-mode required for linear check"
        assert args.expect_bitwise is not None, "--expect-bitwise required for linear check"
        _check_tp_invariance(
            parallel_mode=args.parallel_mode,
            sequence_parallel=args.sequence_parallel,
            expect_bitwise=bool(args.expect_bitwise),
        )
    else:  # deinterleave
        _check_tp_invariance_deinterleave(sequence_parallel=args.sequence_parallel)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
