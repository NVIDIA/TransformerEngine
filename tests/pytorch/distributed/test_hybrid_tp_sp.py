# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest driver for hybrid quantization TP/SP distributed tests.

Launches ``run_hybrid_tp_sp.py`` via ``torchrun --nproc_per_node=N``
and asserts a zero exit code. Rank-level numerical checks are performed
inside ``run_hybrid_tp_sp.py`` and propagated via ``dist.all_reduce``
with ``ReduceOp.MAX`` on a failure flag, so a failure on any rank
fails the whole subprocess (and thus the pytest assertion).

Mirrors the ``test_numerics.py`` ↔ ``run_numerics.py`` split pattern but
scoped to hybrid recipes only. Isolated from the main ``run_numerics.py``
harness so that adding hybrid-specific cases here doesn't perturb the
larger vanilla-recipe test matrix.
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te


if torch.cuda.device_count() < 2:
    pytest.skip(
        "Distributed TP/SP tests need at least 2 GPUs.",
        allow_module_level=True,
    )

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS = min(2, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def _run_test(quantization: str, test: str = "all"):
    script = TEST_ROOT / "run_hybrid_tp_sp.py"
    cmd = LAUNCH_CMD + [str(script), "--quantization", quantization, "--test", test]
    result = subprocess.run(cmd, env=os.environ, check=False)
    assert result.returncode == 0, (
        f"run_hybrid_tp_sp.py (quantization={quantization}, test={test})"
        f" exited with code {result.returncode}"
    )


# ──────────────────────────────────────────────────────────────────────
# Hybrid FP8 current scaling (rowwise + columnwise same format)
# ──────────────────────────────────────────────────────────────────────
#
# FP8 current scaling is stateless per-tensor (amax computed from the
# live tensor) and therefore uses amax reduction across TP ranks when
# sequence parallelism is on. This is the tightest integration of
# hybrid with the distributed codepath: each rank computes an
# independent amax, reduces it across TP group, then both sub-
# quantizers (rowwise + columnwise) use the same reduced scale. If
# ``HybridQuantizer`` mis-plumbs the amax reduction through its two
# inner quantizers, we'd see numerical drift vs single-node.


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
def test_hybrid_fp8_linear():
    """TP column + row × SP on/off for ``te.Linear`` under hybrid FP8
    current-scaling. Fine-grained: this runs first (cheapest, most
    likely to surface TP-path hybrid bugs) so a failure here tells us
    to stop before the more-expensive TransformerLayer run."""
    _run_test("hybrid_fp8", "linear")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
def test_hybrid_fp8_layernorm_linear():
    """Column-parallel ``te.LayerNormLinear`` with and without SP.
    Probes the all-gather-before-quantize path that
    ``layernorm_linear.py`` disables the fused norm for when
    ``isinstance(input_quantizer, HybridQuantizer)``."""
    _run_test("hybrid_fp8", "layernorm_linear")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
def test_hybrid_fp8_transformer_layer():
    """Full ``te.TransformerLayer`` with ``set_parallel_mode=True`` and
    SP on/off. Integration check hitting LayerNormLinear(QKV) → DPA →
    LayerNormMLP → row-parallel output projection all under hybrid
    FP8."""
    _run_test("hybrid_fp8", "transformer_layer")


# ──────────────────────────────────────────────────────────────────────
# Hybrid MXFP8 (rowwise + columnwise same format)
# ──────────────────────────────────────────────────────────────────────
#
# MXFP8 is per-block (32-element microblocks), stateless, no amax
# reduction. Simpler distributed behaviour than FP8 current scaling,
# but exercises the ``[128, 4]`` / ``[4, 128]`` scale alignment padding
# through TP shards (each rank sees its own dim-0 slice which may not
# be a multiple of 128).


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_linear():
    _run_test("hybrid_mxfp8", "linear")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_layernorm_linear():
    _run_test("hybrid_mxfp8", "layernorm_linear")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_transformer_layer():
    _run_test("hybrid_mxfp8", "transformer_layer")


# ──────────────────────────────────────────────────────────────────────
# Hybrid NVFP4 (rowwise + columnwise same format, 1D block scaling)
# ──────────────────────────────────────────────────────────────────────
#
# NVFP4 is the Rubin-era target recipe: 4-bit data (E2M1) with FP8 block
# scales on 16-element microblocks. The default ``NVFP4Quantizer()`` is
# 1D block scaling only — no RHT, no stochastic rounding, no 2D block
# scaling — matching upstream ``run_numerics.py::nvfp4_vanilla()``.
# Those more-sophisticated knobs are orthogonal to hybrid composition
# and can be layered in separately once baseline distributed NVFP4
# hybrid is stable.
#
# Unlike FP8 current scaling, NVFP4 does not reduce amax across TP ranks
# (block-level scales are computed per-microblock locally), so SP
# amax-reduction issues don't apply. The tight interaction to watch is
# the packed FP4 dim-0 alignment in the TP shard — each rank sees a
# weight slice that may not be naturally aligned to the NVFP4 block
# boundary, and hybrid quantizes twice (rowwise + columnwise) on that
# shard.


@pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
def test_hybrid_nvfp4_linear():
    _run_test("hybrid_nvfp4", "linear")


@pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
def test_hybrid_nvfp4_layernorm_linear():
    _run_test("hybrid_nvfp4", "layernorm_linear")


@pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
def test_hybrid_nvfp4_transformer_layer():
    _run_test("hybrid_nvfp4", "transformer_layer")
