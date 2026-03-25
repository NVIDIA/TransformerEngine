# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch

import transformer_engine.pytorch as te

NUM_PROCS: int = torch.cuda.device_count()
_FSDP2_DIR = Path(__file__).parent.resolve() / "fsdp2_tests"


@pytest.mark.skipif(NUM_PROCS % 2 != 0, reason="Requires even number of GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
def test_fsdp2_model_tests():
    """All FSDP2 model tests (parametrized internally by recipe, fp8_init, sharding, layer)."""
    test_path = _FSDP2_DIR / "run_fsdp2_model.py"
    result = subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={NUM_PROCS}",
            "--local-ranks-filter=0",
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "-s",
            "--tb=short",
        ],
        env=os.environ,
        timeout=600,
    )
    assert result.returncode in (0, 5), f"Inner pytest failed with exit code {result.returncode}"


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
def test_fsdp2_fused_adam_tests():
    """All FSDP2 FusedAdam tests (parametrized internally by recipe, test variant)."""
    test_path = _FSDP2_DIR / "run_fsdp2_fused_adam.py"
    nproc = min(NUM_PROCS, 2)
    result = subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "--local-ranks-filter=0",
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "-s",
            "--tb=short",
        ],
        env=os.environ,
        timeout=600,
    )
    assert result.returncode in (0, 5), f"Inner pytest failed with exit code {result.returncode}"


@pytest.mark.skipif(NUM_PROCS < 4, reason="Requires 4+ GPUs for DP4→DP2 resharding test")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
def test_fsdp2_fused_adam_dcp_resharding():
    """DCP checkpoint saved with DP4 loads correctly into DP2 (cross-topology resharding).

    Runs two sequential torchrun invocations against run_fsdp2_fused_adam.py:
      1. nproc=4  →  test_dcp_resharding_save  (train + write checkpoint + ref output)
      2. nproc=2  →  test_dcp_resharding_load  (load checkpoint, assert output parity)
    """
    test_path = _FSDP2_DIR / "run_fsdp2_fused_adam.py"

    # Phase 1: save checkpoint with 4 ranks.
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=4",
            "--local-ranks-filter=0",
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "-s",
            "--tb=short",
            "-k",
            "dcp_resharding_save",
        ],
        env=os.environ,
        timeout=300,
    )
    assert result.returncode in (0, 5), f"DCP resharding save phase failed: {result.returncode}"

    # Phase 2: load checkpoint with 2 ranks (different topology).
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=2",
            "--local-ranks-filter=0",
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "-s",
            "--tb=short",
            "-k",
            "dcp_resharding_load",
        ],
        env=os.environ,
        timeout=300,
    )
    assert result.returncode in (0, 5), f"DCP resharding load phase failed: {result.returncode}"


def test_dummy() -> None:
    """Dummy test

    pytest returns exit code 5 if all tests are skipped.

    """
    pass
