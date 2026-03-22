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
    )
    assert result.returncode in (0, 5), (
        f"Inner pytest failed with exit code {result.returncode}"
    )


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
    )
    assert result.returncode in (0, 5), (
        f"Inner pytest failed with exit code {result.returncode}"
    )


def test_dummy() -> None:
    """Dummy test

    pytest returns exit code 5 if all tests are skipped.

    """
    pass
