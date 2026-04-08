# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import run_distributed

import pytest
import torch

import transformer_engine.pytorch as te

NUM_PROCS: int = torch.cuda.device_count()
_FSDP2_DIR = Path(__file__).parent.resolve() / "fsdp2_tests"

# Import some utilities from PyTest-owned conftest.py.
sys.path.insert(0, str(_FSDP2_DIR))
from conftest import _parametrize_recipes

sys.path.pop(0)


@pytest.mark.skipif(NUM_PROCS % 2 != 0, reason="Requires even number of GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
def test_fsdp2_model_tests():
    """All FSDP2 model tests (parametrized internally by recipe, fp8_init, sharding, layer)."""
    test_path = _FSDP2_DIR / "run_fsdp2_model.py"
    run_distributed(
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
        valid_returncodes=(0, 5),
        env=os.environ,
        timeout=600,
    )


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
def test_fsdp2_fused_adam_tests():
    """All FSDP2 FusedAdam tests (parametrized internally by recipe, test variant)."""
    test_path = _FSDP2_DIR / "run_fsdp2_fused_adam.py"
    nproc = min(NUM_PROCS, 2)
    run_distributed(
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
            # The following 2 tests need to be run in sequence,
            # as they depend on each other.
            "-k",
            "not dcp_resharding_save and not dcp_resharding_load",
        ],
        valid_returncodes=(0, 5),
        env=os.environ,
        timeout=600,
    )


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
def test_fsdp2_mem_leak_tests():
    """FSDP2 memory leak detection tests (parametrized internally by recipe, quantized_model_init)."""
    test_path = _FSDP2_DIR / "run_fsdp2_mem_leak.py"
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
@pytest.mark.parametrize("recipe", _parametrize_recipes())
def test_fsdp2_fused_adam_dcp_resharding(recipe):
    """DCP checkpoint saved with DP4 loads correctly into DP2 (cross-topology resharding).

    Runs two sequential torchrun invocations against run_fsdp2_fused_adam.py:
      1. nproc=4  →  dcp_resharding_save  (train + write checkpoint + ref output)
      2. nproc=2  →  dcp_resharding_load  (load checkpoint, assert output parity)
    """
    if recipe == "MXFP8BlockScaling":
        pytest.xfail(
            "MXFP8BlockScaling: FusedAdam CUDA kernel does not support "
            "MXFP8 quantized tensors, causing illegal memory access. "
            "Fixed by https://github.com/NVIDIA/TransformerEngine/pull/2789."
        )
    if recipe == "NVFP4BlockScaling":
        pytest.xfail(
            "NVFP4BlockScaling: DCP load_state_dict triggers reset_sharded_param() "
            "which calls data_ptr() on NVFP4Tensor wrapper subclass with invalid storage"
        )
    if recipe == "Float8BlockScaling":
        pytest.xfail(
            "Float8BlockScaling doesnt work for DCP resharding with scale inv padding "
            "not being handled correctly for slice ops"
        )

    test_path = _FSDP2_DIR / "run_fsdp2_fused_adam.py"

    # Phase 1: save checkpoint with 4 ranks.
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=4",
            "--local-ranks-filter=0",
            str(test_path),
            "--test",
            "dcp_resharding_save",
            "--recipe",
            recipe,
        ],
        env=os.environ,
        timeout=300,
    )
    assert result.returncode == 0, f"DCP resharding save phase failed: {result.returncode}"

    # Phase 2: load checkpoint with 2 ranks (different topology).
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=2",
            "--local-ranks-filter=0",
            str(test_path),
            "--test",
            "dcp_resharding_load",
            "--recipe",
            recipe,
        ],
        env=os.environ,
        timeout=300,
    )
    assert result.returncode == 0, f"DCP resharding load phase failed: {result.returncode}"


def test_dummy() -> None:
    """Dummy test

    pytest returns exit code 5 if all tests are skipped.

    """
    pass
