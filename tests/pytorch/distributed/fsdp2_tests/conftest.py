# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared pytest fixtures for FSDP2 distributed tests.

Fixtures defined here (dist_init, _cleanup, recipe_name) are auto-discovered
by pytest for every test module in this directory.
"""

import gc
import os
import pytest
import torch
import torch.distributed as dist
from transformer_engine.pytorch import fp8

# Ensure the correct CUDA device is active before _parametrize_recipes()
# runs at collection time, since the session-scoped dist_init fixture
# has not executed yet.
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(_local_rank)


# ── FP8 recipe parametrization ──────────────────────────────────────
def _check_nvfp4_support():
    supported, reason = fp8.check_nvfp4_support()
    if supported and torch.cuda.get_device_capability()[0] == 12:
        return (
            False,
            (
                "NVFP4BlockScaling is failing on SM120 with "
                "hadamard_transform/hadamard_transform_cast_fusion.cu:672 in function "
                "rht_gemm_ntt_w_sfc: CUDA Error: invalid argument"
            ),
        )
    return supported, reason


_FP8_RECIPE_CONFIGS = [
    ("DelayedScaling", fp8.check_fp8_support),
    ("Float8CurrentScaling", fp8.check_fp8_support),
    ("Float8BlockScaling", fp8.check_fp8_block_scaling_support),
    ("MXFP8BlockScaling", fp8.check_mxfp8_support),
    ("NVFP4BlockScaling", _check_nvfp4_support),
]


def _parametrize_recipes():
    params = []
    for name, check_fn in _FP8_RECIPE_CONFIGS:
        supported, reason = check_fn()
        params.append(
            pytest.param(name, id=name, marks=pytest.mark.skipif(not supported, reason=reason))
        )
    return params


# ── Session / per-test fixtures ──────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def dist_init():
    """Initialize the distributed process group once for the entire pytest session."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _cleanup():
    """Release GPU memory and stale NCCL state between tests."""
    yield
    if dist.is_initialized():
        dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(params=_parametrize_recipes())
def recipe_name(request):
    return request.param
