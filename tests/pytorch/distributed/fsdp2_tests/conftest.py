# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared pytest fixtures and utilities for FSDP2 distributed tests.

Fixtures defined here (dist_init, _cleanup, recipe_name) are auto-discovered
by pytest for every test module in this directory.  Utility functions
(get_recipe_from_string, save_custom_attrs, restore_custom_attrs) can be
imported normally: ``from conftest import get_recipe_from_string``.
"""

import gc
import os

import pytest

import torch
import torch.distributed as dist

from transformer_engine.pytorch import fp8, QuantizedTensor
import transformer_engine.common.recipe


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
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(params=_parametrize_recipes())
def recipe_name(request):
    return request.param


# ── Other Shared helpers ───────────────────────────────────────────────────
def get_recipe_from_string(recipe):
    return getattr(transformer_engine.common.recipe, recipe)()


def save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        if isinstance(param, QuantizedTensor):
            ignore_keys = [key for key in param.__dict__.keys() if key.startswith("_")]
        else:
            ignore_keys = []
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items() if k not in ignore_keys}
    return custom_attrs


def restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)
