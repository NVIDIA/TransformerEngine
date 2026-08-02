#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Single-GPU vs FSDP2 parity for MXFP4-QAT under real data parallelism.

Run with:
  torchrun --nproc_per_node=2 -m pytest <this_file> -v -s --tb=short

Each rank trains its own slice of a deterministic global batch and compares
against two local single-GPU references: the full-batch unsharded run (forward
and dgrad must match bitwise) and a split-sum wgrad computed as per-slice wgrads
added together, which reproduces the data-parallel summation structure without
any distributed code.
"""

from collections.abc import Sequence

import pytest
import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed._composable.fsdp import fully_shard

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    MXFP4QATFloat8BlockScaling,
    MXFP4QATMXFP8BlockScaling,
)
from transformer_engine.pytorch import fp8

_DTYPE = torch.bfloat16
_HIDDEN = 128
_TOKENS = 256
_NUM_GEMMS = 2

_RECIPES = [
    pytest.param((MXFP4QATMXFP8BlockScaling, fp8.check_mxfp8_support), id="mxfp8"),
    pytest.param((MXFP4QATFloat8BlockScaling, fp8.check_fp8_block_scaling_support), id="blockwise"),
]

_OVERRIDES = (
    pytest.param(None, id="none"),
    pytest.param("dequantized", id="dequantized"),
    pytest.param("high_precision", id="high_precision"),
)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _make_master(device: torch.device) -> torch.Tensor:
    pattern = torch.tensor(
        [1.0, 0.3125, -0.3125, 0.6875, -0.6875, 0.15625, -0.15625, 0.8125],
        dtype=_DTYPE,
        device=device,
    )
    row = pattern.repeat(_HIDDEN // pattern.numel())[:_HIDDEN]
    return row.unsqueeze(0).expand(_HIDDEN, -1).contiguous()


def _global_data(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(1234)
    x = (torch.randn(_TOKENS, _HIDDEN, generator=gen, device=device) * 0.5).to(_DTYPE)
    dy = (torch.randn(_TOKENS, _HIDDEN, generator=gen, device=device) * 0.1).to(_DTYPE)
    return x, dy


def _build(kind: str, device: torch.device) -> torch.nn.Module:
    kwargs = dict(bias=False, params_dtype=_DTYPE, device=device)
    if kind == "grouped_linear":
        module = te.GroupedLinear(_NUM_GEMMS, _HIDDEN, _HIDDEN, **kwargs)
    elif kind == "layernorm_linear":
        module = te.LayerNormLinear(_HIDDEN, _HIDDEN, **kwargs)
    else:
        module = te.Linear(_HIDDEN, _HIDDEN, **kwargs)
    with torch.no_grad():
        for name in _weight_names(kind):
            getattr(module, name).copy_(_make_master(device))
    return module


def _weight_names(kind: str) -> Sequence[str]:
    if kind == "grouped_linear":
        return [f"weight{i}" for i in range(_NUM_GEMMS)]
    return ["weight"]


def _dp_slice(tensor: torch.Tensor, kind: str, rank: int, world: int) -> torch.Tensor:
    """Slice the global batch for one rank; grouped inputs are sliced per group."""
    if kind != "grouped_linear":
        per_rank = _TOKENS // world
        return tensor[rank * per_rank : (rank + 1) * per_rank]
    per_group = _TOKENS // _NUM_GEMMS
    local = per_group // world
    return torch.cat(
        [
            tensor[g * per_group + rank * local : g * per_group + (rank + 1) * local]
            for g in range(_NUM_GEMMS)
        ]
    )


def _run(
    module: torch.nn.Module,
    kind: str,
    recipe_cls: type,
    override: str | None,
    x: torch.Tensor,
    dy: torch.Tensor,
    m_splits: Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    x = x.detach().clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=recipe_cls(backward_override=override)):
        y = module(x, m_splits) if kind == "grouped_linear" else module(x)
    y.backward(dy)
    wgrads = [getattr(module, name).grad for name in _weight_names(kind)]
    return y.detach(), x.grad.detach(), wgrads


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float(), ref.float()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("kind", ("linear", "layernorm_linear", "grouped_linear"))
@pytest.mark.parametrize("recipes", _RECIPES)
@pytest.mark.parametrize("override", _OVERRIDES)
def test_fsdp2_matches_single_gpu(kind: str, recipes, override: str | None) -> None:
    """FSDP2 forward/dgrad match single GPU bitwise; wgrad matches to summation order."""
    recipe_cls, check_fn = recipes
    supported, reason = check_fn()
    if not supported:
        pytest.skip(reason)
    device = _device()
    rank, world = dist.get_rank(), dist.get_world_size()
    x_global, dy_global = _global_data(device)
    m_splits_local = (
        [_TOKENS // _NUM_GEMMS // world] * _NUM_GEMMS if kind == "grouped_linear" else None
    )
    m_splits_full = [_TOKENS // _NUM_GEMMS] * _NUM_GEMMS if kind == "grouped_linear" else None

    # Single-GPU references, computed locally on every rank.
    y_full, dx_full, wg_full = _run(
        _build(kind, device), kind, recipe_cls, override, x_global, dy_global, m_splits_full
    )
    wg_splitsum = None
    for r in range(world):
        _, _, wg_part = _run(
            _build(kind, device),
            kind,
            recipe_cls,
            override,
            _dp_slice(x_global, kind, r, world),
            _dp_slice(dy_global, kind, r, world),
            m_splits_local,
        )
        wg_part = [w.float() for w in wg_part]
        wg_splitsum = (
            wg_part if wg_splitsum is None else [a + b for a, b in zip(wg_splitsum, wg_part)]
        )

    # FSDP2 run on this rank's slice of the global batch.
    fsdp_module = _build(kind, device)
    fully_shard(fsdp_module, mesh=DeviceMesh("cuda", list(range(world))))
    y, dx, wgrads = _run(
        fsdp_module,
        kind,
        recipe_cls,
        override,
        _dp_slice(x_global, kind, rank, world),
        _dp_slice(dy_global, kind, rank, world),
        m_splits_local,
    )

    torch.testing.assert_close(y, _dp_slice(y_full, kind, rank, world), rtol=0, atol=0)
    torch.testing.assert_close(dx, _dp_slice(dx_full, kind, rank, world), rtol=0, atol=0)
    for name, wgrad, ref_split, ref_full in zip(_weight_names(kind), wgrads, wg_splitsum, wg_full):
        assert isinstance(wgrad, DTensor), f"{name} was not FSDP2-sharded"
        # FSDP2 averages DP gradients; scale back to the summed reference.
        got = wgrad.full_tensor().float() * world
        err_split = _rel_err(got, ref_split)
        assert err_split < 5e-3, (
            f"{name}: wgrad deviates from the split-sum reference beyond summation-order "
            f"noise (rel err {err_split:.2e})"
        )
        err_full = _rel_err(got, ref_full.float())
        assert (
            err_full < 5e-2
        ), f"{name}: wgrad deviates from the full-batch reference (rel err {err_full:.2e})"
