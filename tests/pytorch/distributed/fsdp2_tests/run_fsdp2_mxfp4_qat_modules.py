#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FSDP2 coverage for MXFP4-QAT module-level APIs.

Run with:
  torchrun --nproc_per_node=2 -m pytest <this_file> -v -s --tb=short

Module counterpart of ``run_fsdp2_mxfp4_qat_ops.py`` (same reference scheme:
master vs MXFP4-projected weights discriminate which operand each backward
override used). Covers only configurations that pure recipes already support:

* te.Linear / te.LayerNormLinear: all backward override modes.
* te.LayerNormMLP: mode ``None`` (the module rejects overrides).
* te.GroupedLinear, independent per-expert weights: all override modes.

Both QAT host recipes (MXFP8 and 128x128 blockwise FP8) are parametrized.
"""

from collections.abc import Callable, Sequence

import pytest
import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed._composable.fsdp import fully_shard

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    Float8BlockScaling,
    MXFP4QATFloat8BlockScaling,
    MXFP4QATMXFP8BlockScaling,
    MXFP8BlockScaling,
)
from transformer_engine.pytorch import fp8
from transformer_engine.pytorch.mxfp4_qat import mxfp4_fake_quantize

_DTYPE = torch.bfloat16

_RECIPES = [
    pytest.param(
        (MXFP4QATMXFP8BlockScaling, MXFP8BlockScaling, fp8.check_mxfp8_support),
        id="mxfp8",
    ),
    pytest.param(
        (MXFP4QATFloat8BlockScaling, Float8BlockScaling, fp8.check_fp8_block_scaling_support),
        id="blockwise",
    ),
]

_OVERRIDES = (
    pytest.param(None, id="none"),
    pytest.param("dequantized", id="dequantized"),
    pytest.param("high_precision", id="high_precision"),
)


def _skip_if_unsupported(check_fn) -> None:
    supported, reason = check_fn()
    if not supported:
        pytest.skip(reason)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _named_weights(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """GEMM weight parameters (weight, weightN, fc1_weight, ...), not layernorm affines."""
    weights = [
        (name, param)
        for name, param in module.named_parameters()
        if (lambda last: last.startswith("weight") or last.endswith("_weight"))(
            name.rsplit(".", 1)[-1]
        )
        and "layer_norm" not in name
    ]
    assert weights, f"{type(module).__name__} has no weight parameters"
    return weights


def _make_master_weight(
    out_features: int,
    in_features: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Construct BF16 values that MXFP8/blockwise FP8 preserve but MXFP4 changes."""
    pattern = torch.tensor(
        [
            1.0,
            0.3125,
            -0.3125,
            0.6875,
            -0.6875,
            0.15625,
            -0.15625,
            0.8125,
        ],
        dtype=_DTYPE,
        device=device,
    )
    repeats = (in_features + pattern.numel() - 1) // pattern.numel()
    row = pattern.repeat(repeats)[:in_features]
    return row.unsqueeze(0).expand(out_features, -1).contiguous()


def _initialize_master_weights(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    master_weights = {}
    with torch.no_grad():
        for name, param in _named_weights(module):
            weight = _make_master_weight(
                param.shape[0],
                param.shape[1],
                device=param.device,
            )
            param.copy_(weight)
            master_weights[name] = weight
    return master_weights


def _project_weights(
    master_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    projected_weights = {
        name: mxfp4_fake_quantize(weight).detach() for name, weight in master_weights.items()
    }
    for name in master_weights:
        assert not torch.equal(
            master_weights[name], projected_weights[name]
        ), f"Test weight {name} unexpectedly lies entirely on the MXFP4 grid"
    return projected_weights


@torch.no_grad()
def _load_weights(module: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    named_weights = _named_weights(module)
    assert [name for name, _ in named_weights] == list(weights)
    for name, param in named_weights:
        param.copy_(weights[name])


def _fsdp2_shard(module: torch.nn.Module) -> torch.nn.Module:
    world_size = torch.distributed.get_world_size()
    mesh = DeviceMesh("cuda", list(range(world_size)))
    fully_shard(module, mesh=mesh)
    return module


def _run_reference(
    factory: Callable[[], torch.nn.Module],
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    dy: torch.Tensor,
    extra_args: Sequence,
    *,
    pure_cls: type,
    override: str | None,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    module = factory()
    _load_weights(module, weights)
    x_ref = x.detach().clone().requires_grad_(True)
    recipe = pure_cls(backward_override=override)
    with te.autocast(enabled=True, recipe=recipe):
        y_ref = module(x_ref, *extra_args)
    y_ref.backward(dy)
    assert x_ref.grad is not None
    return module, y_ref.detach(), x_ref.grad.detach()


def _assert_projected_forward(
    y_qat: torch.Tensor,
    y_projected: torch.Tensor,
    y_master: torch.Tensor,
) -> None:
    torch.testing.assert_close(y_qat, y_projected, rtol=0, atol=0)
    assert not torch.equal(
        y_qat, y_master
    ), "QAT forward matched the unprojected master-weight path"


def _assert_fsdp_weight_grads(
    fsdp_module: torch.nn.Module,
    reference_module: torch.nn.Module,
    *,
    rtol: float = 0,
    atol: float = 0,
) -> None:
    fsdp_weights = _named_weights(fsdp_module)
    reference_weights = _named_weights(reference_module)
    assert [name for name, _ in fsdp_weights] == [name for name, _ in reference_weights]
    for (name, fsdp_weight), (_, reference_weight) in zip(fsdp_weights, reference_weights):
        assert isinstance(fsdp_weight, DTensor), f"{name} was not FSDP2-sharded"
        assert fsdp_weight.grad is not None, f"STE did not return a gradient for {name}"
        assert isinstance(fsdp_weight.grad, DTensor), f"{name}.grad is not sharded"
        assert reference_weight.grad is not None
        grad = fsdp_weight.grad.full_tensor()
        assert torch.isfinite(grad).all(), f"{name}.grad contains non-finite values"
        torch.testing.assert_close(
            grad,
            reference_weight.grad,
            rtol=rtol,
            atol=atol,
            msg=lambda msg: (f"{name}: FSDP2 QAT wgrad did not reach the master shard\n{msg}"),
        )


def _make_input(tokens: int, features: int, device: torch.device) -> torch.Tensor:
    values = torch.linspace(-0.25, 0.25, tokens * features, dtype=torch.float32, device=device)
    return values.reshape(tokens, features).to(_DTYPE)


def _run_module_case(
    factory: Callable[[], torch.nn.Module],
    x: torch.Tensor,
    extra_args: Sequence,
    *,
    qat_cls: type,
    pure_cls: type,
    override: str | None,
    wgrad_rtol: float = 0,
    wgrad_atol: float = 0,
) -> None:
    """Shared body: run FSDP2 QAT vs projected/master references, assert the discriminators."""
    qat_module = factory()
    master_weights = _initialize_master_weights(qat_module)
    projected_weights = _project_weights(master_weights)
    _fsdp2_shard(qat_module)

    x = x.detach().clone().requires_grad_(True)
    qat_recipe = qat_cls(backward_override=override)
    with te.autocast(enabled=True, recipe=qat_recipe):
        y_qat = qat_module(x, *extra_args)
    dy = _make_input(y_qat.shape[0], y_qat.shape[1], y_qat.device).flip(0)
    y_qat.backward(dy)
    assert x.grad is not None

    projected_ref, y_projected, dx_projected = _run_reference(
        factory, projected_weights, x, dy, extra_args, pure_cls=pure_cls, override=override
    )
    master_ref, y_master, dx_master = _run_reference(
        factory, master_weights, x, dy, extra_args, pure_cls=pure_cls, override=override
    )

    _assert_projected_forward(y_qat.detach(), y_projected, y_master)
    expected_dx = dx_master if override == "high_precision" else dx_projected
    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
    expected_wgrad_ref = master_ref if override == "high_precision" else projected_ref
    _assert_fsdp_weight_grads(
        qat_module, expected_wgrad_ref, rtol=wgrad_rtol, atol=wgrad_atol
    )


@pytest.mark.parametrize("recipes", _RECIPES)
@pytest.mark.parametrize("override", _OVERRIDES)
def test_fsdp2_module_linear_qat(recipes, override) -> None:
    """te.Linear preserves all pure override modes under FSDP2."""
    qat_cls, pure_cls, check_fn = recipes
    _skip_if_unsupported(check_fn)
    device = _device()
    in_features = out_features = 128

    def factory() -> torch.nn.Module:
        return te.Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=_DTYPE,
            device=device,
        )

    x = _make_input(256, in_features, device)
    _run_module_case(
        factory, x, (), qat_cls=qat_cls, pure_cls=pure_cls, override=override
    )


@pytest.mark.parametrize("recipes", _RECIPES)
@pytest.mark.parametrize("override", _OVERRIDES)
def test_fsdp2_module_layernorm_linear_qat(recipes, override) -> None:
    """te.LayerNormLinear preserves all pure override modes under FSDP2."""
    qat_cls, pure_cls, check_fn = recipes
    _skip_if_unsupported(check_fn)
    device = _device()
    in_features = out_features = 128

    def factory() -> torch.nn.Module:
        return te.LayerNormLinear(
            in_features,
            out_features,
            bias=False,
            params_dtype=_DTYPE,
            device=device,
        )

    x = _make_input(256, in_features, device)
    _run_module_case(
        factory, x, (), qat_cls=qat_cls, pure_cls=pure_cls, override=override
    )


@pytest.mark.parametrize("recipes", _RECIPES)
def test_fsdp2_module_layernorm_mlp_qat_none(recipes) -> None:
    """te.LayerNormMLP mode None under FSDP2 (the module rejects overrides)."""
    qat_cls, pure_cls, check_fn = recipes
    _skip_if_unsupported(check_fn)
    device = _device()
    hidden_size = 128

    def factory() -> torch.nn.Module:
        return te.LayerNormMLP(
            hidden_size,
            4 * hidden_size,
            bias=False,
            params_dtype=_DTYPE,
            device=device,
        )

    x = _make_input(256, hidden_size, device)
    _run_module_case(
        factory, x, (), qat_cls=qat_cls, pure_cls=pure_cls, override=None
    )


@pytest.mark.parametrize("recipes", _RECIPES)
@pytest.mark.parametrize("override", _OVERRIDES)
def test_fsdp2_module_grouped_linear_qat(recipes, override) -> None:
    """te.GroupedLinear (independent expert weights) preserves override modes under FSDP2."""
    qat_cls, pure_cls, check_fn = recipes
    _skip_if_unsupported(check_fn)
    device = _device()
    num_gemms = 2
    in_features = out_features = 128
    m_splits = [128, 128]

    def factory() -> torch.nn.Module:
        return te.GroupedLinear(
            num_gemms,
            in_features,
            out_features,
            bias=False,
            params_dtype=_DTYPE,
            device=device,
        )

    x = _make_input(sum(m_splits), in_features, device)
    _run_module_case(
        factory, x, (m_splits,), qat_cls=qat_cls, pure_cls=pure_cls, override=override
    )
