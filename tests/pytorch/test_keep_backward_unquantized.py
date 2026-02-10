# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from contextlib import nullcontext
import os
from typing import Optional

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common import recipe
from transformer_engine.pytorch.ops.fused import (
    BackwardActivationBias,
    ForwardLinearBiasActivation,
    ForwardLinearBiasAdd,
    ForwardLinearScaleAdd,
)

from utils import quantization_tols, reset_rng_states


fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

# This file is intended to run in dedicated keep-backward-unquantized mode.
pytestmark = pytest.mark.skipif(
    os.environ.get("NVTE_KEEP_BACKWARD_UNQUANTIZED", "0") != "1",
    reason="Requires NVTE_KEEP_BACKWARD_UNQUANTIZED=1",
)


_quantized_numerics_recipe_list = [
    pytest.param(
        "fp8_current_scaling",
        marks=pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8),
        id="Float8CurrentScaling",
    ),
    pytest.param(
        "mxfp8",
        marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
        id="MXFP8BlockScaling",
    ),
    pytest.param(
        "fp8_block_scaling",
        marks=pytest.mark.skipif(
            not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling
        ),
        id="Float8BlockScaling",
    ),
    pytest.param(
        "nvfp4",
        marks=pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4),
        id="NVFP4BlockScaling",
    ),
]

_shape_test_cases = [
    pytest.param((32, 64), 64, id="2d_m32_k64_n64"),
    pytest.param((8, 4, 64), 128, id="3d_m32_k64_n128"),
    pytest.param((16, 2, 128), 64, id="3d_m32_k128_n64"),
]

_bias_activation_shape_cases = [
    pytest.param((32, 64), id="2d_m32_k64"),
    pytest.param((8, 4, 64), id="3d_m32_k64"),
]


def _make_recipe(recipe_name: str, quantize_backward: Optional[bool]) -> recipe.Recipe:
    kwargs = {}
    if quantize_backward is not None:
        kwargs = {"quantize_forward": True, "quantize_backward": quantize_backward}

    if recipe_name == "fp8_current_scaling":
        return recipe.Float8CurrentScaling(fp8_format=recipe.Format.E4M3, **kwargs)
    if recipe_name == "mxfp8":
        return recipe.MXFP8BlockScaling(fp8_format=recipe.Format.E4M3, **kwargs)
    if recipe_name == "fp8_block_scaling":
        return recipe.Float8BlockScaling(fp8_format=recipe.Format.E4M3, **kwargs)
    if recipe_name == "nvfp4":
        return recipe.NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
            **kwargs,
        )

    raise ValueError(f"Unsupported recipe for keep-backward-unquantized test: {recipe_name}")


def _build_keep_backward_unquantized_recipe(recipe_name: str) -> recipe.Recipe:
    fp8_recipe = _make_recipe(recipe_name, quantize_backward=None)
    assert fp8_recipe.quantize_forward
    assert not fp8_recipe.quantize_backward
    return fp8_recipe


def _build_quantized_reference_recipe(recipe_name: str) -> recipe.Recipe:
    return _make_recipe(recipe_name, quantize_backward=True)


def _copy_named_parameters(src_module: torch.nn.Module, dst_module: torch.nn.Module) -> None:
    src_params = dict(src_module.named_parameters())
    with torch.no_grad():
        for name, dst_param in dst_module.named_parameters():
            if name not in src_params:
                raise RuntimeError(f"Parameter {name} missing in source module")
            dst_param.copy_(src_params[name])


def _fprop_tolerances(recipe_name: str) -> dict[str, float]:
    if recipe_name == "mxfp8":
        return quantization_tols("mxfp8")
    if recipe_name in ("fp8_current_scaling", "fp8_block_scaling"):
        return quantization_tols("fp8_current_scaling")
    if recipe_name == "nvfp4":
        return quantization_tols("nvfp4")
    raise ValueError(f"Unsupported recipe for keep-backward-unquantized test: {recipe_name}")


def _make_linear_like_module(
    module_type: str,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    bias: bool = False,
) -> torch.nn.Module:
    if module_type == "linear":
        return te.Linear(
            in_features,
            out_features,
            bias=bias,
            params_dtype=dtype,
            device="cuda",
        )
    if module_type == "layernorm_linear":
        return te.LayerNormLinear(
            in_features,
            out_features,
            bias=bias,
            params_dtype=dtype,
            device="cuda",
        )
    if module_type == "ops_linear":
        return te_ops.Linear(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
            device="cuda",
        )
    raise ValueError(f"Unsupported module type: {module_type}")


def _maybe_skip_unsupported_recipe_module_combo(recipe_name: str, module_type: str) -> None:
    if module_type == "ops_linear" and recipe_name == "fp8_block_scaling":
        pytest.skip("Fusible ops (te_ops.Linear) do not support Float8BlockScaling recipe")


def _run_single_step(
    module: torch.nn.Module,
    x: torch.Tensor,
    dy: torch.Tensor,
    fp8_recipe: Optional[recipe.Recipe],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module.zero_grad(set_to_none=True)
    x_run = x.detach().clone().requires_grad_(True)
    autocast_ctx = (
        te.autocast(enabled=True, recipe=fp8_recipe) if fp8_recipe is not None else nullcontext()
    )
    with autocast_ctx:
        y = module(x_run)
        if isinstance(y, tuple):
            y = y[0]
    y.backward(dy)
    assert x_run.grad is not None
    assert module.weight.grad is not None
    return (
        y.detach().clone(),
        x_run.grad.detach().clone(),
        module.weight.grad.detach().clone(),
    )


def _extract_bias_grad(module: torch.nn.Module) -> Optional[torch.Tensor]:
    bias = getattr(module, "bias", None)
    if bias is None or bias.grad is None:
        return None
    return bias.grad.detach().clone()


def _run_grouped_linear_single_step(
    module: te.GroupedLinear,
    x: torch.Tensor,
    m_splits: list[int],
    dy: torch.Tensor,
    fp8_recipe: Optional[recipe.Recipe],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[Optional[torch.Tensor]]]:
    module.zero_grad(set_to_none=True)
    x_run = x.detach().clone().requires_grad_(True)
    autocast_ctx = (
        te.autocast(enabled=True, recipe=fp8_recipe) if fp8_recipe is not None else nullcontext()
    )
    with autocast_ctx:
        y = module(x_run, m_splits)
    y.backward(dy)
    assert x_run.grad is not None
    weight_grads = [
        getattr(module, f"weight{i}").grad.detach().clone() for i in range(module.num_gemms)
    ]
    bias_grads: list[Optional[torch.Tensor]] = []
    for i in range(module.num_gemms):
        if module.use_bias:
            bias_grads.append(getattr(module, f"bias{i}").grad.detach().clone())
        else:
            bias_grads.append(None)
    return y.detach().clone(), x_run.grad.detach().clone(), weight_grads, bias_grads


def _make_fused_model(
    pattern: str,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    scale: float = 0.5,
) -> te_ops.Sequential:
    if pattern == "bias_activation":
        return te_ops.Sequential(
            te_ops.Linear(in_features, out_features, bias=True, device="cuda", dtype=dtype),
            te_ops.ReLU(),
        )
    if pattern == "bias_add":
        return te_ops.Sequential(
            te_ops.Linear(in_features, out_features, bias=True, device="cuda", dtype=dtype),
            te_ops.AddExtraInput(in_place=True),
        )
    if pattern == "scale_add":
        return te_ops.Sequential(
            te_ops.Linear(in_features, out_features, bias=False, device="cuda", dtype=dtype),
            te_ops.ConstantScale(scale),
            te_ops.AddExtraInput(in_place=True),
        )
    raise ValueError(f"Unsupported fused test pattern: {pattern}")


def _run_fused_single_step(
    pattern: str,
    model: te_ops.Sequential,
    x1: torch.Tensor,
    dy: torch.Tensor,
    fp8_recipe: Optional[recipe.Recipe],
    x2: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]
]:
    model.zero_grad(set_to_none=True)
    x1_run = x1.detach().clone().requires_grad_(True)
    x2_run = x2.detach().clone().requires_grad_(True) if x2 is not None else None
    autocast_ctx = (
        te.autocast(enabled=True, recipe=fp8_recipe) if fp8_recipe is not None else nullcontext()
    )
    with autocast_ctx:
        if pattern in ("bias_add", "scale_add"):
            assert x2_run is not None
            y = model(x1_run, x2_run)
        else:
            y = model(x1_run)
    y.backward(dy)
    assert x1_run.grad is not None
    weight_grad = model[0].weight.grad.detach().clone()
    bias_grad = None
    if getattr(model[0], "bias", None) is not None and model[0].bias.grad is not None:
        bias_grad = model[0].bias.grad.detach().clone()
    x2_grad = (
        x2_run.grad.detach().clone() if x2_run is not None and x2_run.grad is not None else None
    )
    return y.detach().clone(), x1_run.grad.detach().clone(), x2_grad, weight_grad, bias_grad


def _run_quantize_op_single_step(
    model: te_ops.Sequential,
    x: torch.Tensor,
    dy: torch.Tensor,
    fp8_recipe: Optional[recipe.Recipe],
) -> tuple[torch.Tensor, torch.Tensor]:
    x_run = x.detach().clone().requires_grad_(True)
    autocast_ctx = (
        te.autocast(enabled=True, recipe=fp8_recipe) if fp8_recipe is not None else nullcontext()
    )
    with autocast_ctx:
        y = model(x_run)
    y.backward(dy)
    assert x_run.grad is not None
    return y.detach().clone(), x_run.grad.detach().clone()


@pytest.mark.parametrize(
    "recipe_name",
    _quantized_numerics_recipe_list,
)
def test_keep_backward_unquantized_recipe_defaults(recipe_name: str):
    _ = _build_keep_backward_unquantized_recipe(recipe_name)


@pytest.mark.parametrize(
    "recipe_name",
    _quantized_numerics_recipe_list,
)
@pytest.mark.parametrize(
    "module_type",
    ("linear", "layernorm_linear", "ops_linear"),
)
@pytest.mark.parametrize(
    "input_shape,out_features",
    _shape_test_cases,
)
@pytest.mark.parametrize("use_bias", (False, True), ids=("no_bias", "bias"))
def test_keep_backward_unquantized_matches_quantized_fprop_and_unquantized_grads(
    recipe_name: str,
    module_type: str,
    input_shape: tuple[int, ...],
    out_features: int,
    use_bias: bool,
):
    reset_rng_states()
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, module_type)
    dtype = torch.bfloat16
    in_features = input_shape[-1]

    module_quantized_ref = _make_linear_like_module(
        module_type, in_features, out_features, dtype, bias=use_bias
    )
    module_keep_bwd_hp = _make_linear_like_module(
        module_type, in_features, out_features, dtype, bias=use_bias
    )
    module_unquantized_ref = _make_linear_like_module(
        module_type, in_features, out_features, dtype, bias=use_bias
    )

    # Start all runs from identical parameters.
    _copy_named_parameters(module_quantized_ref, module_keep_bwd_hp)
    _copy_named_parameters(module_quantized_ref, module_unquantized_ref)

    output_shape = input_shape[:-1] + (out_features,)
    x = torch.randn(*input_shape, dtype=dtype, device="cuda")
    dy = torch.randn(*output_shape, dtype=dtype, device="cuda")

    quantized_ref_recipe = _build_quantized_reference_recipe(recipe_name)
    keep_bwd_hp_recipe = _build_keep_backward_unquantized_recipe(recipe_name)

    y_quantized_ref, _, _ = _run_single_step(module_quantized_ref, x, dy, quantized_ref_recipe)
    y_keep_bwd_hp, dx_keep_bwd_hp, dw_keep_bwd_hp = _run_single_step(
        module_keep_bwd_hp, x, dy, keep_bwd_hp_recipe
    )
    _, dx_unquantized_ref, dw_unquantized_ref = _run_single_step(
        module_unquantized_ref, x, dy, None
    )

    # Forward pass should still match quantized reference when only backward is unquantized.
    torch.testing.assert_close(
        y_keep_bwd_hp,
        y_quantized_ref,
        **_fprop_tolerances(recipe_name),
    )

    # Backward pass should match unquantized reference for dgrad and wgrad.
    torch.testing.assert_close(dx_keep_bwd_hp, dx_unquantized_ref, rtol=0, atol=0)
    torch.testing.assert_close(dw_keep_bwd_hp, dw_unquantized_ref, rtol=0, atol=0)
    if use_bias:
        bgrad_keep = _extract_bias_grad(module_keep_bwd_hp)
        bgrad_unquantized = _extract_bias_grad(module_unquantized_ref)
        assert bgrad_keep is not None
        assert bgrad_unquantized is not None
        torch.testing.assert_close(bgrad_keep, bgrad_unquantized, rtol=0, atol=0)


@pytest.mark.parametrize(
    "recipe_name",
    _quantized_numerics_recipe_list,
)
@pytest.mark.parametrize("use_bias", (False, True), ids=("no_bias", "bias"))
@pytest.mark.parametrize(
    "m_splits",
    ([32, 32, 32, 32], [64, 0, 32, 32]),
    ids=("uniform_splits", "with_empty_split"),
)
def test_keep_backward_unquantized_grouped_linear_matches_quantized_fprop_and_unquantized_grads(
    recipe_name: str,
    use_bias: bool,
    m_splits: list[int],
):
    if recipe_name == "nvfp4":
        pytest.skip("NVFP4 not supported for grouped linear")

    reset_rng_states()
    dtype = torch.bfloat16
    in_features = 64
    out_features = 64
    num_gemms = len(m_splits)
    num_tokens = sum(m_splits)

    module_quantized_ref = te.GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=use_bias,
        params_dtype=dtype,
        device="cuda",
    )
    module_keep_bwd_hp = te.GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=use_bias,
        params_dtype=dtype,
        device="cuda",
    )
    module_unquantized_ref = te.GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=use_bias,
        params_dtype=dtype,
        device="cuda",
    )

    _copy_named_parameters(module_quantized_ref, module_keep_bwd_hp)
    _copy_named_parameters(module_quantized_ref, module_unquantized_ref)

    x = torch.randn(num_tokens, in_features, dtype=dtype, device="cuda")
    dy = torch.randn(num_tokens, out_features, dtype=dtype, device="cuda")

    quantized_ref_recipe = _build_quantized_reference_recipe(recipe_name)
    keep_bwd_hp_recipe = _build_keep_backward_unquantized_recipe(recipe_name)

    y_quantized_ref, _, _, _ = _run_grouped_linear_single_step(
        module_quantized_ref, x, m_splits, dy, quantized_ref_recipe
    )
    y_keep_bwd_hp, dx_keep_bwd_hp, dw_keep_bwd_hp, db_keep_bwd_hp = _run_grouped_linear_single_step(
        module_keep_bwd_hp, x, m_splits, dy, keep_bwd_hp_recipe
    )
    _, dx_unquantized_ref, dw_unquantized_ref, db_unquantized_ref = _run_grouped_linear_single_step(
        module_unquantized_ref, x, m_splits, dy, None
    )

    torch.testing.assert_close(
        y_keep_bwd_hp,
        y_quantized_ref,
        **_fprop_tolerances(recipe_name),
    )
    torch.testing.assert_close(dx_keep_bwd_hp, dx_unquantized_ref, rtol=0, atol=0)
    for test_dw, ref_dw in zip(dw_keep_bwd_hp, dw_unquantized_ref):
        torch.testing.assert_close(test_dw, ref_dw, rtol=0, atol=0)
    if use_bias:
        for test_db, ref_db in zip(db_keep_bwd_hp, db_unquantized_ref):
            assert test_db is not None
            assert ref_db is not None
            torch.testing.assert_close(test_db, ref_db, rtol=0, atol=0)


@pytest.mark.parametrize(
    "recipe_name",
    _quantized_numerics_recipe_list,
)
@pytest.mark.parametrize(
    "fused_pattern,expected_fused_op",
    (
        ("bias_add", ForwardLinearBiasAdd),
        ("scale_add", ForwardLinearScaleAdd),
    ),
)
def test_keep_backward_unquantized_fused_linear_paths(
    recipe_name: str,
    fused_pattern: str,
    expected_fused_op: type,
):
    # Fused linear op path is based on te_ops.Linear and shares its recipe constraints.
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, "ops_linear")

    reset_rng_states()
    dtype = torch.bfloat16
    in_features = 64
    out_features = 64
    m = 32

    model_quantized_ref = _make_fused_model(fused_pattern, in_features, out_features, dtype)
    model_keep_bwd_hp = _make_fused_model(fused_pattern, in_features, out_features, dtype)
    model_unquantized_ref = _make_fused_model(fused_pattern, in_features, out_features, dtype)

    _copy_named_parameters(model_quantized_ref, model_keep_bwd_hp)
    _copy_named_parameters(model_quantized_ref, model_unquantized_ref)

    x1 = torch.randn(m, in_features, dtype=dtype, device="cuda")
    x2 = None
    if fused_pattern in ("bias_add", "scale_add"):
        x2 = torch.randn(m, out_features, dtype=dtype, device="cuda")
    dy = torch.randn(m, out_features, dtype=dtype, device="cuda")

    quantized_ref_recipe = _build_quantized_reference_recipe(recipe_name)
    keep_bwd_hp_recipe = _build_keep_backward_unquantized_recipe(recipe_name)

    y_quantized_ref, _, _, _, _ = _run_fused_single_step(
        fused_pattern, model_quantized_ref, x1, dy, quantized_ref_recipe, x2=x2
    )
    y_keep_bwd_hp, dx1_keep_bwd_hp, dx2_keep_bwd_hp, dw_keep_bwd_hp, db_keep_bwd_hp = (
        _run_fused_single_step(
            fused_pattern,
            model_keep_bwd_hp,
            x1,
            dy,
            keep_bwd_hp_recipe,
            x2=x2,
        )
    )
    _, dx1_unquantized_ref, dx2_unquantized_ref, dw_unquantized_ref, db_unquantized_ref = (
        _run_fused_single_step(
            fused_pattern,
            model_unquantized_ref,
            x1,
            dy,
            None,
            x2=x2,
        )
    )

    # Ensure this test executes the fused path changed by the keep-bwd feature.
    fused_ops = model_keep_bwd_hp._module_groups[0]._forward_ops
    assert len(fused_ops) >= 1
    assert isinstance(fused_ops[0][0], expected_fused_op)

    torch.testing.assert_close(
        y_keep_bwd_hp,
        y_quantized_ref,
        **_fprop_tolerances(recipe_name),
    )
    torch.testing.assert_close(dx1_keep_bwd_hp, dx1_unquantized_ref, rtol=0, atol=0)
    torch.testing.assert_close(dw_keep_bwd_hp, dw_unquantized_ref, rtol=0, atol=0)
    if dx2_keep_bwd_hp is not None and dx2_unquantized_ref is not None:
        torch.testing.assert_close(dx2_keep_bwd_hp, dx2_unquantized_ref, rtol=0, atol=0)
    if db_keep_bwd_hp is not None and db_unquantized_ref is not None:
        torch.testing.assert_close(db_keep_bwd_hp, db_unquantized_ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "recipe_name",
    _quantized_numerics_recipe_list,
)
@pytest.mark.parametrize("input_shape", _bias_activation_shape_cases)
def test_keep_backward_unquantized_fused_bias_activation_matches_masked_linear_backward(
    recipe_name: str,
    input_shape: tuple[int, ...],
):
    # Fused linear op path is based on te_ops.Linear and shares its recipe constraints.
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, "ops_linear")

    reset_rng_states()
    dtype = torch.bfloat16
    in_features = input_shape[-1]
    out_features = 64

    model_quantized_ref = _make_fused_model("bias_activation", in_features, out_features, dtype)
    model_keep_bwd_hp = _make_fused_model("bias_activation", in_features, out_features, dtype)
    linear_unquantized_ref = _make_linear_like_module(
        "ops_linear", in_features, out_features, dtype, bias=True
    )

    _copy_named_parameters(model_quantized_ref, model_keep_bwd_hp)
    _copy_named_parameters(model_keep_bwd_hp[0], linear_unquantized_ref)

    x1 = torch.randn(*input_shape, dtype=dtype, device="cuda")
    out_shape = x1.shape[:-1] + (out_features,)
    dy = torch.randn(*out_shape, dtype=dtype, device="cuda")

    quantized_ref_recipe = _build_quantized_reference_recipe(recipe_name)
    keep_bwd_hp_recipe = _build_keep_backward_unquantized_recipe(recipe_name)

    y_quantized_ref, _, _, _, _ = _run_fused_single_step(
        "bias_activation", model_quantized_ref, x1, dy, quantized_ref_recipe
    )
    y_keep_bwd_hp, dx1_keep_bwd_hp, _, dw_keep_bwd_hp, db_keep_bwd_hp = _run_fused_single_step(
        "bias_activation", model_keep_bwd_hp, x1, dy, keep_bwd_hp_recipe
    )

    # Ensure this test executes the fused path changed by the keep-bwd feature.
    fused_ops = model_keep_bwd_hp._module_groups[0]._forward_ops
    assert len(fused_ops) >= 1
    assert isinstance(fused_ops[0][0], ForwardLinearBiasActivation)

    # keep-bwd mode should disable backward-activation+bias fusion, while quantized
    # reference should still use it.
    keep_bwd_backward_ops = model_keep_bwd_hp._module_groups[0]._backward_ops
    assert not any(isinstance(op, BackwardActivationBias) for op, _ in keep_bwd_backward_ops)
    quantized_ref_backward_ops = model_quantized_ref._module_groups[0]._backward_ops
    assert any(isinstance(op, BackwardActivationBias) for op, _ in quantized_ref_backward_ops)

    torch.testing.assert_close(
        y_keep_bwd_hp,
        y_quantized_ref,
        **_fprop_tolerances(recipe_name),
    )

    # In keep-backward-unquantized mode, backward should behave as high-precision linear backward
    # given the ReLU mask induced by quantized forward activations.
    dy_after_activation = dy * (y_keep_bwd_hp > 0).to(dy.dtype)
    _, dx1_expected, dw_expected = _run_single_step(
        linear_unquantized_ref, x1, dy_after_activation, None
    )
    db_expected = _extract_bias_grad(linear_unquantized_ref)
    assert db_keep_bwd_hp is not None
    assert db_expected is not None

    torch.testing.assert_close(dx1_keep_bwd_hp, dx1_expected, rtol=0, atol=0)
    torch.testing.assert_close(dw_keep_bwd_hp, dw_expected, rtol=0, atol=0)
    torch.testing.assert_close(db_keep_bwd_hp, db_expected, rtol=0, atol=0)


def test_keep_backward_unquantized_autocast_respects_quantize_forward_flag():
    reset_rng_states()
    dtype = torch.bfloat16
    in_features = 64
    out_features = 64

    module_quantization_disabled = _make_linear_like_module(
        "linear", in_features, out_features, dtype, bias=True
    )
    module_unquantized_ref = _make_linear_like_module(
        "linear", in_features, out_features, dtype, bias=True
    )
    _copy_named_parameters(module_quantization_disabled, module_unquantized_ref)

    x = torch.randn(32, in_features, dtype=dtype, device="cuda")
    dy = torch.randn(32, out_features, dtype=dtype, device="cuda")

    recipe_no_fwd_quant = recipe.Float8CurrentScaling(
        fp8_format=recipe.Format.E4M3,
        quantize_forward=False,
        quantize_backward=False,
    )

    y_test, dx_test, dw_test = _run_single_step(
        module_quantization_disabled, x, dy, recipe_no_fwd_quant
    )
    y_ref, dx_ref, dw_ref = _run_single_step(module_unquantized_ref, x, dy, None)

    torch.testing.assert_close(y_test, y_ref, rtol=0, atol=0)
    torch.testing.assert_close(dx_test, dx_ref, rtol=0, atol=0)
    torch.testing.assert_close(dw_test, dw_ref, rtol=0, atol=0)
    bgrad_test = _extract_bias_grad(module_quantization_disabled)
    bgrad_ref = _extract_bias_grad(module_unquantized_ref)
    assert bgrad_test is not None
    assert bgrad_ref is not None
    torch.testing.assert_close(bgrad_test, bgrad_ref, rtol=0, atol=0)


def test_keep_backward_unquantized_quantize_op_respects_recipe_overrides():
    reset_rng_states()
    dtype = torch.bfloat16
    x = torch.randn(32, 64, dtype=dtype, device="cuda")
    dy = torch.randn(32, 64, dtype=dtype, device="cuda")

    model_override = te_ops.Sequential(te_ops.Quantize(forward=True, backward=True))
    model_ref = te_ops.Sequential(te_ops.Quantize(forward=True, backward=True))

    recipe_no_quant = recipe.Float8CurrentScaling(
        fp8_format=recipe.Format.E4M3,
        quantize_forward=False,
        quantize_backward=False,
    )
    y_override, dx_override = _run_quantize_op_single_step(model_override, x, dy, recipe_no_quant)
    y_ref, dx_ref = _run_quantize_op_single_step(model_ref, x, dy, None)

    torch.testing.assert_close(y_override, y_ref, rtol=0, atol=0)
    torch.testing.assert_close(dx_override, dx_ref, rtol=0, atol=0)


def test_keep_backward_unquantized_is_invalid_for_delayed_scaling():
    with pytest.raises(
        (AssertionError, ValueError),
        match="Delayed scaling does not support quantize_backward=False",
    ):
        _ = recipe.DelayedScaling()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_keep_backward_unquantized_not_implemented_for_layernorm_mlp():
    reset_rng_states()
    layer = te.LayerNormMLP(
        hidden_size=64,
        ffn_hidden_size=64,
        params_dtype=torch.bfloat16,
        bias=False,
        device="cuda",
    )
    x = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    keep_bwd_hp_recipe = _build_keep_backward_unquantized_recipe("fp8_current_scaling")

    with pytest.raises(
        AssertionError, match="NVTE_KEEP_BACKWARD_UNQUANTIZED is not implemented in LayerNormMLP"
    ):
        with te.autocast(enabled=True, recipe=keep_bwd_hp_recipe):
            _ = layer(x)
