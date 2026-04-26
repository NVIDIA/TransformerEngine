# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Optional

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common import recipe
from transformer_engine.pytorch.cpp_extensions import general_gemm, layernorm_bwd
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported
from transformer_engine.pytorch.ops.fused import (
    BackwardActivationBias,
    ForwardLinearBiasActivation,
    ForwardLinearBiasAdd,
    ForwardLinearScaleAdd,
    UserbuffersForwardLinear,
)
from transformer_engine.pytorch.quantized_tensor import restore_from_saved

from utils import (
    assert_close,
    make_recipe,
    reset_rng_states,
    skip_unsupported_backward_override,
)


# --------------------------
# Mode and capability config
# --------------------------

_BACKWARD_OVERRIDES = ("high_precision", "dequantized")

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
bf16_available, reason_for_no_bf16 = te.is_bf16_available(return_reason=True)

_core_dtypes = [torch.float16, torch.float32]
_fused_dtypes = [torch.float16]
if bf16_available:
    _core_dtypes.insert(1, torch.bfloat16)
    _fused_dtypes.insert(1, torch.bfloat16)

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
            not fp8_block_scaling_available,
            reason=reason_for_no_fp8_block_scaling,
        ),
        id="Float8BlockScaling",
    ),
    pytest.param(
        "nvfp4",
        marks=pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4),
        id="NVFP4BlockScaling",
    ),
]


@pytest.fixture(autouse=True)
def _reset_global_fp8_state():
    """Avoid global FP8-state leakage between parametrized cases."""
    yield
    FP8GlobalStateManager.reset()


@pytest.fixture(params=_BACKWARD_OVERRIDES, ids=lambda mode: f"mode_{mode}")
def backward_override(request: pytest.FixtureRequest) -> str:
    """backward override under test."""
    return request.param


# --------------------------
# Test cases
# --------------------------


_shape_test_cases = [
    pytest.param((1, 64), 64, id="2d_m1_k64_n64"),
    pytest.param((32, 64), 64, id="2d_m32_k64_n64"),
    pytest.param((32, 96), 96, id="2d_m32_k96_n96"),
    pytest.param((32, 1, 64), 64, id="3d_m32_s1_k64_n64"),
    pytest.param((8, 4, 64), 128, id="3d_m32_k64_n128"),
    pytest.param((16, 2, 128), 64, id="3d_m32_k128_n64"),
    pytest.param((160, 64), 64, id="2d_m160_k64_n64"),
    pytest.param((5, 64, 64), 64, id="3d_m320_k64_n64"),
    pytest.param((3, 5, 32, 64), 96, id="4d_m480_k64_n96"),
    pytest.param((2, 5, 16, 128), 64, id="4d_m160_k128_n64"),
    # Intentionally unaligned token dimensions to exercise skip/support logic.
    pytest.param((3, 64), 64, id="2d_m3_k64_n64_unaligned"),
    pytest.param((3, 10, 64), 64, id="3d_m30_k64_n64_unaligned"),
    pytest.param((3, 10, 96), 96, id="3d_m30_k96_n96_unaligned"),
]

_bias_activation_shape_cases = [
    pytest.param((32, 64), id="2d_m32_k64"),
    pytest.param((32, 96), id="2d_m32_k96"),
    pytest.param((8, 4, 64), id="3d_m32_k64"),
    pytest.param((160, 64), id="2d_m160_k64"),
    pytest.param((5, 64, 64), id="3d_m320_k64"),
    pytest.param((3, 5, 32, 64), id="4d_m480_k64"),
    # Intentionally unaligned token dimensions to exercise skip/support logic.
    pytest.param((3, 64), id="2d_m3_k64_unaligned"),
    pytest.param((3, 10, 64), id="3d_m30_k64_unaligned"),
    pytest.param((3, 10, 96), id="3d_m30_k96_unaligned"),
]

_grouped_m_split_cases = [
    pytest.param([32, 32, 32, 32], id="uniform_splits"),
    pytest.param([64, 0, 32, 32], id="with_empty_split"),
    pytest.param([1, 31, 0, 96], id="small_and_empty_splits"),
    pytest.param([64, 192, 0, 128], id="64_divisible_splits"),
]

_linear_feature_cases = [
    pytest.param(64, 64, id="k64_n64"),
    pytest.param(64, 128, id="k64_n128"),
    pytest.param(128, 64, id="k128_n64"),
    pytest.param(96, 96, id="k96_n96"),
    pytest.param(64, 96, id="k64_n96"),
    pytest.param(96, 64, id="k96_n64"),
    pytest.param(128, 96, id="k128_n96"),
    pytest.param(96, 128, id="k96_n128"),
]

_output_feature_cases = [
    pytest.param(64, id="n64"),
    pytest.param(96, id="n96"),
    pytest.param(128, id="n128"),
]

# --------------------------
# Skip helpers
# --------------------------


def _maybe_skip_recipe_dtype(
    recipe_name: str,
    dtype: torch.dtype,
    module_type: Optional[str] = None,
) -> None:
    if dtype == torch.bfloat16 and not bf16_available:
        pytest.skip(reason_for_no_bf16)
    if recipe_name == "nvfp4":
        if module_type in ("linear", "layernorm_linear") and dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            pytest.skip(f"NVFP4 only supports BF16 and FP32 for {module_type} in this test")
        elif module_type in ("ops_linear", "grouped_linear") and dtype != torch.bfloat16:
            pytest.skip(f"NVFP4 only supports BF16 for {module_type} in this test")


def _maybe_skip_unsupported_recipe_module_combo(recipe_name: str, module_type: str) -> None:
    if module_type == "ops_linear" and recipe_name == "fp8_block_scaling":
        pytest.skip("Fusible ops (te_ops.Linear) do not support Float8BlockScaling recipe")


def _maybe_skip_unsupported_recipe_shape(
    recipe_name: str,
    input_shape: tuple[int, ...],
    module_type: str,
) -> None:
    flat_first_dim = math.prod(input_shape[:-1])
    last_dim = input_shape[-1]

    if module_type in ("linear", "layernorm_linear"):
        if recipe_name == "mxfp8" and (flat_first_dim % 32 != 0 or last_dim % 32 != 0):
            pytest.skip(
                "Linear/LayerNormLinear + MXFP8 requires prod(shape[:-1]) and shape[-1] divisible"
                " by 32."
            )
            return
        if recipe_name == "nvfp4" and (flat_first_dim % 16 != 0 or last_dim % 16 != 0):
            pytest.skip(
                "Linear/LayerNormLinear + NVFP4 requires prod(shape[:-1]) and shape[-1] divisible"
                " by 16."
            )
            return
        if flat_first_dim % 8 != 0 or last_dim % 16 != 0:
            pytest.skip(
                "Linear/LayerNormLinear FP8 execution requires prod(shape[:-1]) divisible by 8 "
                "and shape[-1] divisible by 16."
            )
    elif module_type == "ops_linear":
        if (
            recipe_name == "fp8_current_scaling"
            and not is_non_tn_fp8_gemm_supported()
            and flat_first_dim % 16 != 0
        ):
            pytest.skip(
                "te_ops.Linear + Float8CurrentScaling on pre-Blackwell requires "
                "prod(shape[:-1]) divisible by 16 for FP8 NT wgrad GEMM."
            )
        if recipe_name == "mxfp8" and (flat_first_dim % 32 != 0 or last_dim % 32 != 0):
            pytest.skip(
                "te_ops.Linear + MXFP8 requires prod(shape[:-1]) and shape[-1] divisible by 32."
            )
        if recipe_name == "nvfp4" and (flat_first_dim % 16 != 0 or last_dim % 16 != 0):
            pytest.skip(
                "te_ops.Linear + NVFP4 requires prod(shape[:-1]) and shape[-1] divisible by 16."
            )


def _maybe_skip_unsupported_grouped_splits(recipe_name: str, m_splits: list[int]) -> None:
    non_empty_splits = [m for m in m_splits if m > 0]
    if (
        recipe_name == "fp8_current_scaling"
        and not is_non_tn_fp8_gemm_supported()
        and any(m % 16 != 0 for m in non_empty_splits)
    ):
        pytest.skip(
            "GroupedLinear + Float8CurrentScaling on pre-Blackwell requires each "
            "non-empty m_split divisible by 16 for FP8 grouped NT wgrad GEMM."
        )
    if recipe_name == "mxfp8" and any(m % 32 != 0 for m in non_empty_splits):
        pytest.skip("GroupedLinear + MXFP8 requires each non-empty m_split divisible by 32.")
    if recipe_name == "nvfp4" and any(m % 16 != 0 for m in non_empty_splits):
        pytest.skip("GroupedLinear + NVFP4 requires each non-empty m_split divisible by 16.")
    if recipe_name == "nvfp4" and any(m % 64 != 0 for m in non_empty_splits):
        pytest.skip(
            "GroupedLinear + NVFP4 grouped split_quantize currently requires each non-empty "
            "m_split divisible by 64 due to grouped amax kernel constraints."
        )
    if recipe_name == "fp8_block_scaling" and any(m % 4 != 0 for m in non_empty_splits):
        pytest.skip(
            "GroupedLinear + Float8BlockScaling requires each non-empty m_split divisible by 4."
        )


# --------------------------
# Shared helpers
# --------------------------


def _make_linear_like_module(
    module_type: str,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    *,
    bias: bool,
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


def _make_fused_model(
    pattern: str,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    *,
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


def _dequantize_saved_operand(
    saved_operand: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if saved_operand is None:
        raise RuntimeError("Expected saved operand but got None")
    # In dequantized mode we must consume the fprop-saved quantized payload directly.
    # If row-wise payload is missing, the tensor was retargeted to a transpose-only
    # layout and no longer represents the original fprop operand.
    if (
        not isinstance(saved_operand, torch.Tensor)
        and hasattr(saved_operand, "_rowwise_data")
        and getattr(saved_operand, "_rowwise_data") is None
    ):
        raise RuntimeError(
            "Saved dequantized operand lost row-wise fprop payload (likely usage retarget)."
        )
    if isinstance(saved_operand, torch.Tensor):
        return saved_operand.to(dtype)
    if not hasattr(saved_operand, "dequantize"):
        raise RuntimeError(f"Unsupported saved operand type: {type(saved_operand)}")
    return saved_operand.dequantize(dtype=dtype)


def _snapshot_saved_quantized_operand_layout(
    saved_operand: Optional[torch.Tensor],
    *,
    name: str,
) -> dict[str, object]:
    _assert_saved_quantized_operand_uses_rowwise_only(saved_operand, name=name)
    rowwise_present = None
    columnwise_present = None
    rowwise_obj_id = None
    if hasattr(saved_operand, "_rowwise_data"):
        rowwise_data = getattr(saved_operand, "_rowwise_data")
        rowwise_present = rowwise_data is not None
        if rowwise_data is not None:
            rowwise_obj_id = id(rowwise_data)
    if hasattr(saved_operand, "_columnwise_data"):
        columnwise_present = getattr(saved_operand, "_columnwise_data") is not None
    return {
        "name": name,
        "saved_operand": saved_operand,
        "rowwise_present": rowwise_present,
        "columnwise_present": columnwise_present,
        "rowwise_obj_id": rowwise_obj_id,
    }


def _snapshot_layout_invariants(
    guard_operands: list[tuple[str, Optional[torch.Tensor]]],
) -> list[dict[str, object]]:
    """Capture saved-operand layout invariants before backward runs."""
    return [
        _snapshot_saved_quantized_operand_layout(saved_operand, name=name)
        for name, saved_operand in guard_operands
    ]


def _snapshot_backward_ctx_state(
    output: torch.Tensor,
) -> tuple[str, bool, object, bool]:
    if output.grad_fn is None:
        raise RuntimeError("Output tensor has no grad_fn; cannot inspect backward context state.")
    required_attrs = (
        "backward_override",
        "fp8",
        "grad_output_quantizer",
        "reduce_and_update_bwd_fp8_tensors",
    )
    missing_attrs = [attr for attr in required_attrs if not hasattr(output.grad_fn, attr)]
    if missing_attrs:
        raise RuntimeError(
            "grad_fn does not expose required backward context attributes: "
            f"{', '.join(missing_attrs)}."
        )
    return (
        getattr(output.grad_fn, "backward_override"),
        bool(getattr(output.grad_fn, "fp8")),
        getattr(output.grad_fn, "grad_output_quantizer"),
        bool(getattr(output.grad_fn, "reduce_and_update_bwd_fp8_tensors")),
    )


def _assert_saved_quantized_operand_uses_rowwise_only(
    saved_operand: Optional[torch.Tensor],
    *,
    name: str,
) -> None:
    if saved_operand is None:
        raise RuntimeError(f"Expected quantized saved {name} operand but got None")
    if isinstance(saved_operand, torch.Tensor):
        raise RuntimeError(
            f"dequantized reference expects quantized saved {name} operand, got torch.Tensor."
        )
    if not hasattr(saved_operand, "dequantize"):
        raise RuntimeError(f"Unsupported saved {name} operand type: {type(saved_operand)}")
    if hasattr(saved_operand, "_rowwise_data") and getattr(saved_operand, "_rowwise_data") is None:
        raise RuntimeError(
            f"Saved dequantized {name} operand lost row-wise fprop payload (likely usage retarget)."
        )
    if (
        hasattr(saved_operand, "_columnwise_data")
        and getattr(saved_operand, "_columnwise_data") is not None
    ):
        raise RuntimeError(
            f"Saved dequantized {name} operand unexpectedly carries column-wise payload."
        )


def _assert_saved_quantized_operand_layout_unchanged(snapshot: dict[str, object]) -> None:
    name = snapshot.get("name")
    if not isinstance(name, str):
        raise RuntimeError(f"Invalid saved operand snapshot name: {name!r}")
    saved_operand = snapshot.get("saved_operand")
    _assert_saved_quantized_operand_uses_rowwise_only(saved_operand, name=name)

    rowwise_present = snapshot.get("rowwise_present")
    if isinstance(rowwise_present, bool):
        rowwise_data_now = getattr(saved_operand, "_rowwise_data", None)
        rowwise_now = rowwise_data_now is not None
        if rowwise_now != rowwise_present:
            raise RuntimeError(
                f"Saved dequantized {name} operand row-wise payload presence changed "
                f"from {rowwise_present} to {rowwise_now}."
            )
        # Guard against hidden requantization that swaps in a new row-wise payload.
        rowwise_obj_id = snapshot.get("rowwise_obj_id")
        if (
            isinstance(rowwise_obj_id, int)
            and rowwise_now
            and id(rowwise_data_now) != rowwise_obj_id
        ):
            raise RuntimeError(
                f"Saved dequantized {name} operand row-wise payload identity changed "
                "(likely rewritten/requantized)."
            )

    columnwise_present = snapshot.get("columnwise_present")
    if isinstance(columnwise_present, bool):
        columnwise_now = getattr(saved_operand, "_columnwise_data", None) is not None
        if columnwise_now != columnwise_present:
            raise RuntimeError(
                f"Saved dequantized {name} operand column-wise payload presence changed "
                f"from {columnwise_present} to {columnwise_now}."
            )


def _assert_layout_invariants_unchanged(layout_invariants: list[dict[str, object]]) -> None:
    """Validate saved-operand layout invariants after backward runs."""
    for layout_invariant in layout_invariants:
        _assert_saved_quantized_operand_layout_unchanged(layout_invariant)


def _raise_if_ref_failed(ref_exc: Optional[Exception]) -> None:
    """Re-raise deferred reference exceptions after layout checks."""
    if ref_exc is not None:
        raise ref_exc


def _copy_named_parameters(src_module: torch.nn.Module, dst_module: torch.nn.Module) -> None:
    src_params = dict(src_module.named_parameters())
    with torch.no_grad():
        for name, dst_param in dst_module.named_parameters():
            if name not in src_params:
                raise RuntimeError(f"Parameter {name} missing in source module")
            dst_param.copy_(src_params[name])


def _compute_linear_backward_reference_from_saved_operands(
    saved_input: Optional[torch.Tensor],
    saved_weight: Optional[torch.Tensor],
    dy: torch.Tensor,
    *,
    dequant_dtype: torch.dtype,
    out_dtype: torch.dtype,
    with_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # dequantized reference path:
    # 1) use the exact operands saved by quantized forward,
    # 2) dequantize them to the active high-precision compute dtype,
    # 3) run backward GEMMs in high precision and compare exactly.
    for name, saved_operand in (("input", saved_input), ("weight", saved_weight)):
        _assert_saved_quantized_operand_uses_rowwise_only(saved_operand, name=name)
    dy_mat = dy.reshape(-1, dy.shape[-1])

    # Empty-token chunks can happen in grouped/fused paths. Reference should be zeros.
    if dy_mat.shape[0] == 0:
        out_features = dy_mat.shape[-1]
        if saved_input is None:
            raise RuntimeError(
                "Expected saved input operand for empty-chunk dequantized reference."
            )
        in_features = saved_input.size(-1)
        dx_ref = torch.zeros(*dy.shape[:-1], in_features, dtype=out_dtype, device=dy.device)
        dw_ref = torch.zeros(out_features, in_features, dtype=out_dtype, device=dy.device)
        db_ref = torch.zeros(out_features, dtype=out_dtype, device=dy.device)
        return dx_ref, dw_ref, db_ref

    x_ref_full = _dequantize_saved_operand(saved_input, dequant_dtype)
    x_ref = x_ref_full.reshape(-1, x_ref_full.shape[-1])
    w_ref = _dequantize_saved_operand(saved_weight, dequant_dtype)

    dx_ref_2d, *_ = general_gemm(
        w_ref,
        dy_mat,
        out_dtype=out_dtype,
        layout="NN",
        grad=True,
        use_split_accumulator=True,
    )
    db_seed = (
        torch.empty(dy_mat.shape[-1], dtype=out_dtype, device=dy_mat.device) if with_bias else None
    )
    # Derive db from the same GEMM primitive used by runtime wgrad when bias exists.
    dw_ref, db_ref, *_ = general_gemm(
        x_ref,
        dy_mat,
        out_dtype=out_dtype,
        layout="NT",
        grad=True,
        bias=db_seed,
        use_split_accumulator=True,
    )
    if db_ref is None:
        db_ref = dy_mat.sum(dim=0).to(out_dtype)
    dx_ref = dx_ref_2d.view(*dy.shape[:-1], dx_ref_2d.shape[-1])
    return dx_ref, dw_ref, db_ref


def _run_single_step(
    module: torch.nn.Module,
    x: torch.Tensor,
    dy: torch.Tensor,
    fp8_recipe: Optional[recipe.Recipe],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
    bias = getattr(module, "bias", None)
    bgrad = None if bias is None or bias.grad is None else bias.grad.detach().clone()
    return (
        y.detach().clone(),
        x_run.grad.detach().clone(),
        module.weight.grad.detach().clone(),
        bgrad,
    )


def _run_single_step_with_saved_operands(
    module: torch.nn.Module,
    x: torch.Tensor,
    fp8_recipe: recipe.Recipe,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[Optional[torch.Tensor]],
]:
    module.zero_grad(set_to_none=True)
    x_run = x.detach().clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=fp8_recipe):
        y = module(x_run)
        if isinstance(y, tuple):
            y = y[0]
        saved_operands = restore_from_saved(y.grad_fn.tensor_objects, list(y.grad_fn.saved_tensors))
    return y, x_run, saved_operands


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

    dw = [getattr(module, f"weight{i}").grad.detach().clone() for i in range(module.num_gemms)]
    db: list[Optional[torch.Tensor]] = []
    for i in range(module.num_gemms):
        if module.use_bias:
            db.append(getattr(module, f"bias{i}").grad.detach().clone())
        else:
            db.append(None)
    return y.detach().clone(), x_run.grad.detach().clone(), dw, db


def _run_grouped_linear_step_with_saved_operands(
    module: te.GroupedLinear,
    x: torch.Tensor,
    m_splits: list[int],
    fp8_recipe: recipe.Recipe,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[Optional[torch.Tensor]],
]:
    module.zero_grad(set_to_none=True)
    x_run = x.detach().clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=fp8_recipe):
        y = module(x_run, m_splits)
        saved_operands = restore_from_saved(y.grad_fn.tensor_objects, list(y.grad_fn.saved_tensors))
    return y, x_run, saved_operands


def _run_fused_single_step(
    pattern: str,
    model: te_ops.Sequential,
    x1: torch.Tensor,
    dy: torch.Tensor,
    fp8_recipe: Optional[recipe.Recipe],
    *,
    x2: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
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

    dw = model[0].weight.grad.detach().clone()
    db = None
    if getattr(model[0], "bias", None) is not None and model[0].bias.grad is not None:
        db = model[0].bias.grad.detach().clone()
    dx2 = x2_run.grad.detach().clone() if x2_run is not None and x2_run.grad is not None else None
    return y.detach().clone(), x1_run.grad.detach().clone(), dx2, dw, db


def _run_fused_single_step_with_saved_operands(
    pattern: str,
    model: te_ops.Sequential,
    x1: torch.Tensor,
    fp8_recipe: recipe.Recipe,
    *,
    x2: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    list[Optional[torch.Tensor]],
]:
    model.zero_grad(set_to_none=True)
    x1_run = x1.detach().clone().requires_grad_(True)
    x2_run = x2.detach().clone().requires_grad_(True) if x2 is not None else None
    with te.autocast(enabled=True, recipe=fp8_recipe):
        if pattern in ("bias_add", "scale_add"):
            assert x2_run is not None
            y = model(x1_run, x2_run)
        else:
            y = model(x1_run)
        saved_operands = restore_from_saved(y.grad_fn.tensor_objects, list(y.grad_fn.saved_tensors))
    return y, x1_run, x2_run, saved_operands


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


def _run_single_step_with_ctx_state(
    module: torch.nn.Module,
    x: torch.Tensor,
    dy: torch.Tensor,
    fp8_recipe: Optional[recipe.Recipe],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    tuple[str, bool, object, bool],
]:
    module.zero_grad(set_to_none=True)
    x_run = x.detach().clone().requires_grad_(True)
    autocast_ctx = (
        te.autocast(enabled=True, recipe=fp8_recipe) if fp8_recipe is not None else nullcontext()
    )
    with autocast_ctx:
        y = module(x_run)
        if isinstance(y, tuple):
            y = y[0]
        ctx_state = _snapshot_backward_ctx_state(y)
    y.backward(dy)
    assert x_run.grad is not None
    assert module.weight.grad is not None
    bias = getattr(module, "bias", None)
    bgrad = None if bias is None or bias.grad is None else bias.grad.detach().clone()
    return (
        y.detach().clone(),
        x_run.grad.detach().clone(),
        module.weight.grad.detach().clone(),
        bgrad,
        ctx_state,
    )


def _run_grouped_linear_single_step_with_ctx_state(
    module: te.GroupedLinear,
    x: torch.Tensor,
    m_splits: list[int],
    dy: torch.Tensor,
    fp8_recipe: Optional[recipe.Recipe],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    list[Optional[torch.Tensor]],
    tuple[str, bool, bool],
]:
    module.zero_grad(set_to_none=True)
    x_run = x.detach().clone().requires_grad_(True)
    autocast_ctx = (
        te.autocast(enabled=True, recipe=fp8_recipe) if fp8_recipe is not None else nullcontext()
    )
    with autocast_ctx:
        y = module(x_run, m_splits)
        if y.grad_fn is None:
            raise RuntimeError(
                "Output tensor has no grad_fn; cannot inspect grouped backward state."
            )
        required_attrs = (
            "backward_override",
            "fp8",
            "reduce_and_update_bwd_fp8_tensors",
        )
        missing_attrs = [attr for attr in required_attrs if not hasattr(y.grad_fn, attr)]
        if missing_attrs:
            raise RuntimeError(
                "Grouped grad_fn does not expose required backward context attributes: "
                f"{', '.join(missing_attrs)}."
            )
        ctx_state = (
            getattr(y.grad_fn, "backward_override"),
            bool(getattr(y.grad_fn, "fp8")),
            bool(getattr(y.grad_fn, "reduce_and_update_bwd_fp8_tensors")),
        )
    y.backward(dy)
    assert x_run.grad is not None

    dw = [getattr(module, f"weight{i}").grad.detach().clone() for i in range(module.num_gemms)]
    db: list[Optional[torch.Tensor]] = []
    for i in range(module.num_gemms):
        if module.use_bias:
            db.append(getattr(module, f"bias{i}").grad.detach().clone())
        else:
            db.append(None)
    return y.detach().clone(), x_run.grad.detach().clone(), dw, db, ctx_state


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
def test_backward_override_recipe_matches_requested_mode(
    recipe_name: str,
    backward_override: str,
) -> None:
    mode_recipe = make_recipe(recipe_name, backward_override=backward_override)
    quant_recipe = make_recipe(recipe_name)
    assert mode_recipe.backward_override == backward_override
    assert quant_recipe.backward_override is None


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize("module_type", ("linear", "layernorm_linear", "ops_linear"))
@pytest.mark.parametrize("input_shape,out_features", _shape_test_cases)
@pytest.mark.parametrize("use_bias", (False, True), ids=("no_bias", "bias"))
@pytest.mark.parametrize("dtype", _core_dtypes, ids=str)
def test_linear_like_backward_override_matches_reference(
    recipe_name: str,
    module_type: str,
    input_shape: tuple[int, ...],
    out_features: int,
    use_bias: bool,
    dtype: torch.dtype,
    backward_override: str,
) -> None:
    reset_rng_states()
    _maybe_skip_recipe_dtype(recipe_name, dtype, module_type)
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, module_type)
    _maybe_skip_unsupported_recipe_shape(recipe_name, input_shape, module_type)

    in_features = input_shape[-1]
    quantized_ref_recipe = make_recipe(recipe_name)
    mode_recipe = make_recipe(recipe_name, backward_override=backward_override)
    skip_unsupported_backward_override(module_type, mode_recipe, backward_override)

    module_quantized_ref = _make_linear_like_module(
        module_type,
        in_features,
        out_features,
        dtype,
        bias=use_bias,
    )
    module_bwd_mode = _make_linear_like_module(
        module_type,
        in_features,
        out_features,
        dtype,
        bias=use_bias,
    )
    _copy_named_parameters(module_quantized_ref, module_bwd_mode)

    output_shape = input_shape[:-1] + (out_features,)
    x = torch.randn(*input_shape, dtype=dtype, device="cuda")
    dy = torch.randn(*output_shape, dtype=dtype, device="cuda")

    y_quantized_ref, _, _, _ = _run_single_step(module_quantized_ref, x, dy, quantized_ref_recipe)
    if backward_override == "high_precision":
        # high_precision reference path: compare against a plain high-precision backward run
        # (no fp8/autocast), starting from the same params and inputs.
        module_unquantized_ref = _make_linear_like_module(
            module_type,
            in_features,
            out_features,
            dtype,
            bias=use_bias,
        )
        _copy_named_parameters(module_quantized_ref, module_unquantized_ref)
        y_bwd_mode, dx_bwd_mode, dw_bwd_mode, db_bwd_mode = _run_single_step(
            module_bwd_mode,
            x,
            dy,
            mode_recipe,
        )
        _, dx_ref, dw_ref, db_ref = _run_single_step(
            module_unquantized_ref,
            x,
            dy,
            None,
        )
    else:
        # dequantized reference path: capture saved forward operands from the real dequantized-override
        # execution, then rebuild backward reference from those saved operands.
        y_bwd_mode, x_bwd_mode, saved_operands = _run_single_step_with_saved_operands(
            module_bwd_mode, x, mode_recipe
        )
        y_bwd_mode_detached = y_bwd_mode.detach().clone()

        dx_ref: Optional[torch.Tensor] = None
        dw_ref: Optional[torch.Tensor] = None
        db_ref: Optional[torch.Tensor] = None
        layout_invariants: list[dict[str, object]] = []
        guard_operands: list[tuple[str, Optional[torch.Tensor]]] = []
        ref_exc: Optional[Exception] = None
        try:
            if module_type == "layernorm_linear":
                # LayerNormLinear dequantized reference:
                # 1) Compute d(ln_out), dw, db from linear backward with saved operands.
                # 2) Compute exact dx via layernorm_bwd with saved norm statistics.
                # _LayerNormLinear forward saves operands as:
                # [inputmat, weightmat, origin_weight, bias, ln_weight, ln_out, mu, rsigma, ...]
                if len(saved_operands) < 8:
                    raise RuntimeError(
                        "Insufficient saved operands for layernorm_linear dequantized reference "
                        f"(got {len(saved_operands)}, expected at least 8)."
                    )
                saved_input = saved_operands[0]
                saved_weight = saved_operands[1]
                saved_ln_weight = saved_operands[4]
                saved_ln_out = saved_operands[5]
                saved_mu = saved_operands[6]
                saved_rsigma = saved_operands[7]
                guard_operands.extend(
                    [
                        ("layernorm_linear_ln_out", saved_ln_out),
                        ("layernorm_linear_weight", saved_weight),
                    ]
                )
                d_ln_out_ref, dw_ref, db_ref = (
                    _compute_linear_backward_reference_from_saved_operands(
                        saved_ln_out,
                        saved_weight,
                        dy,
                        dequant_dtype=dtype,
                        out_dtype=dtype,
                        with_bias=use_bias,
                    )
                )
                input_ref = _dequantize_saved_operand(saved_input, dtype)
                input_ref_2d = input_ref.reshape(-1, input_ref.shape[-1])
                ln_weight_ref = _dequantize_saved_operand(saved_ln_weight, dtype).view(-1)
                if saved_mu is None or saved_rsigma is None:
                    raise RuntimeError("Missing LayerNorm statistics in saved operands")
                if not isinstance(saved_mu, torch.Tensor) or not isinstance(
                    saved_rsigma, torch.Tensor
                ):
                    raise RuntimeError("LayerNorm statistics must be Tensor objects")
                dx_ref, *_ = layernorm_bwd(
                    d_ln_out_ref.reshape(input_ref_2d.shape),
                    input_ref_2d,
                    saved_mu,
                    saved_rsigma,
                    ln_weight_ref,
                    module_bwd_mode.bwd_ln_sm_margin,
                    module_bwd_mode.zero_centered_gamma,
                )
                dx_ref = dx_ref.view_as(x_bwd_mode)
            else:
                saved_input, saved_weight = saved_operands[0], saved_operands[1]
                guard_operands.extend(
                    [
                        (f"{module_type}_input", saved_input),
                        (f"{module_type}_weight", saved_weight),
                    ]
                )
                linear_wgrad_with_bias = use_bias and module_type != "ops_linear"
                dx_ref, dw_ref, db_ref = _compute_linear_backward_reference_from_saved_operands(
                    saved_input,
                    saved_weight,
                    dy,
                    dequant_dtype=dtype,
                    out_dtype=dtype,
                    with_bias=linear_wgrad_with_bias,
                )
                if module_type == "ops_linear" and use_bias:
                    # te_ops bias grad is reduced by the Bias op from incoming dy.
                    db_ref = dy.reshape(-1, dy.shape[-1]).sum(dim=0).to(dtype)
        except Exception as exc:
            ref_exc = exc

        layout_invariants = _snapshot_layout_invariants(guard_operands)

        y_bwd_mode.backward(dy)
        assert x_bwd_mode.grad is not None
        assert module_bwd_mode.weight.grad is not None
        dx_bwd_mode = x_bwd_mode.grad.detach().clone()
        dw_bwd_mode = module_bwd_mode.weight.grad.detach().clone()
        bias = getattr(module_bwd_mode, "bias", None)
        db_bwd_mode = None if bias is None or bias.grad is None else bias.grad.detach().clone()
        y_bwd_mode = y_bwd_mode_detached

        _assert_layout_invariants_unchanged(layout_invariants)
        _raise_if_ref_failed(ref_exc)
        assert dx_ref is not None and dw_ref is not None and db_ref is not None

    assert_close(y_bwd_mode, y_quantized_ref, rtol=0, atol=0, check_dtype=True)
    assert_close(dx_bwd_mode, dx_ref, rtol=0, atol=0, check_dtype=True)
    assert_close(dw_bwd_mode, dw_ref, rtol=0, atol=0, check_dtype=True)
    if use_bias:
        assert db_bwd_mode is not None
        assert db_ref is not None
        assert_close(db_bwd_mode, db_ref, rtol=0, atol=0, check_dtype=True)


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize("in_features,out_features", _linear_feature_cases)
@pytest.mark.parametrize("use_bias", (False, True), ids=("no_bias", "bias"))
@pytest.mark.parametrize("m_splits", _grouped_m_split_cases)
@pytest.mark.parametrize("dtype", _core_dtypes, ids=str)
def test_grouped_linear_backward_override_matches_reference(
    recipe_name: str,
    in_features: int,
    out_features: int,
    use_bias: bool,
    m_splits: list[int],
    dtype: torch.dtype,
    backward_override: str,
) -> None:

    reset_rng_states()
    _maybe_skip_recipe_dtype(recipe_name, dtype, "grouped_linear")
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, "grouped_linear")
    _maybe_skip_unsupported_grouped_splits(recipe_name, m_splits)
    num_gemms = len(m_splits)
    num_tokens = sum(m_splits)

    quantized_ref_recipe = make_recipe(recipe_name)
    mode_recipe = make_recipe(recipe_name, backward_override=backward_override)

    module_quantized_ref = te.GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=use_bias,
        params_dtype=dtype,
        device="cuda",
    )
    module_bwd_mode = te.GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=use_bias,
        params_dtype=dtype,
        device="cuda",
    )
    _copy_named_parameters(module_quantized_ref, module_bwd_mode)

    x = torch.randn(num_tokens, in_features, dtype=dtype, device="cuda")
    dy = torch.randn(num_tokens, out_features, dtype=dtype, device="cuda")

    y_quantized_ref, _, _, _ = _run_grouped_linear_single_step(
        module_quantized_ref,
        x,
        m_splits,
        dy,
        quantized_ref_recipe,
    )
    if backward_override == "high_precision":
        # high_precision reference path: grouped module in plain high precision.
        module_unquantized_ref = te.GroupedLinear(
            num_gemms,
            in_features,
            out_features,
            bias=use_bias,
            params_dtype=dtype,
            device="cuda",
        )
        _copy_named_parameters(module_quantized_ref, module_unquantized_ref)
        y_bwd_mode, dx_bwd_mode, dw_bwd_mode, db_bwd_mode = _run_grouped_linear_single_step(
            module_bwd_mode,
            x,
            m_splits,
            dy,
            mode_recipe,
        )
        _, dx_ref, dw_ref, db_ref = _run_grouped_linear_single_step(
            module_unquantized_ref,
            x,
            m_splits,
            dy,
            None,
        )
    else:
        # dequantized reference path for grouped GEMMs:
        # each GEMM restores its own saved input/weight pair and computes its own ref grads.
        y_bwd_mode, x_bwd_mode, saved_operands = _run_grouped_linear_step_with_saved_operands(
            module_bwd_mode, x, m_splits, mode_recipe
        )
        y_bwd_mode_detached = y_bwd_mode.detach().clone()

        dx_ref: Optional[torch.Tensor] = None
        dw_ref: list[torch.Tensor] = []
        db_ref: list[Optional[torch.Tensor]] = []
        layout_invariants: list[dict[str, object]] = []
        guard_operands: list[tuple[str, Optional[torch.Tensor]]] = []
        ref_exc: Optional[Exception] = None
        try:
            if len(saved_operands) < 2 * num_gemms:
                raise RuntimeError(
                    "Insufficient saved operands for GroupedLinear dequantized reference "
                    f"(got {len(saved_operands)}, expected at least {2 * num_gemms})."
                )

            saved_inputs = saved_operands[:num_gemms]
            saved_weights = saved_operands[num_gemms : 2 * num_gemms]
            for i, (saved_input, saved_weight) in enumerate(zip(saved_inputs, saved_weights)):
                guard_operands.extend(
                    [
                        (f"grouped_input{i}", saved_input),
                        (f"grouped_weight{i}", saved_weight),
                    ]
                )
            dy_chunks = torch.split(dy, m_splits)

            dx_chunks = []
            dw_ref = []
            db_ref = []
            for dy_chunk, saved_input, saved_weight in zip(dy_chunks, saved_inputs, saved_weights):
                dx_i, dw_i, db_i = _compute_linear_backward_reference_from_saved_operands(
                    saved_input,
                    saved_weight,
                    dy_chunk,
                    dequant_dtype=dtype,
                    out_dtype=dtype,
                    with_bias=use_bias,
                )
                dx_chunks.append(dx_i)
                dw_ref.append(dw_i)
                db_ref.append(db_i if use_bias else None)
            dx_ref = torch.cat(dx_chunks, dim=0)
        except Exception as exc:
            ref_exc = exc

        layout_invariants = _snapshot_layout_invariants(guard_operands)

        y_bwd_mode.backward(dy)
        assert x_bwd_mode.grad is not None
        dx_bwd_mode = x_bwd_mode.grad.detach().clone()
        dw_bwd_mode = [
            getattr(module_bwd_mode, f"weight{i}").grad.detach().clone()
            for i in range(module_bwd_mode.num_gemms)
        ]
        db_bwd_mode = []
        for i in range(module_bwd_mode.num_gemms):
            if module_bwd_mode.use_bias:
                db_bwd_mode.append(getattr(module_bwd_mode, f"bias{i}").grad.detach().clone())
            else:
                db_bwd_mode.append(None)
        y_bwd_mode = y_bwd_mode_detached

        _assert_layout_invariants_unchanged(layout_invariants)
        _raise_if_ref_failed(ref_exc)
        assert dx_ref is not None

    assert_close(y_bwd_mode, y_quantized_ref, rtol=0, atol=0, check_dtype=True)
    assert_close(dx_bwd_mode, dx_ref, rtol=0, atol=0, check_dtype=True)
    for test_dw, ref_dw in zip(dw_bwd_mode, dw_ref):
        assert_close(test_dw, ref_dw, rtol=0, atol=0, check_dtype=True)
    if use_bias:
        for test_db, ref_db_i in zip(db_bwd_mode, db_ref):
            assert test_db is not None
            assert ref_db_i is not None
            assert_close(test_db, ref_db_i, rtol=0, atol=0, check_dtype=True)


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize("module_type", ("linear", "layernorm_linear"))
@pytest.mark.parametrize("input_shape,out_features", _shape_test_cases)
@pytest.mark.parametrize("use_bias", (False, True), ids=("no_bias", "bias"))
@pytest.mark.parametrize("dtype", _core_dtypes, ids=str)
def test_linear_like_runtime_backward_override_switch_updates_ctx(
    recipe_name: str,
    module_type: str,
    input_shape: tuple[int, ...],
    out_features: int,
    use_bias: bool,
    dtype: torch.dtype,
    backward_override: str,
) -> None:
    reset_rng_states()
    _maybe_skip_recipe_dtype(recipe_name, dtype, module_type)
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, module_type)
    _maybe_skip_unsupported_recipe_shape(recipe_name, input_shape, module_type)

    module = _make_linear_like_module(
        module_type,
        input_shape[-1],
        out_features,
        dtype,
        bias=use_bias,
    )
    x = torch.randn(*input_shape, dtype=dtype, device="cuda")
    dy = torch.randn(*input_shape[:-1], out_features, dtype=dtype, device="cuda")

    default_recipe = make_recipe(recipe_name)
    mode_recipe = make_recipe(recipe_name, backward_override=backward_override)
    skip_unsupported_backward_override(module_type, mode_recipe, backward_override)

    *_, default_ctx = _run_single_step_with_ctx_state(module, x, dy, default_recipe)
    (
        default_mode,
        default_fp8,
        default_grad_output_quantizer,
        default_reduce_and_update,
    ) = default_ctx
    assert default_mode is None
    assert default_fp8
    assert default_grad_output_quantizer is not None
    assert default_reduce_and_update

    *_, switched_ctx = _run_single_step_with_ctx_state(module, x, dy, mode_recipe)
    switched_mode, switched_fp8, switched_grad_output_quantizer, switched_reduce_and_update = (
        switched_ctx
    )
    assert switched_mode == backward_override
    assert not switched_fp8
    assert switched_grad_output_quantizer is None
    assert not switched_reduce_and_update

    *_, default_ctx_after = _run_single_step_with_ctx_state(module, x, dy, default_recipe)
    (
        default_mode_after,
        default_fp8_after,
        default_grad_output_quantizer_after,
        default_reduce_and_update_after,
    ) = default_ctx_after
    assert default_mode_after is None
    assert default_fp8_after
    assert default_grad_output_quantizer_after is not None
    assert default_reduce_and_update_after


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize("in_features,out_features", _linear_feature_cases)
@pytest.mark.parametrize("m_splits", _grouped_m_split_cases)
@pytest.mark.parametrize("use_bias", (False, True), ids=("no_bias", "bias"))
@pytest.mark.parametrize("dtype", _core_dtypes, ids=str)
def test_grouped_linear_runtime_backward_override_switch_updates_ctx(
    recipe_name: str,
    in_features: int,
    out_features: int,
    m_splits: list[int],
    use_bias: bool,
    dtype: torch.dtype,
    backward_override: str,
) -> None:

    reset_rng_states()
    _maybe_skip_recipe_dtype(recipe_name, dtype, "grouped_linear")
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, "grouped_linear")
    _maybe_skip_unsupported_grouped_splits(recipe_name, m_splits)

    num_tokens = sum(m_splits)
    module = te.GroupedLinear(
        len(m_splits),
        in_features,
        out_features,
        bias=use_bias,
        params_dtype=dtype,
        device="cuda",
    )
    x = torch.randn(num_tokens, in_features, dtype=dtype, device="cuda")
    dy = torch.randn(num_tokens, out_features, dtype=dtype, device="cuda")

    default_recipe = make_recipe(recipe_name)
    mode_recipe = make_recipe(recipe_name, backward_override=backward_override)

    *_, default_ctx = _run_grouped_linear_single_step_with_ctx_state(
        module,
        x,
        m_splits,
        dy,
        default_recipe,
    )
    default_mode, default_fp8, default_reduce_and_update = default_ctx
    assert default_mode is None
    assert default_fp8
    assert default_reduce_and_update

    *_, switched_ctx = _run_grouped_linear_single_step_with_ctx_state(
        module,
        x,
        m_splits,
        dy,
        mode_recipe,
    )
    switched_mode, switched_fp8, switched_reduce_and_update = switched_ctx
    assert switched_mode == backward_override
    assert not switched_fp8
    assert not switched_reduce_and_update

    *_, default_ctx_after = _run_grouped_linear_single_step_with_ctx_state(
        module,
        x,
        m_splits,
        dy,
        default_recipe,
    )
    default_mode_after, default_fp8_after, default_reduce_and_update_after = default_ctx_after
    assert default_mode_after is None
    assert default_fp8_after
    assert default_reduce_and_update_after


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize(
    "fused_pattern,expected_fused_op",
    (
        ("bias_add", ForwardLinearBiasAdd),
        ("scale_add", ForwardLinearScaleAdd),
    ),
)
@pytest.mark.parametrize("in_features,out_features", _linear_feature_cases)
@pytest.mark.parametrize("m", (1, 32), ids=("m1", "m32"))
@pytest.mark.parametrize("dtype", _fused_dtypes, ids=str)
def test_fused_linear_paths_match_backward_override_reference(
    recipe_name: str,
    fused_pattern: str,
    expected_fused_op: type,
    in_features: int,
    out_features: int,
    m: int,
    dtype: torch.dtype,
    backward_override: str,
) -> None:
    _maybe_skip_recipe_dtype(recipe_name, dtype, "ops_linear")
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, "ops_linear")
    _maybe_skip_unsupported_recipe_shape(recipe_name, (m, in_features), "ops_linear")

    reset_rng_states()

    quantized_ref_recipe = make_recipe(recipe_name)
    mode_recipe = make_recipe(recipe_name, backward_override=backward_override)
    skip_unsupported_backward_override("ops_linear", mode_recipe, backward_override)

    model_quantized_ref = _make_fused_model(fused_pattern, in_features, out_features, dtype)
    model_bwd_mode = _make_fused_model(fused_pattern, in_features, out_features, dtype)
    _copy_named_parameters(model_quantized_ref, model_bwd_mode)

    x1 = torch.randn(m, in_features, dtype=dtype, device="cuda")
    x2 = None
    if fused_pattern in ("bias_add", "scale_add"):
        x2 = torch.randn(m, out_features, dtype=dtype, device="cuda")
    dy = torch.randn(m, out_features, dtype=dtype, device="cuda")

    y_quantized_ref, _, _, _, _ = _run_fused_single_step(
        fused_pattern,
        model_quantized_ref,
        x1,
        dy,
        quantized_ref_recipe,
        x2=x2,
    )

    if backward_override == "high_precision":
        # high_precision reference path: replay the same fused model structure in plain
        # high precision and compare backward outputs exactly.
        model_unquantized_ref = _make_fused_model(fused_pattern, in_features, out_features, dtype)
        _copy_named_parameters(model_quantized_ref, model_unquantized_ref)

        y_bwd_mode, dx1_bwd_mode, dx2_bwd_mode, dw_bwd_mode, db_bwd_mode = _run_fused_single_step(
            fused_pattern,
            model_bwd_mode,
            x1,
            dy,
            mode_recipe,
            x2=x2,
        )
        _, dx1_ref, dx2_ref, dw_ref, db_ref = _run_fused_single_step(
            fused_pattern,
            model_unquantized_ref,
            x1,
            dy,
            None,
            x2=x2,
        )
    else:
        # dequantized reference path: compute backward reference from saved quantized
        # linear operands (with branch-specific dy handling for fused epilogues).
        y_bwd_mode, x1_bwd_mode, x2_bwd_mode_ref, saved_operands = (
            _run_fused_single_step_with_saved_operands(
                fused_pattern,
                model_bwd_mode,
                x1,
                mode_recipe,
                x2=x2,
            )
        )
        y_bwd_mode_detached = y_bwd_mode.detach().clone()
        dx1_ref: Optional[torch.Tensor] = None
        dx2_ref: Optional[torch.Tensor] = None
        dw_ref: Optional[torch.Tensor] = None
        db_ref: Optional[torch.Tensor] = None
        layout_invariants: list[dict[str, object]] = []
        guard_operands: list[tuple[str, Optional[torch.Tensor]]] = []
        ref_exc: Optional[Exception] = None
        try:
            saved_input, saved_weight = saved_operands[0], saved_operands[1]
            guard_operands.extend(
                [
                    (f"fused_{fused_pattern}_input", saved_input),
                    (f"fused_{fused_pattern}_weight", saved_weight),
                ]
            )
            dy_for_linear = dy * 0.5 if fused_pattern == "scale_add" else dy
            dx1_ref, dw_ref, db_ref = _compute_linear_backward_reference_from_saved_operands(
                saved_input,
                saved_weight,
                dy_for_linear,
                dequant_dtype=dtype,
                out_dtype=dtype,
                with_bias=False,
            )
            dx2_ref = dy if x2 is not None else None
        except Exception as exc:
            ref_exc = exc

        layout_invariants = _snapshot_layout_invariants(guard_operands)

        y_bwd_mode.backward(dy)
        assert x1_bwd_mode.grad is not None
        dx1_bwd_mode = x1_bwd_mode.grad.detach().clone()
        dx2_bwd_mode = (
            x2_bwd_mode_ref.grad.detach().clone()
            if x2_bwd_mode_ref is not None and x2_bwd_mode_ref.grad is not None
            else None
        )
        dw_bwd_mode = model_bwd_mode[0].weight.grad.detach().clone()
        db_bwd_mode = None
        if (
            getattr(model_bwd_mode[0], "bias", None) is not None
            and model_bwd_mode[0].bias.grad is not None
        ):
            db_bwd_mode = model_bwd_mode[0].bias.grad.detach().clone()
        y_bwd_mode = y_bwd_mode_detached

        _assert_layout_invariants_unchanged(layout_invariants)
        _raise_if_ref_failed(ref_exc)
        assert dx1_ref is not None and dw_ref is not None

    fused_ops = model_bwd_mode._module_groups[0]._forward_ops
    assert len(fused_ops) >= 1
    assert isinstance(fused_ops[0][0], expected_fused_op)

    assert_close(y_bwd_mode, y_quantized_ref, rtol=0, atol=0, check_dtype=True)
    assert_close(dx1_bwd_mode, dx1_ref, rtol=0, atol=0, check_dtype=True)
    assert_close(dw_bwd_mode, dw_ref, rtol=0, atol=0, check_dtype=True)
    if dx2_bwd_mode is not None and dx2_ref is not None:
        assert_close(dx2_bwd_mode, dx2_ref, rtol=0, atol=0, check_dtype=True)
    if db_bwd_mode is not None and db_ref is not None:
        assert_close(db_bwd_mode, db_ref, rtol=0, atol=0, check_dtype=True)


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize("input_shape", _bias_activation_shape_cases)
@pytest.mark.parametrize("out_features", _output_feature_cases)
@pytest.mark.parametrize("dtype", _fused_dtypes, ids=str)
def test_fused_bias_activation_matches_masked_linear_backward(
    recipe_name: str,
    input_shape: tuple[int, ...],
    out_features: int,
    dtype: torch.dtype,
    backward_override: str,
) -> None:
    _maybe_skip_recipe_dtype(recipe_name, dtype, "ops_linear")
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, "ops_linear")
    _maybe_skip_unsupported_recipe_shape(recipe_name, input_shape, "ops_linear")

    reset_rng_states()
    in_features = input_shape[-1]

    quantized_ref_recipe = make_recipe(recipe_name)
    mode_recipe = make_recipe(recipe_name, backward_override=backward_override)
    skip_unsupported_backward_override("ops_linear", mode_recipe, backward_override)

    model_quantized_ref = _make_fused_model("bias_activation", in_features, out_features, dtype)
    model_bwd_mode = _make_fused_model("bias_activation", in_features, out_features, dtype)
    _copy_named_parameters(model_quantized_ref, model_bwd_mode)

    x1 = torch.randn(*input_shape, dtype=dtype, device="cuda")
    dy = torch.randn(*((*x1.shape[:-1], out_features)), dtype=dtype, device="cuda")

    y_quantized_ref, _, _, _, _ = _run_fused_single_step(
        "bias_activation",
        model_quantized_ref,
        x1,
        dy,
        quantized_ref_recipe,
    )

    if backward_override == "high_precision":
        # high_precision reference path: build a plain linear reference and apply the
        # same activation mask (from quantized forward output) before backward.
        linear_unquantized_ref = _make_linear_like_module(
            "ops_linear",
            in_features,
            out_features,
            dtype,
            bias=True,
        )
        _copy_named_parameters(model_bwd_mode[0], linear_unquantized_ref)

        y_bwd_mode, dx1_bwd_mode, _, dw_bwd_mode, db_bwd_mode = _run_fused_single_step(
            "bias_activation",
            model_bwd_mode,
            x1,
            dy,
            mode_recipe,
        )
        dy_after_activation = dy * (y_bwd_mode > 0).to(dy.dtype)
        _, dx1_ref, dw_ref, db_ref = _run_single_step(
            linear_unquantized_ref,
            x1,
            dy_after_activation,
            None,
        )
    else:
        # dequantized reference path: restore saved linear operands from fused forward,
        # apply the same activation mask, then run linear backward reference.
        y_bwd_mode, x1_bwd_mode, _, saved_operands = _run_fused_single_step_with_saved_operands(
            "bias_activation",
            model_bwd_mode,
            x1,
            mode_recipe,
        )
        y_bwd_mode_detached = y_bwd_mode.detach().clone()
        dy_after_activation = dy * (y_bwd_mode > 0).to(dy.dtype)
        dx1_ref: Optional[torch.Tensor] = None
        dw_ref: Optional[torch.Tensor] = None
        db_ref: Optional[torch.Tensor] = None
        layout_invariants: list[dict[str, object]] = []
        guard_operands: list[tuple[str, Optional[torch.Tensor]]] = []
        ref_exc: Optional[Exception] = None
        try:
            saved_input, saved_weight = saved_operands[0], saved_operands[1]
            guard_operands.extend(
                [
                    ("fused_bias_activation_input", saved_input),
                    ("fused_bias_activation_weight", saved_weight),
                ]
            )
            dx1_ref, dw_ref, db_ref = _compute_linear_backward_reference_from_saved_operands(
                saved_input,
                saved_weight,
                dy_after_activation,
                dequant_dtype=dtype,
                out_dtype=dtype,
                with_bias=False,
            )
        except Exception as exc:
            ref_exc = exc

        layout_invariants = _snapshot_layout_invariants(guard_operands)

        y_bwd_mode.backward(dy)
        assert x1_bwd_mode.grad is not None
        dx1_bwd_mode = x1_bwd_mode.grad.detach().clone()
        dw_bwd_mode = model_bwd_mode[0].weight.grad.detach().clone()
        db_bwd_mode = (
            model_bwd_mode[0].bias.grad.detach().clone()
            if model_bwd_mode[0].bias.grad is not None
            else None
        )
        y_bwd_mode = y_bwd_mode_detached

        _assert_layout_invariants_unchanged(layout_invariants)
        _raise_if_ref_failed(ref_exc)
        assert dx1_ref is not None and dw_ref is not None and db_ref is not None

    fused_ops = model_bwd_mode._module_groups[0]._forward_ops
    assert len(fused_ops) >= 1
    assert isinstance(fused_ops[0][0], ForwardLinearBiasActivation)

    # In high_precision/dequantized modes, backward-activation+bias fusion should be disabled.
    bwd_mode_backward_ops = model_bwd_mode._module_groups[0]._backward_ops
    assert not any(isinstance(op, BackwardActivationBias) for op, _ in bwd_mode_backward_ops)

    # Quantized reference should still use fused backward path.
    quantized_ref_backward_ops = model_quantized_ref._module_groups[0]._backward_ops
    assert any(isinstance(op, BackwardActivationBias) for op, _ in quantized_ref_backward_ops)

    assert_close(y_bwd_mode, y_quantized_ref, rtol=0, atol=0, check_dtype=True)
    assert_close(dx1_bwd_mode, dx1_ref, rtol=0, atol=0, check_dtype=True)
    assert_close(dw_bwd_mode, dw_ref, rtol=0, atol=0, check_dtype=True)
    assert db_bwd_mode is not None
    assert db_ref is not None
    assert_close(db_bwd_mode, db_ref, rtol=0, atol=0, check_dtype=True)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize("in_features,out_features", _linear_feature_cases)
@pytest.mark.parametrize("dtype", _core_dtypes, ids=str)
def test_operation_fuser_rebuilds_userbuffers_fusion_on_backward_override_switch(
    recipe_name: str,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    backward_override: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a distributed setup to exercise Userbuffers fusion eligibility
    # without launching a multi-rank job.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *_args, **_kwargs: 2)

    # Use a mutable recipe holder so we can switch fusion behavior on the same
    # fuser object and verify that the cached fusion plan is refreshed.
    current_recipe = {"value": make_recipe(recipe_name)}
    monkeypatch.setattr(FP8GlobalStateManager, "get_fp8_recipe", lambda: current_recipe["value"])

    reset_rng_states()
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, "ops_linear")

    # Build a Userbuffers-eligible fuser and representative inputs.
    linear = te_ops.BasicLinear(
        in_features,
        out_features,
        device="cuda",
        dtype=dtype,
        userbuffers_options={"comm_name": "qkv"},
    )
    linear.tensor_parallel_mode = "column"
    linear.tensor_parallel_size = 2
    linear.sequence_parallel = True
    bias = te_ops.Bias(out_features, device="cuda", dtype=dtype)
    model = te_ops.Sequential(linear, bias)
    model._module_groups = model._make_module_groups(model._modules.values())
    fuser = model._module_groups[0]
    x = torch.randn(32, in_features, dtype=dtype, device="cuda", requires_grad=True)
    extra_inputs = [() for _ in range(fuser._num_basic_ops)]

    quant_recipe = make_recipe(recipe_name)
    skip_unsupported_backward_override("ops_linear", quant_recipe, backward_override)
    fuser.maybe_fuse_ops(
        is_grad_enabled=True,
        recipe=quant_recipe,
        input_=x,
        extra_inputs=extra_inputs,
    )
    assert any(isinstance(op, UserbuffersForwardLinear) for op, _ in fuser._forward_ops)

    non_quant_recipe = make_recipe(recipe_name, backward_override=backward_override)
    skip_unsupported_backward_override("ops_linear", non_quant_recipe, backward_override)
    current_recipe["value"] = non_quant_recipe
    fuser.maybe_fuse_ops(
        is_grad_enabled=True,
        recipe=non_quant_recipe,
        input_=x,
        extra_inputs=extra_inputs,
    )
    assert not any(isinstance(op, UserbuffersForwardLinear) for op, _ in fuser._forward_ops)


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize("dtype", _core_dtypes, ids=str)
def test_quantize_op_respects_backward_override(
    recipe_name: str,
    dtype: torch.dtype,
    backward_override: str,
) -> None:
    _maybe_skip_recipe_dtype(recipe_name, dtype, "ops_linear")
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, "ops_linear")
    reset_rng_states()

    x = torch.randn(32, 64, dtype=dtype, device="cuda")
    dy = torch.randn(32, 64, dtype=dtype, device="cuda")

    model_override = te_ops.Sequential(te_ops.Quantize(forward=True, backward=True))
    model_ref = te_ops.Sequential(te_ops.Quantize(forward=True, backward=False))

    mode_recipe = make_recipe(recipe_name, backward_override=backward_override)
    skip_unsupported_backward_override("ops_linear", mode_recipe, backward_override)

    y_override, dx_override = _run_quantize_op_single_step(model_override, x, dy, mode_recipe)
    y_ref, dx_ref = _run_quantize_op_single_step(model_ref, x, dy, mode_recipe)

    assert_close(y_override, y_ref, rtol=0, atol=0, check_dtype=True)
    assert_close(dx_override, dx_ref, rtol=0, atol=0, check_dtype=True)


@pytest.mark.parametrize("recipe_name", _quantized_numerics_recipe_list)
@pytest.mark.parametrize("module_type", ("linear", "layernorm_linear"))
def test_backward_override_memory_peak_report(
    recipe_name: str,
    module_type: str,
) -> None:
    """Diagnostic-only memory report for None/high_precision/dequantized backward overrides."""
    reset_rng_states()
    dtype = torch.bfloat16
    input_shape = (2048, 2048)
    out_features = 2048 * 4
    in_features = input_shape[-1]
    use_bias = True

    _maybe_skip_recipe_dtype(recipe_name, dtype, module_type)
    _maybe_skip_unsupported_recipe_module_combo(recipe_name, module_type)
    _maybe_skip_unsupported_recipe_shape(recipe_name, input_shape, module_type)

    base_module = _make_linear_like_module(
        module_type,
        in_features,
        out_features,
        dtype,
        bias=use_bias,
    )

    x = torch.randn(*input_shape, dtype=dtype, device="cuda")
    dy = torch.randn(*input_shape[:-1], out_features, dtype=dtype, device="cuda")

    modes = (None, "high_precision", "dequantized")
    mode_results: dict[str, dict[str, float] | str] = {}

    for mode in modes:
        mode_str = "default" if mode is None else mode
        # try:
        mode_recipe = make_recipe(recipe_name, backward_override=mode)

        # Keep params identical across modes for a cleaner apples-to-apples read.
        module = _make_linear_like_module(
            module_type,
            in_features,
            out_features,
            dtype,
            bias=use_bias,
        )
        _copy_named_parameters(base_module, module)

        # Warmup run to reduce first-use kernel setup noise.
        _run_single_step(module, x, dy, mode_recipe)

        module.zero_grad(set_to_none=True)
        x_run = x.detach().clone().requires_grad_(True)
        autocast_ctx = te.autocast(enabled=True, recipe=mode_recipe)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fwd_start_mem = torch.cuda.memory_allocated()
        with autocast_ctx:
            y = module(x_run)
            if isinstance(y, tuple):
                y = y[0]
        torch.cuda.synchronize()
        fwd_peak_alloc = float(torch.cuda.max_memory_allocated() - fwd_start_mem)
        fwd_peak_reserved = float(torch.cuda.max_memory_reserved())

        torch.cuda.reset_peak_memory_stats()
        bwd_start_mem = torch.cuda.memory_allocated()
        y.backward(dy)
        torch.cuda.synchronize()
        bwd_peak_alloc = float(torch.cuda.max_memory_allocated() - bwd_start_mem)
        bwd_peak_reserved = float(torch.cuda.max_memory_reserved())

        module.zero_grad(set_to_none=True)
        x_run = x.detach().clone().requires_grad_(True)
        autocast_ctx = te.autocast(enabled=True, recipe=mode_recipe)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        e2e_start_mem = torch.cuda.memory_allocated()
        with autocast_ctx:
            y = module(x_run)
            if isinstance(y, tuple):
                y = y[0]
        y.backward(dy)
        torch.cuda.synchronize()
        e2e_peak_alloc = float(torch.cuda.max_memory_allocated() - e2e_start_mem)
        e2e_peak_reserved = float(torch.cuda.max_memory_reserved())

        mode_results[mode_str] = {
            "fwd_peak_alloc_mb": fwd_peak_alloc / (1024**2),
            "fwd_peak_reserved_mb": fwd_peak_reserved / (1024**2),
            "bwd_peak_alloc_mb": bwd_peak_alloc / (1024**2),
            "bwd_peak_reserved_mb": bwd_peak_reserved / (1024**2),
            "e2e_peak_alloc_mb": e2e_peak_alloc / (1024**2),
            "e2e_peak_reserved_mb": e2e_peak_reserved / (1024**2),
        }
        # except Exception as exc:  # pragma: no cover - diagnostic reporting path
        #     mode_results[mode_str] = f"{type(exc).__name__}: {exc}"

    print(
        "\n[backward_override_memory_peak_report] "
        f"recipe={recipe_name} module_type={module_type} "
        f"dtype={dtype} input_shape={input_shape} out_features={out_features}"
    )
    print("  units=MB")
    metric_col_width = 9
    delta_col_width = 18
    columns = (
        ("mode_str", delta_col_width),
        ("fwd_alloc", metric_col_width),
        ("bwd_alloc", metric_col_width),
        ("e2e_alloc", metric_col_width),
        ("fwd_resrv", metric_col_width),
        ("bwd_resrv", metric_col_width),
        ("e2e_resrv", metric_col_width),
        ("delta_fwd", delta_col_width),
        ("delta_bwd", delta_col_width),
        ("delta_e2e", delta_col_width),
    )
    print(" | ".join(f"{name:>{width}}" for name, width in columns))
    print("-+-".join("-" * width for _, width in columns))

    def _format_delta_with_pct(delta: float, base: float) -> str:
        if math.isclose(base, 0.0, abs_tol=1e-12):
            return f"{delta:+.2f} (n/a)"
        pct = 100.0 * delta / base
        return f"{delta:+.2f} ({pct:+.2f}%)"

    default_metrics = mode_results.get("default")
    for mode in modes:
        mode_str = "default" if mode is None else mode
        metrics = mode_results[mode_str]
        if isinstance(metrics, str):
            print(f"{mode_str:>{delta_col_width}} | ERROR: {metrics}")
            continue

        if isinstance(default_metrics, dict):
            delta_fwd = metrics["fwd_peak_alloc_mb"] - default_metrics["fwd_peak_alloc_mb"]
            delta_bwd = metrics["bwd_peak_alloc_mb"] - default_metrics["bwd_peak_alloc_mb"]
            delta_e2e = metrics["e2e_peak_alloc_mb"] - default_metrics["e2e_peak_alloc_mb"]
            delta_fwd_str = _format_delta_with_pct(delta_fwd, default_metrics["fwd_peak_alloc_mb"])
            delta_bwd_str = _format_delta_with_pct(delta_bwd, default_metrics["bwd_peak_alloc_mb"])
            delta_e2e_str = _format_delta_with_pct(delta_e2e, default_metrics["e2e_peak_alloc_mb"])
        else:
            delta_fwd_str = "n/a"
            delta_bwd_str = "n/a"
            delta_e2e_str = "n/a"

        print(
            f"{mode_str:>{delta_col_width}} | "
            f"{metrics['fwd_peak_alloc_mb']:{metric_col_width}.2f} | "
            f"{metrics['bwd_peak_alloc_mb']:{metric_col_width}.2f} | "
            f"{metrics['e2e_peak_alloc_mb']:{metric_col_width}.2f} | "
            f"{metrics['fwd_peak_reserved_mb']:{metric_col_width}.2f} | "
            f"{metrics['bwd_peak_reserved_mb']:{metric_col_width}.2f} | "
            f"{metrics['e2e_peak_reserved_mb']:{metric_col_width}.2f} | "
            f"{delta_fwd_str:>{delta_col_width}} | "
            f"{delta_bwd_str:>{delta_col_width}} | "
            f"{delta_e2e_str:>{delta_col_width}}"
        )
