# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import check_fp8_support, fp8_autocast
from transformer_engine.pytorch import Linear
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
)


def test_custom_recipe_linear():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(0)

    # Simple linear layer with dims divisible by 16
    in_features = 64
    out_features = 64
    batch = 32

    model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Provide only input, weight, and grad_output quantizers
    input_quantizer = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    weight_quantizer = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    grad_output_quantizer = Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")

    qparams = recipe.QLinearParams(
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
        grad_output_quantizer=grad_output_quantizer,
    )

    custom_recipe = recipe.CustomRecipe(qparams=qparams)

    # Execute with custom recipe
    with fp8_autocast(enabled=True, fp8_recipe=custom_recipe):
        out = model(inp)
    loss = out.float().sum()
    loss.backward()

    # Basic sanity: gradients exist
    assert inp.grad is not None


def test_custom_recipe_matches_current_scaling():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(123)

    in_features = 64
    out_features = 64
    batch = 32

    # Create two identical models
    model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    model_custom = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    model_custom.load_state_dict(model_ref.state_dict())

    # Identical inputs for both paths
    base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
    inp_ref = base_inp.clone().detach().requires_grad_(True)
    inp_custom = base_inp.clone().detach().requires_grad_(True)

    # Reference: use Float8CurrentScaling recipe
    ref_recipe = recipe.Float8CurrentScaling()
    with fp8_autocast(enabled=True, fp8_recipe=ref_recipe):
        out_ref = model_ref(inp_ref)
    loss_ref = out_ref.float().sum()
    loss_ref.backward()

    # Custom: use explicit Float8CurrentScalingQuantizers (only input, weight, grad_output)
    input_q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    weight_q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    grad_out_q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")

    qparams = recipe.QLinearParams(
        input_quantizer=input_q,
        weight_quantizer=weight_q,
        grad_output_quantizer=grad_out_q,
    )
    custom_recipe = recipe.CustomRecipe(qparams=qparams)

    with fp8_autocast(enabled=True, fp8_recipe=custom_recipe):
        out_custom = model_custom(inp_custom)
    loss_custom = out_custom.float().sum()
    loss_custom.backward()

    # Compare forward outputs (exact match expected)
    assert torch.allclose(out_ref, out_custom, rtol=0.0, atol=0.0)

    # Compare input gradients
    assert inp_ref.grad is not None and inp_custom.grad is not None
    assert torch.allclose(inp_ref.grad, inp_custom.grad, rtol=0.0, atol=0.0)

    # Compare parameter gradients (weights and bias if present)
    ref_params = dict(model_ref.named_parameters())
    custom_params = dict(model_custom.named_parameters())
    for name, p_ref in ref_params.items():
        p_cus = custom_params[name]
        # Parameter tensors should be identical initially; now check grads
        if p_ref.grad is None and p_cus.grad is None:
            continue
        assert p_ref.grad is not None and p_cus.grad is not None
        assert torch.allclose(p_ref.grad, p_cus.grad, rtol=0.0, atol=0.0)


def test_custom_recipe_ops_linear_2_1_layout():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(7)

    in_features = 64
    out_features = 64
    batch = 16

    # Use ops.Linear which consumes 2 forward quantizers and 1 backward quantizer
    op = te_ops.Linear(in_features, out_features, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    input_q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    weight_q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    grad_out_q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")

    qparams = recipe.QLinearParams(
        input_quantizer=input_q,
        weight_quantizer=weight_q,
        grad_output_quantizer=grad_out_q,
    )
    custom = recipe.CustomRecipe(qparams=qparams)

    with fp8_autocast(enabled=True, fp8_recipe=custom):
        out = op(inp)
    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None


def test_custom_recipe_ops_linear_partial_qparam_reuse():
    available, reason = check_fp8_support()
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(11)

    in_features = 64
    out_features = 64
    batch = 8

    op = te_ops.Linear(in_features, out_features, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Provide only input quantizer for forward and only grad_output for backward.
    # Forward will reuse input quantizer for weight due to cycling, backward uses grad_output.
    input_q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    grad_out_q = Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")

    qparams = recipe.QLinearParams(
        input_quantizer=input_q,
        weight_quantizer=None,  # deliberately missing to exercise reuse
        output_quantizer=None,
        grad_output_quantizer=grad_out_q,
        grad_input_quantizer=None,
    )
    custom = recipe.CustomRecipe(qparams=qparams)

    with fp8_autocast(enabled=True, fp8_recipe=custom):
        out = op(inp)
    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None
