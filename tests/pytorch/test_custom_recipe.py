# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    autocast,
    Linear,
    LayerNormLinear,
    LayerNormMLP,
    GroupedLinear,
    Float8CurrentScalingQuantizer,
)
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import (
    nvfp4_ref_rht_2d_quantizer_factory,
)


@pytest.mark.parametrize("module_type", ["Linear", "LayerNormLinear", "OpsLinear"])
def test_custom_recipe_sanity_modules_nvfp4(module_type):
    """Test modules with NVFP4 custom recipe support"""
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(0)

    # Simple linear layer with dims divisible by 16
    in_features = 64
    out_features = 64
    batch = 32

    if module_type == "Linear":
        model = Linear(in_features, out_features, params_dtype=torch.bfloat16, bias=False).cuda()
    elif module_type == "LayerNormLinear":
        model = LayerNormLinear(
            in_features, out_features, params_dtype=torch.bfloat16, bias=False
        ).cuda()
    else:  # OpsLinear
        model = te_ops.Linear(
            in_features, out_features, device="cuda", dtype=torch.bfloat16, bias=False
        )
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Use NVFP4 quantizer factory
    custom_recipe = recipe.CustomRecipe(qfactory=nvfp4_ref_rht_2d_quantizer_factory)

    # Execute with custom recipe
    with autocast(enabled=True, recipe=custom_recipe):
        out = model(inp)
    loss = out.float().sum()
    loss.backward()

    # Basic sanity: gradients exist
    assert inp.grad is not None


@pytest.mark.parametrize("module_type", ["Linear", "LayerNormLinear", "OpsLinear", "LayerNormMLP"])
def test_custom_recipe_sanity(module_type):
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(0)

    # Simple linear layer with dims divisible by 16
    in_features = 64
    out_features = 64
    batch = 32

    if module_type == "Linear":
        model = Linear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    elif module_type == "LayerNormLinear":
        model = LayerNormLinear(in_features, out_features, params_dtype=torch.bfloat16).cuda()
    elif module_type == "LayerNormMLP":
        # hidden_size == in_features == out_features for simplicity
        model = LayerNormMLP(
            hidden_size=in_features, ffn_hidden_size=out_features, params_dtype=torch.bfloat16
        ).cuda()
    else:
        # OpsLinear path
        model = te_ops.Linear(in_features, out_features, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Single factory: map roles to quantizers
    def quantizer_factory(role):
        if role in ("linear_input", "linear_weight", "linear_output"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role in ("linear_grad_output", "linear_grad_input"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

    custom_recipe = recipe.CustomRecipe(qfactory=quantizer_factory)

    # Execute with custom recipe
    with autocast(enabled=True, recipe=custom_recipe):
        out = model(inp)
    loss = out.float().sum()
    loss.backward()

    # Basic sanity: gradients exist
    assert inp.grad is not None


def test_custom_recipe_grouped_linear_sanity():
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(0)

    num_gemms = 3
    in_features = 64
    out_features = 64
    batch = 32
    base = batch // num_gemms
    rem = batch % num_gemms
    m_splits = [base + (1 if i < rem else 0) for i in range(num_gemms)]

    model = GroupedLinear(num_gemms, in_features, out_features, params_dtype=torch.bfloat16).cuda()
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def quantizer_factory(role):
        if role in ("linear_input", "linear_weight", "linear_output"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role in ("linear_grad_output", "linear_grad_input"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

    custom_recipe = recipe.CustomRecipe(qfactory=quantizer_factory)

    with autocast(enabled=True, recipe=custom_recipe):
        out = model(inp, m_splits)
    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None


def test_custom_recipe_matches_current_scaling():
    available, reason = te.is_fp8_available(return_reason=True)
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
    with autocast(enabled=True, recipe=ref_recipe):
        out_ref = model_ref(inp_ref)
    # Assert dtypes for reference quantizers: HYBRID = E4M3 (fwd), E5M2 (bwd)
    ref_fwd_in = model_ref.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
    ref_fwd_w = model_ref.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
    ref_fwd_out = model_ref.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
    ref_bwd_go = model_ref.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
    ref_bwd_gi = model_ref.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT1]
    assert ref_fwd_in.dtype == tex.DType.kFloat8E4M3
    assert ref_fwd_w.dtype == tex.DType.kFloat8E4M3
    assert ref_fwd_out.dtype == tex.DType.kFloat8E4M3
    assert ref_bwd_go.dtype == tex.DType.kFloat8E5M2
    assert ref_bwd_gi.dtype == tex.DType.kFloat8E5M2

    # Stress dynamic range in grad_output
    scale = torch.ones(out_features, device="cuda", dtype=torch.float32)
    scale[0] = 1e8
    scale[1] = 1e-8
    loss_ref = (out_ref.float() * scale.view(1, -1)).sum()
    loss_ref.backward()

    # Custom: single factory returning quantizers per role to match Float8CurrentScaling
    def quantizer_factory(role):
        if role in ("linear_input", "linear_weight", "linear_output"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role in ("linear_grad_output", "linear_grad_input"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

    custom_recipe = recipe.CustomRecipe(qfactory=quantizer_factory)

    with autocast(enabled=True, recipe=custom_recipe):
        out_custom = model_custom(inp_custom)
    # Assert dtypes for custom quantizers match reference mapping
    cus_fwd_in = model_custom.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
    cus_fwd_w = model_custom.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
    cus_fwd_out = model_custom.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
    cus_bwd_go = model_custom.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
    cus_bwd_gi = model_custom.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT1]
    assert cus_fwd_in.dtype == tex.DType.kFloat8E4M3
    assert cus_fwd_w.dtype == tex.DType.kFloat8E4M3
    assert cus_fwd_out.dtype == tex.DType.kFloat8E4M3
    assert cus_bwd_go.dtype == tex.DType.kFloat8E5M2
    assert cus_bwd_gi.dtype == tex.DType.kFloat8E5M2

    loss_custom = (out_custom.float() * scale.view(1, -1)).sum()
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
        assert p_ref.grad is not None and p_cus.grad is not None
        assert torch.allclose(p_ref.grad, p_cus.grad, rtol=0.0, atol=0.0)


def test_custom_recipe_ops_linear_2_1_layout():
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(7)

    in_features = 64
    out_features = 64
    batch = 16

    # Use ops.Linear which consumes 2 forward quantizers and 1 backward quantizer
    op = te_ops.Linear(in_features, out_features, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def quantizer_factory(role):
        if role in ("linear_input", "linear_weight", "linear_output"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role in ("linear_grad_output", "linear_grad_input"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

    custom = recipe.CustomRecipe(qfactory=quantizer_factory)

    with autocast(enabled=True, recipe=custom):
        out = op(inp)
    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None


def test_custom_recipe_factory_invocation_counts_and_cycling():
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(13)

    in_features = 64
    out_features = 64
    batch = 8

    op = Linear(in_features, out_features, params_dtype=torch.bfloat16)
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Counters per role
    counts = {
        "linear_input": 0,
        "linear_weight": 0,
        "linear_output": 0,
        "linear_grad_output": 0,
        "linear_grad_input": 0,
    }

    def quantizer_factory(role):
        if role in counts:
            counts[role] += 1
        if role in ("linear_input", "linear_weight", "linear_output"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=torch.device("cuda"))
        if role in ("linear_grad_output", "linear_grad_input"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device=torch.device("cuda"))
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=torch.device("cuda"))

    custom = recipe.CustomRecipe(qfactory=quantizer_factory)

    # Run fwd+bwd once; for a single GEMM, expect forward to build 3 quantizers (cycled from 1 factory),
    # and backward to build 2 quantizers (cycled from 1 factory).
    with autocast(enabled=True, recipe=custom):
        out = op(inp)
    loss = out.float().sum()
    loss.backward()

    # Single GEMM: forward should request input, weight, output; backward grad_output, grad_input
    assert counts["linear_input"] == 1
    assert counts["linear_weight"] == 1
    assert counts["linear_output"] == 1
    assert counts["linear_grad_output"] == 1
    assert counts["linear_grad_input"] == 1


def test_factories_return_distinct_instances_and_buffers():
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    # Two calls should produce distinct quantizer objects and distinct tensor buffers
    def factory():
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=torch.device("cuda"))

    q1 = factory()
    q2 = factory()

    assert q1 is not q2
    assert q1.scale.data_ptr() != q2.scale.data_ptr()
    assert q1.amax.data_ptr() != q2.amax.data_ptr()

    # Mutating one should not affect the other
    q1.scale.fill_(123.0)
    assert not torch.equal(q1.scale, q2.scale)
