# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from transformer_engine.pytorch.quantization import QuantizerRole
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.custom_recipes.quantization_recipes_base import (
    current_scaling_quantizer_factory,
    mxfp8_quantizer_factory,
    float8_block_scaling_quantizer_factory,
    nvfp4_quantizer_factory,
)
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import (
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
        if role is None:
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role.tensor_type in ("grad_output"):
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
        if role is None:
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role.tensor_type in ("grad_output"):
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
        if role is None:
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role.tensor_type in ("grad_output"):
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

    custom_recipe = recipe.CustomRecipe(qfactory=quantizer_factory)

    with autocast(enabled=True, recipe=custom_recipe):
        out_custom = model_custom(inp_custom)
    # Assert dtypes for custom quantizers match reference mapping.
    # The output (fwd) and grad_input (bwd) slots receive role=None
    # (unknown consumer) and get E4M3 from our factory.  The reference
    # recipe uses E4M3 for fwd output and E5M2 for bwd grad_input,
    # but these quantizers are typically unused so the mismatch doesn't
    # affect GEMM results.
    cus_fwd_in = model_custom.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
    cus_fwd_w = model_custom.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
    cus_fwd_out = model_custom.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
    cus_bwd_go = model_custom.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
    cus_bwd_gi = model_custom.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT1]
    assert cus_fwd_in.dtype == tex.DType.kFloat8E4M3
    assert cus_fwd_w.dtype == tex.DType.kFloat8E4M3
    assert cus_fwd_out.dtype == tex.DType.kFloat8E4M3
    assert cus_bwd_go.dtype == tex.DType.kFloat8E5M2
    assert cus_bwd_gi.dtype == tex.DType.kFloat8E4M3  # role=None fallback

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
        if role is None:
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role.tensor_type in ("grad_output"):
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

    # Counters per tensor_type.  The output (fwd) and grad_input (bwd)
    # slots have role=None by default (unknown consumer), so we count
    # those separately.
    counts = {
        "input": 0,
        "weight": 0,
        "grad_output": 0,
        None: 0,
    }

    def quantizer_factory(role):
        if role is None:
            counts[None] += 1
            return Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3, device=torch.device("cuda")
            )
        assert isinstance(role, QuantizerRole), f"Expected QuantizerRole, got {type(role)}"
        assert role.module_type == "linear"
        if role.tensor_type in counts:
            counts[role.tensor_type] += 1
        if role.tensor_type == "grad_output":
            return Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device=torch.device("cuda")
            )
        return Float8CurrentScalingQuantizer(
            tex.DType.kFloat8E4M3, device=torch.device("cuda")
        )

    custom = recipe.CustomRecipe(qfactory=quantizer_factory)

    with autocast(enabled=True, recipe=custom):
        out = op(inp)
    loss = out.float().sum()
    loss.backward()

    # Forward: input, weight, output(None); backward: grad_output, grad_input(None)
    assert counts["input"] == 1
    assert counts["weight"] == 1
    assert counts["grad_output"] == 1
    assert counts[None] == 2, f"Expected 2 None roles (output + grad_input), got {counts[None]}"


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


def _run_linear_fwd_bwd(model, inp, recipe):
    """Run forward + backward with a given recipe and return (output, inp.grad, param grads)."""
    with autocast(enabled=True, recipe=recipe):
        out = model(inp)
    loss = out.float().sum()
    loss.backward()
    param_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.clone(), inp.grad.clone(), param_grads


def _make_pair(in_features=128, out_features=128, batch=32, seed=42):
    """Create a pair of identical Linear models and matching inputs."""
    torch.manual_seed(seed)
    model_ref = Linear(in_features, out_features, params_dtype=torch.bfloat16, bias=False).cuda()
    model_cus = Linear(in_features, out_features, params_dtype=torch.bfloat16, bias=False).cuda()
    model_cus.load_state_dict(model_ref.state_dict())

    base_inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
    inp_ref = base_inp.clone().detach().requires_grad_(True)
    inp_cus = base_inp.clone().detach().requires_grad_(True)
    return model_ref, model_cus, inp_ref, inp_cus


def _assert_match(out_ref, out_cus, grad_ref, grad_cus, pgrads_ref, pgrads_cus):
    """Assert exact match of outputs and all gradients."""
    assert torch.allclose(
        out_ref, out_cus, rtol=0.0, atol=0.0
    ), f"Forward mismatch: max diff = {(out_ref - out_cus).abs().max()}"
    assert torch.allclose(
        grad_ref, grad_cus, rtol=0.0, atol=0.0
    ), f"Input grad mismatch: max diff = {(grad_ref - grad_cus).abs().max()}"
    for name in pgrads_ref:
        assert torch.allclose(pgrads_ref[name], pgrads_cus[name], rtol=0.0, atol=0.0), (
            f"Param grad '{name}' mismatch: max diff = "
            f"{(pgrads_ref[name] - pgrads_cus[name]).abs().max()}"
        )


def test_factory_matches_current_scaling():
    """current_scaling_quantizer_factory should produce bit-identical results
    to the built-in Float8CurrentScaling recipe."""
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    model_ref, model_cus, inp_ref, inp_cus = _make_pair()

    out_ref, grad_ref, pgrads_ref = _run_linear_fwd_bwd(
        model_ref, inp_ref, recipe.Float8CurrentScaling()
    )
    out_cus, grad_cus, pgrads_cus = _run_linear_fwd_bwd(
        model_cus, inp_cus, recipe.CustomRecipe(qfactory=current_scaling_quantizer_factory)
    )
    _assert_match(out_ref, out_cus, grad_ref, grad_cus, pgrads_ref, pgrads_cus)


def test_factory_matches_mxfp8():
    """mxfp8_quantizer_factory should produce bit-identical results
    to the built-in MXFP8BlockScaling recipe."""
    available, reason = te.is_mxfp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"MXFP8 unsupported: {reason}")

    model_ref, model_cus, inp_ref, inp_cus = _make_pair()

    out_ref, grad_ref, pgrads_ref = _run_linear_fwd_bwd(
        model_ref, inp_ref, recipe.MXFP8BlockScaling()
    )
    out_cus, grad_cus, pgrads_cus = _run_linear_fwd_bwd(
        model_cus, inp_cus, recipe.CustomRecipe(qfactory=mxfp8_quantizer_factory)
    )
    _assert_match(out_ref, out_cus, grad_ref, grad_cus, pgrads_ref, pgrads_cus)


def test_factory_matches_block_scaling():
    """float8_block_scaling_quantizer_factory should produce bit-identical results
    to the built-in Float8BlockScaling recipe."""
    available = te.is_fp8_block_scaling_available()
    if not torch.cuda.is_available() or not available:
        pytest.skip("Float8 block scaling unsupported on this device")

    model_ref, model_cus, inp_ref, inp_cus = _make_pair()

    out_ref, grad_ref, pgrads_ref = _run_linear_fwd_bwd(
        model_ref, inp_ref, recipe.Float8BlockScaling()
    )
    out_cus, grad_cus, pgrads_cus = _run_linear_fwd_bwd(
        model_cus, inp_cus, recipe.CustomRecipe(qfactory=float8_block_scaling_quantizer_factory)
    )
    _assert_match(out_ref, out_cus, grad_ref, grad_cus, pgrads_ref, pgrads_cus)


def test_factory_matches_nvfp4():
    """nvfp4_quantizer_factory should produce bit-identical results
    to the built-in NVFP4BlockScaling recipe."""
    available = te.is_nvfp4_available()
    if not torch.cuda.is_available() or not available:
        pytest.skip("NVFP4 unsupported on this device")

    model_ref, model_cus, inp_ref, inp_cus = _make_pair()

    out_ref, grad_ref, pgrads_ref = _run_linear_fwd_bwd(
        model_ref, inp_ref, recipe.NVFP4BlockScaling()
    )
    out_cus, grad_cus, pgrads_cus = _run_linear_fwd_bwd(
        model_cus, inp_cus, recipe.CustomRecipe(qfactory=nvfp4_quantizer_factory)
    )

    _assert_match(out_ref, out_cus, grad_ref, grad_cus, pgrads_ref, pgrads_cus)
