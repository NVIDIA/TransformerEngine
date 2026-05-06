# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.constants import FP8BwdTensorIdx, FP8FwdTensorIdx
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
    delayed_scaling_quantizer_factory,
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
        if role.tensor_type == "grad_output":
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
    # Each per-GEMM M dim must be a multiple of 16 to satisfy cuBLAS FP8 GEMM's
    # leading-dimension alignment requirement on Hopper (sm_90).
    m_splits = [16] * num_gemms
    batch = sum(m_splits)

    model = GroupedLinear(num_gemms, in_features, out_features, params_dtype=torch.bfloat16).cuda()
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def quantizer_factory(role):
        if role is None:
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
        if role.tensor_type == "grad_output":
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
    ref_fwd_in = model_ref.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
    ref_fwd_w = model_ref.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
    ref_fwd_out = model_ref.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_OUTPUT]
    ref_bwd_go = model_ref.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1]
    ref_bwd_gi = model_ref.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_INPUT1]
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
        if role.tensor_type == "grad_output":
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device="cuda")
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

    custom_recipe = recipe.CustomRecipe(qfactory=quantizer_factory)

    with autocast(enabled=True, recipe=custom_recipe):
        out_custom = model_custom(inp_custom)
    # Assert dtypes for custom quantizers match reference mapping
    cus_fwd_in = model_custom.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
    cus_fwd_w = model_custom.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
    cus_fwd_out = model_custom.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_OUTPUT]
    cus_bwd_go = model_custom.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1]
    cus_bwd_gi = model_custom.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_INPUT1]
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
        if role.tensor_type == "grad_output":
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
    # batch must be a multiple of 16 to satisfy cuBLAS FP8 GEMM's leading-dim
    # alignment requirement on Hopper (sm_90).
    batch = 16

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
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=torch.device("cuda"))
        assert isinstance(role, QuantizerRole), f"Expected QuantizerRole, got {type(role)}"
        assert role.module_type == "linear"
        if role.tensor_type in counts:
            counts[role.tensor_type] += 1
        if role.tensor_type == "grad_output":
            return Float8CurrentScalingQuantizer(tex.DType.kFloat8E5M2, device=torch.device("cuda"))
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=torch.device("cuda"))

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

    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

    # Two calls should produce distinct quantizer objects with distinct
    # scale/amax buffers (Float8Quantizer / delayed-scaling is the class
    # that owns persistent per-quantizer state; current scaling has none).
    def factory():
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        amax = torch.zeros(1, dtype=torch.float32, device="cuda")
        return Float8Quantizer(scale=scale, amax=amax, fp8_dtype=tex.DType.kFloat8E4M3)

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


def test_factory_matches_delayed_scaling():
    """delayed_scaling_quantizer_factory should produce bit-identical results
    to the built-in DelayedScaling recipe."""
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    model_ref, model_cus, inp_ref, inp_cus = _make_pair()

    out_ref, grad_ref, pgrads_ref = _run_linear_fwd_bwd(model_ref, inp_ref, recipe.DelayedScaling())
    out_cus, grad_cus, pgrads_cus = _run_linear_fwd_bwd(
        model_cus, inp_cus, recipe.CustomRecipe(qfactory=delayed_scaling_quantizer_factory)
    )
    _assert_match(out_ref, out_cus, grad_ref, grad_cus, pgrads_ref, pgrads_cus)


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


def test_custom_recipe_quantization_targets():
    """Validate fine-grained per-module quantization targeting via QuantizerRole.

    Four transformer layers, each assembled at a different abstraction level.
    The default recipe is NVFP4; specific modules are overridden:

      Layer 0 - ``TransformerLayer`` (name="tl0")    -> all MXFP8
      Layer 1 - ``TransformerLayer`` (name="tl1")    -> NVFP4 (default),
                except fc2 overridden to MXFP8
      Layer 2 - ``MultiheadAttention`` + ``LayerNormMLP``
                (name prefix "tl2")                   -> NVFP4 (default),
                except qkv and fc1 overridden to Float8 block-scaling
      Layer 3 - Individual blocks (name prefix "tl3") -> NVFP4 (default),
                except proj overridden to Float8 current-scaling

    The test validates that:
      * The factory receives QuantizerRole objects with correct names
      * Different quantizer types are dispatched per module
      * Forward + backward complete successfully through all four layers
    """
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")
    if not te.is_mxfp8_available():
        pytest.skip("MXFP8 unsupported on this device")
    if not te.is_nvfp4_available():
        pytest.skip("NVFP4 unsupported on this device")
    if not te.is_fp8_block_scaling_available():
        pytest.skip("Float8 block scaling unsupported on this device")

    torch.manual_seed(42)

    H = 64  # hidden_size
    FFN = 64  # ffn_hidden_size
    NH = 4  # num_heads
    KV = H // NH  # kv_channels
    B = 4  # batch
    S = 8  # seq_len
    common = dict(params_dtype=torch.bfloat16, bias=False)

    # Layer 0: TransformerLayer -> MXFP8
    tl0 = te.TransformerLayer(
        H,
        FFN,
        NH,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        name="tl0",
        **common,
    ).cuda()

    # Layer 1: TransformerLayer -> NVFP4 default, fc2 overridden to MXFP8
    tl1 = te.TransformerLayer(
        H,
        FFN,
        NH,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        name="tl1",
        **common,
    ).cuda()

    # Layer 2: MHA + LayerNormMLP -> NVFP4 default, qkv and fc1 to block-scaling
    tl2_mha = te.MultiheadAttention(
        H,
        NH,
        KV,
        attention_dropout=0.0,
        input_layernorm=True,
        return_bias=True,
        name="tl2.self_attention",
        **common,
    ).cuda()
    tl2_mlp = LayerNormMLP(H, FFN, name="tl2.layernorm_mlp", **common).cuda()

    # Layer 3: Individual blocks with DPA -> NVFP4 default, proj to current-scaling
    tl3_qkv = LayerNormLinear(H, 3 * H, name="tl3.qkv", **common).cuda()
    tl3_dpa = te.DotProductAttention(NH, KV, attention_dropout=0.0, name="tl3.core_attention")
    tl3_proj = Linear(H, H, name="tl3.proj", **common).cuda()
    tl3_fc1 = LayerNormLinear(H, FFN, name="tl3.fc1", **common).cuda()
    tl3_fc2 = Linear(FFN, H, name="tl3.fc2", **common).cuda()

    # ------------------------------------------------------------------
    # Recording + dispatching factory
    # ------------------------------------------------------------------
    recorded_roles = []

    def targeting_factory(role):
        recorded_roles.append(role)

        if role is None:
            return nvfp4_quantizer_factory(role)

        assert isinstance(role, QuantizerRole), f"Expected QuantizerRole, got {type(role)}"

        # Layer 0 (tl0.*): all MXFP8
        if role.name.startswith("tl0"):
            return mxfp8_quantizer_factory(role)

        # Layer 1 (tl1.*): NVFP4 default, but fc2 overridden to MXFP8
        if role.name == "tl1.layernorm_mlp.fc2":
            return mxfp8_quantizer_factory(role)

        # Layer 2: block scaling for qkv and fc1, rest falls through to default
        if role.name == "tl2.self_attention.layernorm_linear_qkv":
            return float8_block_scaling_quantizer_factory(role)
        if role.name == "tl2.layernorm_mlp.fc1":
            return float8_block_scaling_quantizer_factory(role)

        # Layer 3: current-scaling for proj, rest falls through to default
        if role.name == "tl3.proj":
            return current_scaling_quantizer_factory(role)

        # Default: NVFP4
        return nvfp4_quantizer_factory(role)

    custom_recipe = recipe.CustomRecipe(qfactory=targeting_factory)

    # ------------------------------------------------------------------
    # Forward + backward
    # ------------------------------------------------------------------
    inp = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with autocast(enabled=True, recipe=custom_recipe):
        # Layer 0 & 1: TransformerLayer
        h = tl1(tl0(inp))

        # Layer 2: MHA + residual + LayerNormMLP + residual
        attn_out, _ = tl2_mha(h)
        h = h + attn_out
        h = h + tl2_mlp(h)

        # Layer 3: individual blocks with DPA
        residual = h
        qkv = tl3_qkv(h).view(S, B, 3, NH, KV)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = tl3_dpa(q, k, v).view(S, B, H)
        h = residual + tl3_proj(attn)
        residual = h
        h = residual + tl3_fc2(torch.nn.functional.gelu(tl3_fc1(h)))

    loss = h.float().sum()
    loss.backward()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    assert inp.grad is not None, "Input gradient is None"

    # -- Name propagation check --
    # The factory dispatches on role.name, so if a TE module fails to propagate
    # names (e.g. TransformerLayer -> MHA -> LayerNormLinear) the factory would
    # silently fall through to the default recipe.  The quantizer-type assertions
    # below would catch that too, but checking names explicitly gives a clearer
    # error message pointing at the broken name rather than a wrong quantizer type.
    role_names = {r.name for r in recorded_roles if r is not None}

    def _tl_names(prefix):
        """Expected role names for a standard TransformerLayer with given prefix."""
        return {
            f"{prefix}.self_attention.layernorm_linear_qkv",
            f"{prefix}.self_attention.proj",
            f"{prefix}.layernorm_mlp.fc1",
            f"{prefix}.layernorm_mlp.fc2",
        }

    all_expected = (
        _tl_names("tl0")
        | _tl_names("tl1")
        | _tl_names("tl2")
        | {"tl3.qkv", "tl3.proj", "tl3.fc1", "tl3.fc2"}
    )
    missing = all_expected - role_names
    assert not missing, (
        f"Expected module names not seen in QuantizerRole.name: {missing}\n"
        f"Recorded names: {sorted(role_names)}"
    )

    for r in recorded_roles:
        if r is not None and r.module_type:
            assert r.module_type in (
                "linear",
                "dpa",
            ), f"Unexpected module_type={r.module_type} for role {r}"

    # -- Quantizer-type checks --
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

    def _check_q(mod, expected_cls, label=""):
        q = mod.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
        assert isinstance(q, expected_cls), (
            f"{mod.name}{' (' + label + ')' if label else ''}: "
            f"expected {expected_cls.__name__}, got {type(q).__name__}"
        )

    # Layer 0: all MXFP8
    _check_q(tl0.self_attention.layernorm_qkv, MXFP8Quantizer)
    _check_q(tl0.self_attention.proj, MXFP8Quantizer)

    # Layer 1: NVFP4 default, fc2 overridden to MXFP8
    _check_q(tl1.self_attention.layernorm_qkv, NVFP4Quantizer, "default")
    _check_q(tl1.self_attention.proj, NVFP4Quantizer, "default")
    assert any(
        r is not None and r.name == "tl1.layernorm_mlp.fc2" and r.tensor_type == "input"
        for r in recorded_roles
    ), "tl1.layernorm_mlp.fc2 input role not recorded"

    # Layer 2: block-scaling on qkv and fc1, NVFP4 on proj and fc2
    _check_q(tl2_mha.layernorm_qkv, Float8BlockQuantizer)
    _check_q(tl2_mha.proj, NVFP4Quantizer, "default")

    # Layer 3: current-scaling on proj, NVFP4 on everything else
    _check_q(tl3_proj, Float8CurrentScalingQuantizer)
    for mod in [tl3_qkv, tl3_fc1, tl3_fc2]:
        _check_q(mod, NVFP4Quantizer, "default")


def test_grouped_linear_module_type_dispatch():
    """Verify GroupedLinear emits module_type='grouped_linear' so factories can
    distinguish it from regular Linear (critical for MoE mixed-recipe dispatch)."""
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")

    torch.manual_seed(0)

    num_gemms = 2
    in_features = 64
    out_features = 64
    # Each per-GEMM M dim must be a multiple of 16 to satisfy cuBLAS FP8 GEMM's
    # leading-dimension alignment requirement on Hopper (sm_90).
    batch = 32
    m_splits = [batch // num_gemms] * num_gemms

    model = GroupedLinear(
        num_gemms, in_features, out_features, params_dtype=torch.bfloat16, name="experts"
    ).cuda()
    inp = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    recorded_roles = []

    def recording_factory(role):
        recorded_roles.append(role)
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

    custom_recipe = recipe.CustomRecipe(qfactory=recording_factory)

    with autocast(enabled=True, recipe=custom_recipe):
        out = model(inp, m_splits)
    loss = out.float().sum()
    loss.backward()

    non_none = [r for r in recorded_roles if r is not None]
    assert len(non_none) > 0, "No QuantizerRole objects recorded"
    for r in non_none:
        assert isinstance(r, QuantizerRole)
        assert (
            r.module_type == "grouped_linear"
        ), f"Expected module_type='grouped_linear', got '{r.module_type}'"
        assert r.name == "experts", f"Expected name='experts', got '{r.name}'"

    fwd_types = {r.tensor_type for r in non_none if r.tensor_type in ("input", "weight")}
    bwd_types = {r.tensor_type for r in non_none if r.tensor_type == "grad_output"}
    assert "input" in fwd_types, "Missing 'input' tensor_type in forward roles"
    assert "weight" in fwd_types, "Missing 'weight' tensor_type in forward roles"
    assert "grad_output" in bwd_types, "Missing 'grad_output' tensor_type in backward roles"


def test_delayed_scaling_request_wiring():
    """Shared buffers, correct views, Float8Quantizer instances."""
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    from transformer_engine.pytorch.quantization import (
        DelayedScalingRequest,
        CustomRecipeState,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.common.recipe import Format

    def ds_factory(role):
        return DelayedScalingRequest(fp8_format=Format.HYBRID, amax_history_len=16)

    custom_recipe = recipe.CustomRecipe(qfactory=ds_factory)

    # 3 quantizers (input, weight, output) like a Linear fwd
    state = CustomRecipeState(
        custom_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            QuantizerRole(module_type="linear", tensor_type="input"),
            QuantizerRole(module_type="linear", tensor_type="weight"),
            QuantizerRole(module_type="linear", tensor_type="output"),
        ],
    )
    quantizers = state.make_quantizers()

    # All quantizers should be Float8Quantizer
    assert len(quantizers) == 3
    for q in quantizers:
        assert isinstance(q, Float8Quantizer), f"Expected Float8Quantizer, got {type(q).__name__}"

    # Managed state should exist
    assert state._has_delayed_scaling
    assert state.scale is not None
    assert state.amax_history is not None

    # Shared buffers: scale shape = (3,), amax_history shape = (16, 3)
    assert state.scale.shape == (3,)
    assert state.amax_history.shape == (16, 3)

    # Each quantizer's scale should be a view into the shared buffer
    for i, q in enumerate(quantizers):
        assert q.scale.data_ptr() == state.scale[i].data_ptr()

    # Each quantizer's amax should be a view into amax_history[0]
    for i, q in enumerate(quantizers):
        assert q.amax.data_ptr() == state.amax_history[0][i].reshape((1,)).data_ptr()

    # Inner recipe should be a DelayedScaling
    inner = state._inner_delayed_scaling_recipe
    assert isinstance(inner, recipe.DelayedScaling)
    assert inner.amax_history_len == 16
    assert inner.fp8_format == Format.HYBRID


def test_custom_recipe_mixed_ds_and_stateless():
    """Mix DelayedScalingRequest + stateless quantizers in same CustomRecipeState."""
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    from transformer_engine.pytorch.quantization import (
        DelayedScalingRequest,
        CustomRecipeState,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.common.recipe import Format

    def mixed_factory(role):
        # Only weight gets delayed scaling, rest get current scaling
        if role is not None and role.tensor_type == "weight":
            return DelayedScalingRequest(fp8_format=Format.HYBRID)
        return Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")

    custom_recipe = recipe.CustomRecipe(qfactory=mixed_factory)

    # 3 quantizers: input(current), weight(DS), output(current)
    state = CustomRecipeState(
        custom_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            QuantizerRole(module_type="linear", tensor_type="input"),
            QuantizerRole(module_type="linear", tensor_type="weight"),
            QuantizerRole(module_type="linear", tensor_type="output"),
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 3

    # Slot 0 (input): current scaling
    assert isinstance(quantizers[0], Float8CurrentScalingQuantizer)
    # Slot 1 (weight): delayed scaling
    assert isinstance(quantizers[1], Float8Quantizer)
    # Slot 2 (output): current scaling
    assert isinstance(quantizers[2], Float8CurrentScalingQuantizer)

    # Only 1 DS request => shared buffers have size 1
    assert state._has_delayed_scaling
    assert state.scale.shape == (1,)
    assert state.amax_history.shape == (1024, 1)


def test_custom_recipe_ds_multi_step():
    """amax_history updates across multiple forward steps."""
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    from transformer_engine.pytorch.quantization import DelayedScalingRequest
    from transformer_engine.common.recipe import Format

    def ds_factory(role):
        return DelayedScalingRequest(fp8_format=Format.HYBRID)

    in_features = 128
    out_features = 128
    batch = 32
    num_steps = 3

    torch.manual_seed(99)
    model = Linear(in_features, out_features, params_dtype=torch.bfloat16, bias=False).cuda()
    custom = recipe.CustomRecipe(qfactory=ds_factory)

    amax_snapshots = []
    for step in range(num_steps):
        inp = torch.randn(
            batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with autocast(enabled=True, recipe=custom):
            out = model(inp)
        loss = out.float().sum()
        loss.backward()

        # Capture amax_history snapshot
        fwd_state = model.fp8_meta["scaling_fwd"]
        amax_snapshots.append(fwd_state.amax_history.clone())

    # After 3 steps, amax_history should have been updated at least once
    # The first row (amax_history[0]) should differ from the initial zeros
    # after the first step
    assert not torch.all(amax_snapshots[0] == 0), "amax_history should be updated after first step"


# ----------------------------------------------------------------------
# State preservation across role-driven rebuilds
# ----------------------------------------------------------------------
#
# Setting ``output_quantizer_role`` / ``grad_input_quantizer_role`` to a
# different value flips ``fp8_meta_tensors_initialized = False`` so the
# next ``set_meta_tensor`` call rebuilds the recipe state and quantizers
# with up-to-date roles. That rebuild MUST preserve persistent training
# buffers (delayed scaling's ``scale`` / ``amax_history``); otherwise
# checkpointed amax history is silently destroyed on the first forward
# pass after ``load_state_dict`` (when MHA wires boundary roles for the
# first time on the freshly-loaded module). The buffers must also be
# preserved by tensor-object identity, not just by value: the
# ``FP8GlobalStateManager`` reduction buffer holds a direct reference to
# the tensor created at first init, so any rebuild that allocates fresh
# tensors would break amax all-reduce.


def test_role_change_preserves_delayed_scaling_state():
    """Built-in DelayedScaling: role-driven rebuild preserves scale / amax_history.

    Stashes sentinel values into the buffers, forces a rebuild via the role
    setter, and verifies values + tensor-object identity survive.
    """
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    torch.manual_seed(0)
    model = Linear(64, 64, params_dtype=torch.bfloat16, bias=False).cuda()
    inp = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fp8_recipe = recipe.DelayedScaling(amax_history_len=8)

    # Initialize state via a forward pass.
    with autocast(enabled=True, recipe=fp8_recipe):
        model(inp).float().sum().backward()
    assert model.fp8_meta_tensors_initialized

    state_before = model.fp8_meta["scaling_fwd"]
    state_before.scale.fill_(3.14)
    state_before.amax_history.fill_(2.71)
    scale_obj_id = id(state_before.scale)
    amax_obj_id = id(state_before.amax_history)
    scale_data_ptr = state_before.scale.data_ptr()
    amax_data_ptr = state_before.amax_history.data_ptr()

    # Trigger role-driven invalidation. Setting a non-None role flips
    # ``fp8_meta_tensors_initialized = False`` so the next ``set_meta_tensor``
    # falls through and creates a fresh ``RecipeState``.
    model.output_quantizer_role = QuantizerRole(
        module_type="dpa", tensor_type="qkv", name="downstream"
    )
    assert not model.fp8_meta_tensors_initialized

    # Trigger the rebuild directly (no forward, so we can compare buffers exactly).
    model.init_fp8_meta_tensors(fp8_recipe)
    assert model.fp8_meta_tensors_initialized

    state_after = model.fp8_meta["scaling_fwd"]
    assert state_after is not state_before, "state should have been rebuilt"
    # Tensor objects must be inherited (not freshly allocated) so the
    # FP8GlobalStateManager reduction buffer's reference stays valid.
    assert (
        id(state_after.scale) == scale_obj_id
    ), "scale tensor object replaced by rebuild; global reduction buffer would dangle"
    assert id(state_after.amax_history) == amax_obj_id
    assert state_after.scale.data_ptr() == scale_data_ptr
    assert state_after.amax_history.data_ptr() == amax_data_ptr
    # Sentinel values must be preserved.
    assert state_after.scale.eq(3.14).all(), "scale was wiped by role-driven rebuild"
    assert state_after.amax_history.eq(2.71).all(), "amax_history was wiped"


def test_role_change_preserves_custom_delayed_scaling_state():
    """CustomRecipe + DelayedScalingRequest: role-driven rebuild preserves inner DSRS.

    Same property as the built-in case, but for the
    ``CustomRecipeState`` -> composed ``DelayedScalingRecipeState`` path.
    The inner DS state must be re-used across the rebuild so its
    accumulated buffers (and any external references to them) survive.
    """
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    from transformer_engine.pytorch.quantization import (
        CustomRecipeState,
        DelayedScalingRequest,
    )
    from transformer_engine.common.recipe import Format

    def ds_factory(role):
        return DelayedScalingRequest(fp8_format=Format.HYBRID, amax_history_len=8)

    torch.manual_seed(0)
    model = Linear(64, 64, params_dtype=torch.bfloat16, bias=False).cuda()
    inp = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    custom_recipe = recipe.CustomRecipe(qfactory=ds_factory)

    # Initialize state via a forward pass.
    with autocast(enabled=True, recipe=custom_recipe):
        model(inp).float().sum().backward()
    assert model.fp8_meta_tensors_initialized

    state_before = model.fp8_meta["scaling_fwd"]
    assert isinstance(state_before, CustomRecipeState)
    assert state_before._has_delayed_scaling
    inner_before = state_before._ds_state
    inner_before.scale.fill_(3.14)
    inner_before.amax_history.fill_(2.71)
    scale_obj_id = id(inner_before.scale)
    amax_obj_id = id(inner_before.amax_history)

    # Trigger role-driven invalidation.
    model.output_quantizer_role = QuantizerRole(
        module_type="dpa", tensor_type="qkv", name="downstream"
    )
    assert not model.fp8_meta_tensors_initialized

    # Rebuild.
    model.init_fp8_meta_tensors(custom_recipe)
    assert model.fp8_meta_tensors_initialized

    state_after = model.fp8_meta["scaling_fwd"]
    assert isinstance(state_after, CustomRecipeState)
    assert state_after is not state_before, "outer CustomRecipeState should have been rebuilt"
    assert state_after._has_delayed_scaling, "rebuild lost the inner DS state"
    inner_after = state_after._ds_state
    # Inner DSRS object identity is preserved (we reuse the existing inner state),
    # which means its buffers' tensor objects are also preserved.
    assert (
        inner_after is inner_before
    ), "inner DSRS replaced; FP8GlobalStateManager reduction buffer would dangle"
    assert id(inner_after.scale) == scale_obj_id
    assert id(inner_after.amax_history) == amax_obj_id
    # Sentinel values preserved.
    assert inner_after.scale.eq(3.14).all()
    assert inner_after.amax_history.eq(2.71).all()


def test_role_change_does_not_invalidate_when_role_unchanged():
    """Setting the role to its current value is a no-op (no rebuild)."""
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    torch.manual_seed(0)
    model = Linear(64, 64, params_dtype=torch.bfloat16, bias=False).cuda()
    inp = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fp8_recipe = recipe.DelayedScaling(amax_history_len=8)

    role = QuantizerRole(module_type="dpa", tensor_type="qkv", name="x")
    model.output_quantizer_role = role  # initial set: state not yet built, no-op

    with autocast(enabled=True, recipe=fp8_recipe):
        model(inp).float().sum().backward()
    assert model.fp8_meta_tensors_initialized

    # Re-setting the same role value must not invalidate.
    model.output_quantizer_role = QuantizerRole(module_type="dpa", tensor_type="qkv", name="x")
    assert (
        model.fp8_meta_tensors_initialized
    ), "Setting role to an equal value should be a no-op (frozen-dataclass __eq__)"


def test_custom_recipe_dpa_fp8():
    """DotProductAttention forward+backward with CustomRecipe and role-based mixed quantizers.

    Uses the nvfp4_linear_fp8_dpa_factory which dispatches:
      * DPA S/dP slots -> DelayedScalingRequest (stateful)
      * DPA QKV/O/dO/dQKV slots -> Float8CurrentScalingQuantizer
      * Linear slots -> NVFP4Quantizer
    """
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")
    if not te.is_nvfp4_available():
        pytest.skip("NVFP4 unsupported on this device")

    from transformer_engine.pytorch.utils import get_device_compute_capability

    cc = get_device_compute_capability()
    if cc < (9, 0) or cc >= (12, 0):
        pytest.skip(f"FP8 attention not supported on sm{cc[0]*10+cc[1]}")

    from transformer_engine.pytorch.quantization import (
        DelayedScalingRequest,
        CustomRecipeState,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8Quantizer,
        Float8CurrentScalingQuantizer,
    )
    from transformer_engine.pytorch.custom_recipes.quantization_factory_examples import (
        nvfp4_linear_fp8_dpa_factory,
    )

    torch.manual_seed(42)

    H = 64
    NH = 4
    KV = H // NH
    B = 2
    S = 32

    # Build a small model: Linear -> DPA -> Linear
    qkv_proj = Linear(H, 3 * H, params_dtype=torch.bfloat16, bias=False, name="qkv").cuda()
    dpa = te.DotProductAttention(
        NH, KV, attention_dropout=0.0, qkv_format="bshd", name="core_attention"
    )
    out_proj = Linear(H, H, params_dtype=torch.bfloat16, bias=False, name="proj").cuda()

    custom_recipe = recipe.CustomRecipe(
        qfactory=nvfp4_linear_fp8_dpa_factory,
        fp8_dpa=True,
    )

    inp = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with autocast(enabled=True, recipe=custom_recipe):
        qkv = qkv_proj(inp).view(B, S, 3, NH, KV)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn_out = dpa(q, k, v, qkv_format="bshd").reshape(B, S, H)
        out = out_proj(attn_out)

    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None, "Input gradient should exist"

    # Verify DPA recipe state is CustomRecipeState
    fwd_state = dpa.fp8_meta["scaling_fwd"]
    assert isinstance(
        fwd_state, CustomRecipeState
    ), f"Expected CustomRecipeState for DPA fwd, got {type(fwd_state).__name__}"

    # Verify DPA quantizers: 9 forward slots (3 GEMMs x 3)
    fwd_quantizers = dpa.quantizers["scaling_fwd"]
    assert len(fwd_quantizers) == 9, f"Expected 9 fwd quantizers, got {len(fwd_quantizers)}"

    # Slots 0-2: QKV (GEMM1) -> current scaling (role: module_type="dpa")
    # Slots 3-5: O   (GEMM2) -> current scaling (role: name hint "dpa_output")
    # Slots 6-8: S   (GEMM3) -> delayed scaling (Float8Quantizer from DelayedScalingRequest)
    for i in range(6):
        assert isinstance(fwd_quantizers[i], Float8CurrentScalingQuantizer), (
            f"Slot {i} (QKV/O): expected Float8CurrentScalingQuantizer, "
            f"got {type(fwd_quantizers[i]).__name__}"
        )
    for i in range(6, 9):
        assert isinstance(fwd_quantizers[i], Float8Quantizer), (
            f"Slot {i} (S): expected Float8Quantizer (delayed scaling), "
            f"got {type(fwd_quantizers[i]).__name__}"
        )

    # Verify DS state exists for the S/dP delayed scaling requests
    assert fwd_state._has_delayed_scaling, "DPA fwd state should have delayed scaling for S slots"

    # Verify backward quantizers exist too
    bwd_quantizers = dpa.quantizers["scaling_bwd"]
    assert len(bwd_quantizers) == 6, f"Expected 6 bwd quantizers, got {len(bwd_quantizers)}"

    # Slots 0-1: dQKV (GEMM1) -> current scaling (role: name hint "dpa_grad_input")
    # Slots 2-3: dO   (GEMM2) -> current scaling (role: module_type="dpa")
    # Slots 4-5: dP   (GEMM3) -> delayed scaling
    for i in range(4):
        assert isinstance(bwd_quantizers[i], Float8CurrentScalingQuantizer), (
            f"Bwd slot {i} (dQKV/dO): expected Float8CurrentScalingQuantizer, "
            f"got {type(bwd_quantizers[i]).__name__}"
        )
    for i in range(4, 6):
        assert isinstance(bwd_quantizers[i], Float8Quantizer), (
            f"Bwd slot {i} (dP): expected Float8Quantizer (delayed scaling), "
            f"got {type(bwd_quantizers[i]).__name__}"
        )

    # Linear modules should have CustomRecipeState with NVFP4 quantizers
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    qkv_fwd = qkv_proj.fp8_meta["scaling_fwd"]
    assert isinstance(
        qkv_fwd, CustomRecipeState
    ), f"Expected CustomRecipeState for qkv_proj, got {type(qkv_fwd).__name__}"
    qkv_fwd_quantizers = qkv_proj.quantizers["scaling_fwd"]
    for i, q in enumerate(qkv_fwd_quantizers):
        if q is not None:
            assert isinstance(
                q, NVFP4Quantizer
            ), f"qkv_proj fwd slot {i}: expected NVFP4Quantizer, got {type(q).__name__}"


def test_custom_recipe_dpa_mxfp8():
    """DotProductAttention forward+backward with CustomRecipe and MXFP8 attention.

    Uses the nvfp4_linear_mxfp8_dpa_factory which dispatches:
      * DPA roles (QKV/O/S/dO/dP/dQKV) -> MXFP8Quantizer (S/dP later nulled
        out by ``get_attention_quantizers`` since the MXFP8 fused-attention
        kernel handles those slots internally)
      * DPA boundary hints -> MXFP8Quantizer
      * Linear slots -> NVFP4Quantizer

    Mirrors the documented "NVFP4 linear + MXFP8 attention" combo from
    ``dot_product_attention.py``'s recipe-combination table.
    """
    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported on this device: {reason}")
    if not te.is_mxfp8_available():
        pytest.skip("MXFP8 unsupported on this device")
    if not te.is_nvfp4_available():
        pytest.skip("NVFP4 unsupported on this device")

    from transformer_engine.pytorch.utils import get_device_compute_capability

    cc = get_device_compute_capability()
    if cc < (9, 0) or cc >= (12, 0):
        pytest.skip(f"FP8 attention not supported on sm{cc[0]*10+cc[1]}")

    from transformer_engine.pytorch.quantization import CustomRecipeState
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    from transformer_engine.pytorch.custom_recipes.quantization_factory_examples import (
        nvfp4_linear_mxfp8_dpa_factory,
    )

    torch.manual_seed(42)

    # MXFP8 fused attention requires s_q % 128 == 0, s_kv % 128 == 0,
    # d_qk % 32 == 0, d_v % 32 == 0.
    H = 128
    NH = 4
    KV = H // NH  # 32
    B = 2
    S = 128

    # Build a small model: Linear -> DPA -> Linear
    qkv_proj = Linear(H, 3 * H, params_dtype=torch.bfloat16, bias=False, name="qkv").cuda()
    dpa = te.DotProductAttention(
        NH, KV, attention_dropout=0.0, qkv_format="bshd", name="core_attention"
    )
    out_proj = Linear(H, H, params_dtype=torch.bfloat16, bias=False, name="proj").cuda()

    custom_recipe = recipe.CustomRecipe(
        qfactory=nvfp4_linear_mxfp8_dpa_factory,
        fp8_dpa=True,
    )

    inp = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with autocast(enabled=True, recipe=custom_recipe):
        qkv = qkv_proj(inp).view(B, S, 3, NH, KV)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # MXFP8 fused attention requires s_q % 128 == 0, s_kv % 128 == 0,
        # d_qk % 32 == 0, d_v % 32 == 0. The B/S/H values above are picked
        # to satisfy all four constraints (S=128, KV=32).
        attn_out = dpa(q, k, v, qkv_format="bshd").reshape(B, S, H)
        out = out_proj(attn_out)

    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None, "Input gradient should exist"

    # DPA recipe state should be CustomRecipeState
    fwd_state = dpa.fp8_meta["scaling_fwd"]
    assert isinstance(
        fwd_state, CustomRecipeState
    ), f"Expected CustomRecipeState for DPA fwd, got {type(fwd_state).__name__}"

    # All DPA slots should resolve to MXFP8Quantizer (the factory returns MXFP8
    # uniformly for DPA roles; S/dP nulling happens inside get_attention_quantizers
    # at fused-attn dispatch time, not here).
    fwd_quantizers = dpa.quantizers["scaling_fwd"]
    assert len(fwd_quantizers) == 9, f"Expected 9 fwd quantizers, got {len(fwd_quantizers)}"
    for i, q in enumerate(fwd_quantizers):
        assert isinstance(
            q, MXFP8Quantizer
        ), f"DPA fwd slot {i}: expected MXFP8Quantizer, got {type(q).__name__}"

    bwd_quantizers = dpa.quantizers["scaling_bwd"]
    assert len(bwd_quantizers) == 6, f"Expected 6 bwd quantizers, got {len(bwd_quantizers)}"
    for i, q in enumerate(bwd_quantizers):
        assert isinstance(
            q, MXFP8Quantizer
        ), f"DPA bwd slot {i}: expected MXFP8Quantizer, got {type(q).__name__}"

    # MXFP8 attention has no delayed-scaling state (no S/dP DS-request slots).
    assert (
        not fwd_state._has_delayed_scaling
    ), "DPA fwd state should NOT have delayed scaling for the all-MXFP8 factory"

    # Linear modules should still be NVFP4
    qkv_fwd = qkv_proj.fp8_meta["scaling_fwd"]
    assert isinstance(
        qkv_fwd, CustomRecipeState
    ), f"Expected CustomRecipeState for qkv_proj, got {type(qkv_fwd).__name__}"
    qkv_fwd_quantizers = qkv_proj.quantizers["scaling_fwd"]
    for i, q in enumerate(qkv_fwd_quantizers):
        if q is not None:
            assert isinstance(
                q, NVFP4Quantizer
            ), f"qkv_proj fwd slot {i}: expected NVFP4Quantizer, got {type(q).__name__}"


def test_custom_recipe_debug_tool_compat():
    """Custom recipe quantizers should work when wrapped by DebugQuantizer.

    Verifies that the debug tool (nvdlfw_inspect) can wrap custom-recipe
    quantizers produced via QuantizerRole dispatch without errors.
    """
    try:
        import nvdlfw_inspect.api as debug_api
    except ImportError:
        pytest.skip("nvdlfw_inspect not installed")

    available, reason = te.is_fp8_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 unsupported: {reason}")

    import pathlib
    import tempfile

    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    te_debug_features = str(
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "transformer_engine"
        / "debug"
        / "features"
    )

    # Log config that keeps DebugQuantizer active (not bypassed by no_debug_features_active)
    log_config = """log:
  layers:
    layer_types: [linear]
  enabled: True
  transformer_engine:
    LogTensorStats:
      enabled: True
      tensors: [activation, weight]
      stats: [max]
      start_step: 0
      end_step: 3
"""

    torch.manual_seed(0)

    in_features = 64
    out_features = 64
    batch = 16

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as cfg:
        cfg.write(log_config)
        cfg.flush()
        config_path = cfg.name

    try:
        with tempfile.TemporaryDirectory() as log_dir:
            debug_api.initialize(
                config_file=config_path,
                feature_dirs=te_debug_features,
                log_dir=log_dir,
            )

            model = Linear(
                in_features, out_features, params_dtype=torch.bfloat16, name="layer"
            ).cuda()

            custom_recipe = recipe.CustomRecipe(qfactory=current_scaling_quantizer_factory)

            assert TEDebugState.debug_enabled, "Debug mode should be active"

            for _ in range(3):
                inp_step = torch.randn(
                    batch, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                with autocast(enabled=True, recipe=custom_recipe):
                    out = model(inp_step)
                out.float().sum().backward()
                debug_api.step()

            assert inp_step.grad is not None, "Input gradient should exist"

            log_files = list(pathlib.Path(log_dir).rglob("*.log"))
            assert (
                len(log_files) > 0
            ), f"Debug log output expected in {log_dir} but no .log files found"
    finally:
        debug_api.end_debug()
        TEDebugState._reset()
        import os

        os.unlink(config_path)


# ----------------------------------------------------------------------
# Role-aware dispatch in built-in block-scaling recipe states
# ----------------------------------------------------------------------
#
# These tests exercise ``Float8BlockScalingRecipeState.make_quantizers`` and
# ``NVFP4BlockScalingRecipeState.make_quantizers`` directly to verify that
# per-slot dispatch is driven by ``QuantizerRole.tensor_type`` with a
# positional fallback that matches the legacy behavior. They construct the
# recipe state objects directly (no autocast / no fwd pass) so they don't
# depend on any module's ``get_quantizer_roles`` implementation.


def _fp8block_role(tensor_type):
    """QuantizerRole helper for FP8-block tests."""
    return QuantizerRole(module_type="linear", tensor_type=tensor_type, name="t")


def test_fp8block_recipe_state_role_dispatch_forward():
    """Forward dispatch: input/output -> x cfg, weight -> w cfg."""
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    from transformer_engine.pytorch.quantization import Float8BlockScalingRecipeState

    fp8_recipe = recipe.Float8BlockScaling()
    state = Float8BlockScalingRecipeState(
        fp8_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            _fp8block_role("input"),
            _fp8block_role("weight"),
            _fp8block_role("output"),
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 3
    # input slot uses x cfg
    assert quantizers[0].block_scaling_dim == fp8_recipe.x_block_scaling_dim
    # weight slot uses w cfg
    assert quantizers[1].block_scaling_dim == fp8_recipe.w_block_scaling_dim
    # output slot mirrors input cfg (legacy behavior preserved)
    assert quantizers[2].block_scaling_dim == fp8_recipe.x_block_scaling_dim
    # Sanity: the recipe defaults distinguish x and w block scaling dims so
    # the test would fail if dispatch were uniform.
    assert fp8_recipe.x_block_scaling_dim != fp8_recipe.w_block_scaling_dim


def test_fp8block_recipe_state_role_dispatch_backward():
    """Backward dispatch: grad_output / grad_input both -> grad cfg."""
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    from transformer_engine.pytorch.quantization import Float8BlockScalingRecipeState

    fp8_recipe = recipe.Float8BlockScaling()
    state = Float8BlockScalingRecipeState(
        fp8_recipe,
        mode="backward",
        num_quantizers=2,
        roles=[
            _fp8block_role("grad_output"),
            _fp8block_role("grad_input"),
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 2
    for q in quantizers:
        assert q.block_scaling_dim == fp8_recipe.grad_block_scaling_dim


def test_fp8block_recipe_state_positional_fallback_matches_explicit_roles():
    """``roles=None`` produces the same per-slot configs as explicit ``[input, weight, output]``."""
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    from transformer_engine.pytorch.quantization import Float8BlockScalingRecipeState

    fp8_recipe = recipe.Float8BlockScaling()

    explicit = Float8BlockScalingRecipeState(
        fp8_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            _fp8block_role("input"),
            _fp8block_role("weight"),
            _fp8block_role("output"),
        ],
    ).make_quantizers()

    fallback = Float8BlockScalingRecipeState(
        fp8_recipe,
        mode="forward",
        num_quantizers=3,
        roles=None,
    ).make_quantizers()

    assert len(explicit) == len(fallback) == 3
    for a, b in zip(explicit, fallback):
        assert a.block_scaling_dim == b.block_scaling_dim
        assert a.dtype == b.dtype
        assert a.amax_epsilon == b.amax_epsilon
        assert a.force_pow_2_scales == b.force_pow_2_scales


def test_fp8block_recipe_state_supports_non_multiple_of_three():
    """Two-slot forward (fusible-Linear shape) used to fail ``% 3 == 0`` assert."""
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    from transformer_engine.pytorch.quantization import Float8BlockScalingRecipeState

    fp8_recipe = recipe.Float8BlockScaling()
    state = Float8BlockScalingRecipeState(
        fp8_recipe,
        mode="forward",
        num_quantizers=2,
        roles=[
            _fp8block_role("input"),
            _fp8block_role("weight"),
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 2
    assert quantizers[0].block_scaling_dim == fp8_recipe.x_block_scaling_dim
    assert quantizers[1].block_scaling_dim == fp8_recipe.w_block_scaling_dim


def test_fp8block_recipe_state_unknown_or_none_role_falls_back_positionally():
    """Per-slot ``None`` and unknown ``tensor_type`` use the positional pattern."""
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    from transformer_engine.pytorch.quantization import Float8BlockScalingRecipeState

    fp8_recipe = recipe.Float8BlockScaling()
    # Slot 0: bare role (empty tensor_type) -> positional "input" -> x cfg
    # Slot 1: unknown tensor_type "qkv" (DPA-style) -> positional "weight" -> w cfg
    # Slot 2: None role -> positional "output" -> x cfg
    state = Float8BlockScalingRecipeState(
        fp8_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            QuantizerRole(),
            QuantizerRole(module_type="dpa", tensor_type="qkv"),
            None,
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 3
    assert quantizers[0].block_scaling_dim == fp8_recipe.x_block_scaling_dim
    assert quantizers[1].block_scaling_dim == fp8_recipe.w_block_scaling_dim
    assert quantizers[2].block_scaling_dim == fp8_recipe.x_block_scaling_dim


def _nvfp4_role(tensor_type):
    return QuantizerRole(module_type="linear", tensor_type=tensor_type, name="t")


def test_nvfp4_recipe_state_role_dispatch_forward():
    """Forward dispatch: input/output -> inp cfg (RHT, 1D), weight -> weight cfg (no RHT, 2D)."""
    if not torch.cuda.is_available() or not te.is_nvfp4_available():
        pytest.skip("NVFP4 unsupported on this device")

    from transformer_engine.pytorch.quantization import NVFP4BlockScalingRecipeState

    nvfp4_recipe = recipe.NVFP4BlockScaling()
    state = NVFP4BlockScalingRecipeState(
        nvfp4_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            _nvfp4_role("input"),
            _nvfp4_role("weight"),
            _nvfp4_role("output"),
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 3
    # input slot
    assert quantizers[0].with_rht == nvfp4_recipe.fp4_quant_fwd_inp.random_hadamard_transform
    assert quantizers[0].with_2d_quantization == nvfp4_recipe.fp4_quant_fwd_inp.fp4_2d_quantization
    # weight slot
    assert quantizers[1].with_rht == nvfp4_recipe.fp4_quant_fwd_weight.random_hadamard_transform
    assert (
        quantizers[1].with_2d_quantization == nvfp4_recipe.fp4_quant_fwd_weight.fp4_2d_quantization
    )
    # output slot mirrors input cfg
    assert quantizers[2].with_rht == nvfp4_recipe.fp4_quant_fwd_inp.random_hadamard_transform
    assert quantizers[2].with_2d_quantization == nvfp4_recipe.fp4_quant_fwd_inp.fp4_2d_quantization
    # Sanity: defaults distinguish input vs weight (RHT and 2D toggles differ).
    assert (
        nvfp4_recipe.fp4_quant_fwd_inp.random_hadamard_transform
        != nvfp4_recipe.fp4_quant_fwd_weight.random_hadamard_transform
    ) or (
        nvfp4_recipe.fp4_quant_fwd_inp.fp4_2d_quantization
        != nvfp4_recipe.fp4_quant_fwd_weight.fp4_2d_quantization
    )


def test_nvfp4_recipe_state_role_dispatch_backward():
    """Backward dispatch: any slot -> grad cfg (uniform)."""
    if not torch.cuda.is_available() or not te.is_nvfp4_available():
        pytest.skip("NVFP4 unsupported on this device")

    from transformer_engine.pytorch.quantization import NVFP4BlockScalingRecipeState

    nvfp4_recipe = recipe.NVFP4BlockScaling()
    state = NVFP4BlockScalingRecipeState(
        nvfp4_recipe,
        mode="backward",
        num_quantizers=2,
        roles=[
            _nvfp4_role("grad_output"),
            _nvfp4_role("grad_input"),
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 2
    for q in quantizers:
        assert q.with_rht == nvfp4_recipe.fp4_quant_bwd_grad.random_hadamard_transform
        assert q.with_2d_quantization == nvfp4_recipe.fp4_quant_bwd_grad.fp4_2d_quantization
        assert q.stochastic_rounding == nvfp4_recipe.fp4_quant_bwd_grad.stochastic_rounding


def test_nvfp4_recipe_state_positional_fallback_matches_explicit_roles():
    """``roles=None`` matches explicit ``[input, weight, output]`` slot-for-slot."""
    if not torch.cuda.is_available() or not te.is_nvfp4_available():
        pytest.skip("NVFP4 unsupported on this device")

    from transformer_engine.pytorch.quantization import NVFP4BlockScalingRecipeState

    nvfp4_recipe = recipe.NVFP4BlockScaling()

    explicit = NVFP4BlockScalingRecipeState(
        nvfp4_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            _nvfp4_role("input"),
            _nvfp4_role("weight"),
            _nvfp4_role("output"),
        ],
    ).make_quantizers()

    fallback = NVFP4BlockScalingRecipeState(
        nvfp4_recipe,
        mode="forward",
        num_quantizers=3,
        roles=None,
    ).make_quantizers()

    assert len(explicit) == len(fallback) == 3
    for a, b in zip(explicit, fallback):
        assert a.with_rht == b.with_rht
        assert a.with_post_rht_amax == b.with_post_rht_amax
        assert a.with_2d_quantization == b.with_2d_quantization
        assert a.stochastic_rounding == b.stochastic_rounding
        assert a.dtype == b.dtype


def test_nvfp4_recipe_state_supports_non_multiple_of_three():
    """Two-slot forward (fusible-Linear shape) succeeds with role-driven dispatch."""
    if not torch.cuda.is_available() or not te.is_nvfp4_available():
        pytest.skip("NVFP4 unsupported on this device")

    from transformer_engine.pytorch.quantization import NVFP4BlockScalingRecipeState

    nvfp4_recipe = recipe.NVFP4BlockScaling()
    state = NVFP4BlockScalingRecipeState(
        nvfp4_recipe,
        mode="forward",
        num_quantizers=2,
        roles=[
            _nvfp4_role("input"),
            _nvfp4_role("weight"),
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 2
    assert quantizers[0].with_rht == nvfp4_recipe.fp4_quant_fwd_inp.random_hadamard_transform
    assert quantizers[1].with_rht == nvfp4_recipe.fp4_quant_fwd_weight.random_hadamard_transform


def test_nvfp4_recipe_state_unknown_or_none_role_falls_back_positionally():
    """Per-slot ``None`` and unknown ``tensor_type`` use the positional pattern."""
    if not torch.cuda.is_available() or not te.is_nvfp4_available():
        pytest.skip("NVFP4 unsupported on this device")

    from transformer_engine.pytorch.quantization import NVFP4BlockScalingRecipeState

    nvfp4_recipe = recipe.NVFP4BlockScaling()
    # Slot 0: bare role (empty tensor_type) -> positional "input" -> inp cfg
    # Slot 1: DPA-style unknown tensor_type "qkv" -> positional "weight" -> weight cfg
    # Slot 2: None role -> positional "output" -> inp cfg
    state = NVFP4BlockScalingRecipeState(
        nvfp4_recipe,
        mode="forward",
        num_quantizers=3,
        roles=[
            QuantizerRole(),
            QuantizerRole(module_type="dpa", tensor_type="qkv"),
            None,
        ],
    )
    quantizers = state.make_quantizers()
    assert len(quantizers) == 3
    assert quantizers[0].with_rht == nvfp4_recipe.fp4_quant_fwd_inp.random_hadamard_transform
    assert quantizers[1].with_rht == nvfp4_recipe.fp4_quant_fwd_weight.random_hadamard_transform
    assert quantizers[2].with_rht == nvfp4_recipe.fp4_quant_fwd_inp.random_hadamard_transform


# ----------------------------------------------------------------------
# RecipeState._slot_role primitive
# ----------------------------------------------------------------------
#
# `_slot_role` is the primitive that role-driven recipe states use to
# resolve per-slot dispatch info. It returns the real role when one was
# provided and synthesizes one with the positional ``tensor_type`` fallback
# (and empty ``module_type``/``name``) otherwise. Future recipes that
# dispatch on ``module_type`` / ``name`` rely on this contract.
#
# We exercise these via a concrete ``Float8BlockScalingRecipeState`` since
# ``RecipeState`` is abstract; the helper itself is mode-aware but
# recipe-agnostic.


def _make_fp8block_state(*, mode, num_quantizers, roles):
    from transformer_engine.pytorch.quantization import Float8BlockScalingRecipeState

    return Float8BlockScalingRecipeState(
        recipe.Float8BlockScaling(),
        mode=mode,
        num_quantizers=num_quantizers,
        roles=roles,
    )


def test_slot_role_passes_real_role_through_unchanged():
    """A real ``QuantizerRole`` from the producer is returned as-is."""
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    real = QuantizerRole(module_type="linear", tensor_type="weight", name="layer37.fc1")
    state = _make_fp8block_state(mode="forward", num_quantizers=1, roles=[real])
    resolved = state._slot_role(0)
    # Identity: no copying, the real instance is returned.
    assert resolved is real
    assert resolved.module_type == "linear"
    assert resolved.tensor_type == "weight"
    assert resolved.name == "layer37.fc1"


def test_slot_role_passes_unknown_tensor_type_through_unchanged():
    """A real role with non-canonical ``tensor_type`` is NOT remapped by ``_slot_role``.

    ``_slot_tensor_type`` would fall back to positional, but ``_slot_role``
    must preserve the original so module-type / name dispatch still works.
    """
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    dpa_role = QuantizerRole(module_type="dpa", tensor_type="qkv", name="self_attention.dpa")
    state = _make_fp8block_state(mode="forward", num_quantizers=1, roles=[dpa_role])
    resolved = state._slot_role(0)
    assert resolved is dpa_role
    assert resolved.tensor_type == "qkv"  # unchanged, NOT folded into known set
    # ``_slot_tensor_type`` still falls back to positional pattern[0] = "input".
    assert state._slot_tensor_type(0) == "input"


def test_slot_role_returns_bare_role_when_per_slot_role_is_none():
    """Boundary slot (``roles[i] is None``) returns a bare ``QuantizerRole()``.

    The primitive does NOT synthesize a positional ``tensor_type`` — that's
    a tensor-type-dispatch policy owned by ``_slot_tensor_type``.
    """
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    real_input = QuantizerRole(module_type="linear", tensor_type="input", name="t")
    real_weight = QuantizerRole(module_type="linear", tensor_type="weight", name="t")
    # Slot 2 (output) is None: typical for Linear without parent setting
    # ``_output_quantizer_role``.
    state = _make_fp8block_state(
        mode="forward", num_quantizers=3, roles=[real_input, real_weight, None]
    )
    # Real slots pass through.
    assert state._slot_role(0) is real_input
    assert state._slot_role(1) is real_weight
    # None slot returns a bare QuantizerRole(): all fields empty, no
    # tensor-type-specific synthesis.
    bare = state._slot_role(2)
    assert bare.tensor_type == ""
    assert bare.module_type == ""
    assert bare.name == ""
    # Consumers get positional fallback through _slot_tensor_type, not _slot_role.
    assert state._slot_tensor_type(2) == "output"


def test_slot_role_returns_bare_role_when_roles_list_is_none():
    """``roles=None`` yields bare ``QuantizerRole()`` for every slot, fwd and bwd.

    Positional fallback for tensor types lives in ``_slot_tensor_type``, not here.
    """
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    fwd = _make_fp8block_state(mode="forward", num_quantizers=4, roles=None)
    # _slot_role is field-agnostic: every slot is a bare QuantizerRole().
    for i in range(4):
        role = fwd._slot_role(i)
        assert role.tensor_type == ""
        assert role.module_type == ""
        assert role.name == ""
    # _slot_tensor_type applies the positional fallback (with wrap).
    fwd_types = [fwd._slot_tensor_type(i) for i in range(4)]
    assert fwd_types == ["input", "weight", "output", "input"]

    bwd = _make_fp8block_state(mode="backward", num_quantizers=3, roles=None)
    for i in range(3):
        assert bwd._slot_role(i).tensor_type == ""
    bwd_types = [bwd._slot_tensor_type(i) for i in range(3)]
    assert bwd_types == ["grad_output", "grad_input", "grad_output"]


def test_slot_role_supports_module_type_only_role():
    """A role that fills ONLY ``module_type`` is preserved as-is.

    This is the producer convention for future module-type-driven recipes:
    fill only the field(s) you have signal for. ``_slot_role`` must not
    invent a ``tensor_type`` to mask the empty one (otherwise the module-type
    branch in a mixed recipe would never see a clean signal).
    """
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not torch.cuda.is_available() or not available:
        pytest.skip(f"FP8 block scaling unsupported: {reason}")

    moe = QuantizerRole(module_type="moe_expert")
    state = _make_fp8block_state(mode="forward", num_quantizers=1, roles=[moe])
    resolved = state._slot_role(0)
    assert resolved is moe
    assert resolved.module_type == "moe_expert"
    assert resolved.tensor_type == ""  # NOT auto-filled
    assert resolved.name == ""
    # Tensor-type-only recipes fall back to positional for this slot.
    assert state._slot_tensor_type(0) == "input"
