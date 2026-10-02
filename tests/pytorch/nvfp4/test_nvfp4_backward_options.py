# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for NVFP4BlockScaling backward options (MXFP8 dgrad and gradient 4over6).

Also covers zero-valued grad_output rows and blocks, which RL losses produce for loss-masked
prompt and padding tokens and for micro-batches whose advantages are all zero.
"""

import os
import subprocess
import sys

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import CustomRecipe, NVFP4BlockScaling
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.quantization import RecipeState
from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
requires_nvfp4 = pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
requires_nvfp4_and_mxfp8 = pytest.mark.skipif(
    not (nvfp4_available and mxfp8_available),
    reason=reason_for_no_nvfp4 if not nvfp4_available else reason_for_no_mxfp8,
)

IN_FEATURES = 256
OUT_FEATURES = 512
M_SPLITS = [128, 256, 128]
NUM_TOKENS = sum(M_SPLITS)

# References are FP32 GEMMs on dequantized operands; the tolerance covers BF16 output rounding.
BF16_OUTPUT_TOLS = {"rtol": 1.6e-2, "atol": 1e-5}


def make_recipe(**kwargs) -> NVFP4BlockScaling:
    """NVFP4 recipe with RHT, stochastic rounding, and 2D weight quantization disabled by default."""
    kwargs = {
        "disable_rht": True,
        "disable_stochastic_rounding": True,
        "disable_2d_quantization": True,
        **kwargs,
    }
    return NVFP4BlockScaling(**kwargs)


def nvfp4_quantizer(row_scaled: bool = False, use_4over6: bool = False) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=DType.kFloat4E2M1,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=row_scaled,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=256 if use_4over6 else 448,
    )


def mxfp8_quantizer() -> MXFP8Quantizer:
    return MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3)


def dgrad_mxfp8_custom_recipe(row_scaled_activation: bool = False) -> CustomRecipe:
    """CustomRecipe equivalent to ``make_recipe(dgrad_mxfp8=True)``."""

    def qfactory(role):
        is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
        if is_linear and role.tensor_type == "input":
            return nvfp4_quantizer(row_scaled=row_scaled_activation)
        if is_linear and role.tensor_type == "weight":
            return HybridQuantizer(
                rowwise_quantizer=nvfp4_quantizer(),
                columnwise_quantizer=mxfp8_quantizer(),
                columnwise_source="rowwise_dequantized",
            )
        if is_linear and role.tensor_type == "grad_output":
            return HybridQuantizer(
                rowwise_quantizer=mxfp8_quantizer(),
                columnwise_quantizer=nvfp4_quantizer(),
            )
        return nvfp4_quantizer()

    return CustomRecipe(qfactory=qfactory)


def make_quantizers(recipe, mode: str, num_quantizers: int) -> list:
    return RecipeState.create(recipe, mode=mode, num_quantizers=num_quantizers).make_quantizers()


def make_module(kind: str, bias: bool = False, activation: str = "gelu") -> torch.nn.Module:
    torch.manual_seed(1234)
    kwargs = {"bias": bias, "params_dtype": torch.bfloat16}
    if kind == "linear":
        return te.Linear(IN_FEATURES, OUT_FEATURES, **kwargs)
    if kind == "layernorm_linear":
        return te.LayerNormLinear(IN_FEATURES, OUT_FEATURES, normalization="RMSNorm", **kwargs)
    if kind == "layernorm_mlp":
        return te.LayerNormMLP(IN_FEATURES, OUT_FEATURES, activation=activation, **kwargs)
    if kind == "grouped_linear":
        return te.GroupedLinear(len(M_SPLITS), IN_FEATURES, OUT_FEATURES, **kwargs)
    raise ValueError(f"Unknown module kind: {kind}")


def make_inputs(out_features: int = OUT_FEATURES):
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(NUM_TOKENS, IN_FEATURES, device="cuda", generator=generator)
    dy = torch.randn(NUM_TOKENS, out_features, device="cuda", generator=generator) * 1e-3
    return x.to(torch.bfloat16), dy.to(torch.bfloat16)


def default_forward(module, x):
    if isinstance(module, te.GroupedLinear):
        return module(x, M_SPLITS)
    if isinstance(module, te_ops.GroupedLinear):
        return module(x, torch.tensor(M_SPLITS, device="cuda"))
    return module(x)


def run_step(module, recipe, x, dy, forward=default_forward):
    """One forward and backward step. Returns the output, dgrad, and parameter gradients.

    Parameters with ``main_grad`` (fused wgrad accumulation) report ``main_grad``.
    """
    x = x.detach().clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=recipe):
        y = forward(module, x)
    y.backward(dy)
    if (
        getattr(module, "wgrad_store", None) is not None
        and module.wgrad_store.delay_wgrad_compute()
    ):
        module.backward_dw()
    grads = {
        name: (p.main_grad if hasattr(p, "main_grad") else p.grad).detach().clone()
        for name, p in module.named_parameters()
    }
    return y.detach(), x.grad.detach(), grads


def assert_steps_equal(step, step_ref):
    y, dx, grads = step
    y_ref, dx_ref, grads_ref = step_ref
    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
    torch.testing.assert_close(dx, dx_ref, rtol=0, atol=0)
    assert grads.keys() == grads_ref.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, grads_ref[name], rtol=0, atol=0)


def rel_err(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x.float() - ref.float()).norm() / ref.float().norm()).item()


def dequantize(quantizer, tensor: torch.Tensor) -> torch.Tensor:
    """Rowwise-quantize and dequantize to FP32."""
    quantizer.set_usage(rowwise=True, columnwise=False)
    return quantizer(tensor).dequantize(dtype=torch.float32)


def dequantize_columnwise(quantizer, tensor: torch.Tensor) -> torch.Tensor:
    """Emulate columnwise quantization by quantizing the transpose rowwise; dequantize to FP32."""
    return dequantize(quantizer, tensor.t().contiguous()).t()


def test_recipe_env_vars():
    """Frameworks that construct a default NVFP4BlockScaling can opt in through env vars."""
    env = dict(os.environ, NVTE_NVFP4_DGRAD_MXFP8="1", NVTE_NVFP4_4OVER6_GRAD="1")
    code = (
        "from transformer_engine.common.recipe import NVFP4BlockScaling\n"
        "recipe = NVFP4BlockScaling()\n"
        "assert recipe.dgrad_mxfp8 and recipe.nvfp4_4over6_grad, recipe\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_dgrad_mxfp8_rejects_backward_override():
    # pydantic reports failed __post_init__ assertions as ValidationError, a ValueError.
    with pytest.raises(ValueError, match="NVTE_NVFP4_DGRAD_MXFP8"):
        make_recipe(dgrad_mxfp8=True, backward_override="high_precision")


@requires_nvfp4
def test_4over6_grad_rejects_rht_and_stochastic_rounding():
    recipe = NVFP4BlockScaling(
        nvfp4_4over6_grad=True, disable_rht=False, disable_stochastic_rounding=False
    )
    with pytest.raises(ValueError, match="4over6"):
        make_quantizers(recipe, "backward", 2)


@requires_nvfp4_and_mxfp8
def test_dgrad_mxfp8_quantizers():
    recipe = make_recipe(dgrad_mxfp8=True, row_scaled_activation=True)
    input_quantizer, weight_quantizer, _ = make_quantizers(recipe, "forward", 3)
    assert isinstance(input_quantizer, NVFP4Quantizer)
    assert input_quantizer.row_scaled_nvfp4
    assert isinstance(weight_quantizer, HybridQuantizer)
    assert isinstance(weight_quantizer.rowwise_quantizer, NVFP4Quantizer)
    assert isinstance(weight_quantizer.columnwise_quantizer, MXFP8Quantizer)
    assert weight_quantizer.columnwise_source == "rowwise_dequantized"

    grad_output_quantizer, grad_input_quantizer = make_quantizers(recipe, "backward", 2)
    assert isinstance(grad_output_quantizer, HybridQuantizer)
    assert isinstance(grad_output_quantizer.rowwise_quantizer, MXFP8Quantizer)
    assert isinstance(grad_output_quantizer.columnwise_quantizer, NVFP4Quantizer)
    assert grad_output_quantizer.columnwise_source == "original"
    assert not grad_output_quantizer.columnwise_quantizer.row_scaled_nvfp4
    assert isinstance(grad_input_quantizer, NVFP4Quantizer)

    # The NVFP4 wgrad operand keeps the recipe's gradient RHT and stochastic rounding.
    recipe = NVFP4BlockScaling(
        dgrad_mxfp8=True, disable_rht=False, disable_stochastic_rounding=False
    )
    grad_output_quantizer, _ = make_quantizers(recipe, "backward", 2)
    assert grad_output_quantizer.columnwise_quantizer.with_rht
    assert grad_output_quantizer.columnwise_quantizer.stochastic_rounding


@requires_nvfp4_and_mxfp8
@pytest.mark.parametrize("e4m3_use_256, e4m3_max", [("all", 256), ("activations", 448)])
def test_4over6_grad_quantizers(e4m3_use_256, e4m3_max):
    recipe = make_recipe(nvfp4_4over6_grad=True, nvfp4_4over6_e4m3_use_256=e4m3_use_256)
    for quantizer in make_quantizers(recipe, "backward", 2):
        assert quantizer.nvfp4_use_4over6
        assert quantizer.nvfp4_e4m3_max == e4m3_max
    for quantizer in make_quantizers(recipe, "forward", 3):
        assert not quantizer.nvfp4_use_4over6

    # nvfp4_4over6="all" does not cover gradients.
    grad_output_quantizer, _ = make_quantizers(make_recipe(nvfp4_4over6="all"), "backward", 2)
    assert not grad_output_quantizer.nvfp4_use_4over6

    # With MXFP8 dgrad, 4over6 applies to the NVFP4 wgrad operand.
    recipe = make_recipe(nvfp4_4over6_grad=True, dgrad_mxfp8=True)
    grad_output_quantizer, _ = make_quantizers(recipe, "backward", 2)
    assert grad_output_quantizer.columnwise_quantizer.nvfp4_use_4over6


@requires_nvfp4_and_mxfp8
def test_dgrad_mxfp8_linear_reference():
    """te.Linear matches GEMMs on dequantized operands.

    fprop and wgrad use NVFP4 operands; dgrad uses MXFP8 operands, with the weight requantized
    from its dequantized NVFP4 value. The NVFP4 dgrad and the MXFP8 weight quantized from the
    original weight are negative controls.
    """
    x, dy = make_inputs()
    module = make_module("linear")
    y, dx, grads = run_step(module, make_recipe(dgrad_mxfp8=True), x, dy)

    w = module.weight.detach()
    w_nvfp4 = dequantize(nvfp4_quantizer(), w)
    y_ref = dequantize(nvfp4_quantizer(), x) @ w_nvfp4.t()
    dx_ref = dequantize(mxfp8_quantizer(), dy) @ dequantize_columnwise(
        mxfp8_quantizer(), w_nvfp4.to(torch.bfloat16)
    )
    dw_ref = dequantize_columnwise(nvfp4_quantizer(), dy).t() @ dequantize_columnwise(
        nvfp4_quantizer(), x
    )
    torch.testing.assert_close(y, y_ref.to(torch.bfloat16), **BF16_OUTPUT_TOLS)
    torch.testing.assert_close(dx, dx_ref.to(torch.bfloat16), **BF16_OUTPUT_TOLS)
    torch.testing.assert_close(grads["weight"], dw_ref.to(torch.bfloat16), **BF16_OUTPUT_TOLS)

    dx_nvfp4 = dequantize(nvfp4_quantizer(), dy) @ dequantize_columnwise(nvfp4_quantizer(), w)
    dx_mxfp8_from_original = dequantize(mxfp8_quantizer(), dy) @ dequantize_columnwise(
        mxfp8_quantizer(), w
    )
    for dx_wrong in (dx_nvfp4, dx_mxfp8_from_original):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(dx, dx_wrong.to(torch.bfloat16), **BF16_OUTPUT_TOLS)


@requires_nvfp4
def test_4over6_grad_linear_reference():
    """te.Linear with gradient 4over6 matches GEMMs on dequantized 4over6 gradients."""
    x, dy = make_inputs()
    module = make_module("linear")
    _, dx, grads = run_step(module, make_recipe(nvfp4_4over6_grad=True), x, dy)

    w = module.weight.detach()
    for use_4over6 in (True, False):
        dx_ref = dequantize(nvfp4_quantizer(use_4over6=use_4over6), dy) @ dequantize_columnwise(
            nvfp4_quantizer(), w
        )
        dy_col = dequantize_columnwise(nvfp4_quantizer(use_4over6=use_4over6), dy)
        dw_ref = dy_col.t() @ dequantize_columnwise(nvfp4_quantizer(), x)
        if use_4over6:
            torch.testing.assert_close(dx, dx_ref.to(torch.bfloat16), **BF16_OUTPUT_TOLS)
            torch.testing.assert_close(
                grads["weight"], dw_ref.to(torch.bfloat16), **BF16_OUTPUT_TOLS
            )
        else:  # Negative control: standard NVFP4 gradients.
            with pytest.raises(AssertionError):
                torch.testing.assert_close(dx, dx_ref.to(torch.bfloat16), **BF16_OUTPUT_TOLS)


@requires_nvfp4_and_mxfp8
@pytest.mark.parametrize("kind", ["linear", "layernorm_linear", "grouped_linear"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("row_scaled_activation", [False, True])
def test_dgrad_mxfp8_matches_custom_recipe(kind, bias, row_scaled_activation):
    """The built-in recipe dispatches exactly like the equivalent CustomRecipe."""
    x, dy = make_inputs()
    recipe = make_recipe(dgrad_mxfp8=True, row_scaled_activation=row_scaled_activation)
    custom_recipe = dgrad_mxfp8_custom_recipe(row_scaled_activation)
    assert_steps_equal(
        run_step(make_module(kind, bias), recipe, x, dy),
        run_step(make_module(kind, bias), custom_recipe, x, dy),
    )


def _fused_wgrad_linear():
    module = te.Linear(
        IN_FEATURES,
        OUT_FEATURES,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=True,
    )
    module.weight.main_grad = torch.zeros(OUT_FEATURES, IN_FEATURES, device="cuda")
    return module


def _microbatches(module, x):
    return module(x, is_first_microbatch=True) + module(x, is_first_microbatch=False)


MODULE_OPTIONS = {
    "fuse_wgrad_accumulation": (_fused_wgrad_linear, default_forward),
    "delay_wgrad_compute": (
        lambda: te.Linear(
            IN_FEATURES,
            OUT_FEATURES,
            bias=False,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=True,
        ),
        default_forward,
    ),
    "save_original_input": (
        lambda: te.Linear(
            IN_FEATURES,
            OUT_FEATURES,
            bias=False,
            params_dtype=torch.bfloat16,
            save_original_input=True,
        ),
        default_forward,
    ),
    "weight_caching": (lambda: make_module("linear"), _microbatches),
    "single_grouped_weight": (
        lambda: te.GroupedLinear(
            len(M_SPLITS),
            IN_FEATURES,
            OUT_FEATURES,
            bias=False,
            params_dtype=torch.bfloat16,
            single_grouped_weight=True,
        ),
        default_forward,
    ),
    "ops_basic_linear": (
        lambda: te_ops.BasicLinear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
        default_forward,
    ),
    "ops_grouped_linear": (
        lambda: te_ops.GroupedLinear(
            len(M_SPLITS), IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.bfloat16
        ),
        default_forward,
    ),
    "ops_grouped_linear_single_grouped_weight": (
        lambda: te_ops.GroupedLinear(
            len(M_SPLITS),
            IN_FEATURES,
            OUT_FEATURES,
            bias=False,
            dtype=torch.bfloat16,
            single_grouped_weight=True,
        ),
        default_forward,
    ),
}


@requires_nvfp4_and_mxfp8
@pytest.mark.parametrize("option", list(MODULE_OPTIONS))
def test_dgrad_mxfp8_module_options(option):
    build, forward = MODULE_OPTIONS[option]
    x, dy = make_inputs()
    torch.manual_seed(1234)
    step = run_step(build(), make_recipe(dgrad_mxfp8=True), x, dy, forward)
    torch.manual_seed(1234)
    step_ref = run_step(build(), dgrad_mxfp8_custom_recipe(), x, dy, forward)
    assert_steps_equal(step, step_ref)
    assert torch.isfinite(step[1]).all()
    assert all(torch.isfinite(grad).all() for grad in step[2].values())


@requires_nvfp4_and_mxfp8
@pytest.mark.parametrize("kind", ["linear", "layernorm_linear", "grouped_linear"])
def test_dgrad_mxfp8_error(kind):
    """MXFP8 dgrad keeps the NVFP4 forward and wgrad and reduces the dgrad error."""
    x, dy = make_inputs()
    _, dx_ref, _ = run_step(make_module(kind), make_recipe(backward_override="dequantized"), x, dy)
    y_nvfp4, dx_nvfp4, grads_nvfp4 = run_step(make_module(kind), make_recipe(), x, dy)
    y, dx, grads = run_step(make_module(kind), make_recipe(dgrad_mxfp8=True), x, dy)
    torch.testing.assert_close(y, y_nvfp4, rtol=0, atol=0)
    for name, grad in grads.items():
        if not name.startswith("layer_norm"):  # The norm weight gradient depends on dgrad.
            torch.testing.assert_close(grad, grads_nvfp4[name], rtol=0, atol=0)
    assert rel_err(dx, dx_ref) < 0.5 * rel_err(dx_nvfp4, dx_ref)


@requires_nvfp4_and_mxfp8
@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
@pytest.mark.parametrize("bias", [False, True])
def test_dgrad_mxfp8_layernorm_mlp(activation, bias):
    x, dy = make_inputs(out_features=IN_FEATURES)
    module = make_module("layernorm_mlp", bias, activation)
    y_nvfp4, _, grads_nvfp4 = run_step(module, make_recipe(), x, dy)
    module = make_module("layernorm_mlp", bias, activation)
    y, dx, grads = run_step(module, make_recipe(dgrad_mxfp8=True), x, dy)
    torch.testing.assert_close(y, y_nvfp4, rtol=0, atol=0)
    assert torch.isfinite(dx).all()
    for name, grad in grads.items():
        assert torch.isfinite(grad).all()
        if name.startswith("fc2"):  # FC2 gradients do not depend on dgrad.
            torch.testing.assert_close(grad, grads_nvfp4[name], rtol=0, atol=0)


@requires_nvfp4_and_mxfp8
def test_dgrad_mxfp8_rejects_quantized_primary_weights():
    with pytest.raises(ValueError, match="dgrad_mxfp8"):
        with te.quantized_model_init(enabled=True, recipe=make_recipe(dgrad_mxfp8=True)):
            make_module("linear")

    # NVFP4 primary weights provide no MXFP8 columnwise data for dgrad.
    with te.quantized_model_init(enabled=True, recipe=make_recipe()):
        module = make_module("linear")
    x, _ = make_inputs()
    with pytest.raises(RuntimeError, match="dgrad_mxfp8"):
        with te.autocast(enabled=True, recipe=make_recipe(dgrad_mxfp8=True)):
            module(x)


BACKWARD_OPTIONS = {
    "nvfp4": {},
    "row_scaled_activation": {"row_scaled_activation": True},
    "4over6_grad": {"nvfp4_4over6_grad": True},
    "dgrad_mxfp8": {"dgrad_mxfp8": True},
    "dgrad_mxfp8_4over6_grad_row_scaled_activation": {
        "dgrad_mxfp8": True,
        "nvfp4_4over6_grad": True,
        "row_scaled_activation": True,
    },
}


def zero_mask(pattern: str) -> torch.Tensor:
    mask = torch.zeros(NUM_TOKENS, OUT_FEATURES, dtype=torch.bool, device="cuda")
    if pattern == "prompt_rows":  # Loss-masked prompt tokens at the start of each sequence
        for start in range(0, NUM_TOKENS, 128):
            mask[start : start + 96] = True
    elif pattern == "sparse_rows":  # Few tokens carry gradient
        mask[:] = True
        mask[::37] = False
    elif pattern == "blocks":  # All-zero 16-element blocks in both directions
        mask[:, 16:48] = True
        mask[32:64, :] = True
    elif pattern == "all":  # Micro-batch with zero advantages
        mask[:] = True
    else:
        raise ValueError(f"Unknown zero-mask pattern: {pattern}")
    return mask


@requires_nvfp4_and_mxfp8
@pytest.mark.parametrize("options", list(BACKWARD_OPTIONS))
@pytest.mark.parametrize("pattern", ["prompt_rows", "sparse_rows", "blocks", "all"])
@pytest.mark.parametrize("kind", ["linear", "grouped_linear"])
def test_zero_grad_output(options, pattern, kind):
    x, dy = make_inputs()
    mask = zero_mask(pattern)
    dy = dy.masked_fill(mask, 0)
    _, dx, grads = run_step(make_module(kind), make_recipe(**BACKWARD_OPTIONS[options]), x, dy)
    assert torch.isfinite(dx).all()
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    zero_rows = mask.all(dim=1)
    assert (dx[zero_rows] == 0).all()
    if pattern == "all":
        assert all((grad == 0).all() for grad in grads.values())
    else:
        assert (dx[~zero_rows] != 0).any()


@requires_nvfp4_and_mxfp8
@pytest.mark.parametrize("options", list(BACKWARD_OPTIONS))
def test_zero_input_rows(options):
    """Padding tokens produce all-zero input rows."""
    x, dy = make_inputs()
    x[::3] = 0
    y, dx, grads = run_step(make_module("linear"), make_recipe(**BACKWARD_OPTIONS[options]), x, dy)
    assert torch.isfinite(y).all()
    assert torch.isfinite(dx).all()
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    assert (y[::3] == 0).all()
