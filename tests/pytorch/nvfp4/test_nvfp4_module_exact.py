# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.custom_recipes import quantization_ref_nvfp4
from transformer_engine.pytorch.custom_recipes import utils


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


class GetRecipes:
    @staticmethod
    def nvfp4_vanilla():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
        return nvfp4_recipe

    @staticmethod
    def nvfp4_rht_only():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(random_hadamard_transform=True)
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(random_hadamard_transform=False)
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(random_hadamard_transform=True)
        return nvfp4_recipe

    @staticmethod
    def nvfp4_2d_quantization_only():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(fp4_2d_quantization=False)
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(fp4_2d_quantization=True)
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(fp4_2d_quantization=False)
        return nvfp4_recipe

    @staticmethod
    def nvfp4_rht_and_2d_quantization():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(
            random_hadamard_transform=True, fp4_2d_quantization=False
        )
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(
            random_hadamard_transform=False, fp4_2d_quantization=True
        )
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(
            random_hadamard_transform=True, fp4_2d_quantization=False
        )
        return nvfp4_recipe

    @staticmethod
    def nvfp4_row_scaled():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
        # Emit row-scaled (per-token) NVFP4 for the forward activation. In a
        # training Linear this also drives the row-scaled columnwise (transpose)
        # of the activation that the wgrad GEMM consumes in the backward pass.
        nvfp4_recipe.row_scaled_activation = True
        return nvfp4_recipe

    @staticmethod
    def nvfp4_recipe_to_test(
        with_rht: bool = False, with_2d_quantization: bool = False, row_scaled: bool = False
    ):
        if row_scaled:
            return GetRecipes.nvfp4_row_scaled()
        if with_rht and with_2d_quantization:
            return GetRecipes.nvfp4_rht_and_2d_quantization()
        elif with_rht:
            return GetRecipes.nvfp4_rht_only()
        elif with_2d_quantization:
            return GetRecipes.nvfp4_2d_quantization_only()
        else:
            return GetRecipes.nvfp4_vanilla()


def get_nvfp4_quantizer_factory(
    with_rht: bool = False, with_2d_quantization: bool = False, row_scaled: bool = False
):
    """
    Create a quantizer factory for NVFP4 reference implementation.

    This factory returns NVFP4QuantizerRef instances based on the role and configuration.
    Used with CustomRecipe to create reference quantizers.

    Args:
        with_rht: Whether to enable random Hadamard transform
        with_2d_quantization: Whether to use 2D quantization (16x16 tiles for weights)

    Returns:
        A factory function that takes a QuantizerRole (or None for boundary slots)
        and returns a quantizer instance.
    """

    # Boundary slots (output, grad_input) get role=None from Linear.get_quantizer_roles
    # when no consumer is configured. CustomRecipeState rejects None returns from
    # qfactory, so we return a valid quantizer for those slots; it is harmless because
    # the GEMM outputs in the high-precision activation dtype, not in NVFP4.
    def _default_quantizer():
        return quantization_ref_nvfp4.NVFP4QuantizerRef(
            dtype=utils.Fp4Formats.E2M1,
            quant_tile_shape=(1, 16),
            pow_2_scales=False,
            with_rht=with_rht,
        )

    def factory(role):
        if role is None:
            return _default_quantizer()
        if role.tensor_type == "input":
            # Only the forward activation is row-scaled, mirroring the production
            # wiring in quantization.py (mode == "forward" and tensor_type != "weight").
            return quantization_ref_nvfp4.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=with_rht,
                row_scaled_nvfp4=row_scaled,
            )
        if role.tensor_type == "weight":
            return quantization_ref_nvfp4.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(16, 16) if with_2d_quantization else (1, 16),
                pow_2_scales=False,
                with_rht=False,
            )
        if role.tensor_type == "grad_output":
            return _default_quantizer()
        return _default_quantizer()

    return factory


def reset_rng_states():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def assert_close_relative(name: str, native, ref, step: int, rel_tol: float) -> None:
    """Compare two tensors by relative Frobenius-norm error.

    Used for the row-scaled NVFP4 path, where native (cuBLAS + post-scale) and the
    dequantized reference (torch matmul) share the same 4-bit quantization error but
    differ at the kernel/impl level (accumulation order, post-scaling). The relative
    error therefore stays small even though the absolute NVFP4 error is large.
    """
    if native is None or ref is None:
        return
    ref_f = ref.detach().float()
    diff = (native.detach().float() - ref_f).norm().item()
    denom = ref_f.norm().item()
    rel = diff / denom if denom > 0 else diff
    assert rel <= rel_tol, f"{name} relative error {rel:.5f} > {rel_tol} at step {step}"


def check_nvfp4_module_versus_reference(
    module_class,
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    with_rht: bool = False,
    with_2d_quantization: bool = False,
    row_scaled: bool = False,
    rel_tol: float = None,
):
    """
    Compare native NVFP4 module against reference implementation.

    Args:
        module_class: te.Linear or te.LayerNormLinear
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to use bias
        x_dtype: Input tensor dtype
        num_steps: Number of forward/backward steps to test
        row_scaled: Enable row-scaled (per-token) NVFP4 activation quantization,
            exercising the row-scaled columnwise (transpose) path consumed by wgrad.
        rel_tol: If set, compare tensors by relative Frobenius-norm error instead of
            the default bitwise-tight ``assert_close`` (used for the row-scaled path).
    """
    device = "cuda"
    batch_size = 32
    seq_len = 128

    # Create both modules with identical initialization
    reset_rng_states()

    # Create native module
    print("\nCreate native module")
    if module_class == te.Linear:
        native_module = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )
    elif module_class == te.LayerNormLinear:
        native_module = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )
    else:
        raise ValueError(f"Unsupported module class: {module_class}")

    # Create reference module with same weights
    reset_rng_states()

    # Create reference module
    print("Create reference module")
    if module_class == te.Linear:
        ref_module = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )
    elif module_class == te.LayerNormLinear:
        ref_module = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )

    # Sync weights between native and reference modules
    with torch.no_grad():
        # Copy main weight and bias parameters
        if hasattr(native_module, "weight") and hasattr(ref_module, "weight"):
            ref_module.weight.copy_(native_module.weight)
        if bias and hasattr(native_module, "bias") and hasattr(ref_module, "bias"):
            ref_module.bias.copy_(native_module.bias)

        # Copy layer norm parameters if they exist
        if hasattr(native_module, "layer_norm_weight") and hasattr(ref_module, "layer_norm_weight"):
            ref_module.layer_norm_weight.copy_(native_module.layer_norm_weight)
        if hasattr(native_module, "layer_norm_bias") and hasattr(ref_module, "layer_norm_bias"):
            ref_module.layer_norm_bias.copy_(native_module.layer_norm_bias)

    # Create recipes for native and reference implementations
    nvfp4_recipe = GetRecipes.nvfp4_recipe_to_test(with_rht, with_2d_quantization, row_scaled)
    nvfp4_ref_factory = get_nvfp4_quantizer_factory(with_rht, with_2d_quantization, row_scaled)
    nvfp4_ref_recipe = recipe.CustomRecipe(qfactory=nvfp4_ref_factory)

    # Training loop comparison
    native_outputs = []
    ref_outputs = []

    for step in range(num_steps):
        torch.manual_seed(1234 + step)
        torch.cuda.manual_seed(1234 + step)

        x_shape = (batch_size, seq_len, in_features)
        x_val = torch.normal(mean=0.0, std=1.0, size=x_shape, dtype=x_dtype, device=device)
        x_native = x_val.clone().detach().requires_grad_(True)
        x_ref = x_native.clone().detach().requires_grad_(True)

        grad_output_shape = (batch_size, seq_len, out_features)
        grad_output_val = torch.normal(
            mean=0.0, std=1.0, size=grad_output_shape, dtype=x_dtype, device=device
        )
        grad_output = grad_output_val.clone().detach()

        # Native forward/backward
        with te.autocast(enabled=True, recipe=nvfp4_recipe):
            # enable weight cache by giving is_first_microbatch
            y_native = native_module(x_native, is_first_microbatch=(step == 0))
        y_native.backward(grad_output)

        # Reference forward/backward
        with te.autocast(enabled=True, recipe=nvfp4_ref_recipe):
            y_ref = ref_module(x_ref)
        y_ref.backward(grad_output)

        # Store results
        native_outputs.append(
            {
                "output": y_native.detach().clone(),
                "input_grad": (
                    x_native.grad.detach().clone() if x_native.grad is not None else None
                ),
                "weight_grad": (
                    native_module.weight.grad.detach().clone()
                    if native_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    native_module.bias.grad.detach().clone()
                    if bias and native_module.bias.grad is not None
                    else None
                ),
            }
        )

        ref_outputs.append(
            {
                "output": y_ref.detach().clone(),
                "input_grad": (x_ref.grad.detach().clone() if x_ref.grad is not None else None),
                "weight_grad": (
                    ref_module.weight.grad.detach().clone()
                    if ref_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    ref_module.bias.grad.detach().clone()
                    if bias and ref_module.bias.grad is not None
                    else None
                ),
            }
        )

    # Compare results across all steps
    for step in range(num_steps):
        native_out = native_outputs[step]
        ref_out = ref_outputs[step]

        if rel_tol is not None:
            # Row-scaled path: native cuBLAS + post-scale vs dequantized torch
            # reference share the same 4-bit error, so compare by relative norm.
            assert_close_relative("Output", native_out["output"], ref_out["output"], step, rel_tol)
            assert_close_relative(
                "Input gradient", native_out["input_grad"], ref_out["input_grad"], step, rel_tol
            )
            assert_close_relative(
                "Weight gradient", native_out["weight_grad"], ref_out["weight_grad"], step, rel_tol
            )
            if bias:
                assert_close_relative(
                    "Bias gradient", native_out["bias_grad"], ref_out["bias_grad"], step, rel_tol
                )
            continue

        # Compare outputs
        torch.testing.assert_close(
            native_out["output"],
            ref_out["output"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Output mismatch at step {step}",
        )

        # Compare input gradients
        torch.testing.assert_close(
            native_out["input_grad"],
            ref_out["input_grad"],
            atol=1e-6,
            rtol=1e-6,
            msg=(
                f"Input gradient mismatch at step {step}. Native: {native_out['input_grad']}, Ref:"
                f" {ref_out['input_grad']}"
            ),
        )

        # Compare weight gradients
        torch.testing.assert_close(
            native_out["weight_grad"],
            ref_out["weight_grad"],
            atol=1e-6,
            rtol=1e-6,
            msg=(
                f"Weight gradient mismatch at step {step}. Native: {native_out['weight_grad']},"
                f" Ref: {ref_out['weight_grad']}"
            ),
        )

        # Compare bias gradients
        if bias and native_out["bias_grad"] is not None and ref_out["bias_grad"] is not None:
            torch.testing.assert_close(
                native_out["bias_grad"],
                ref_out["bias_grad"],
                atol=1e-6,
                rtol=1e-6,
                msg=f"Bias gradient mismatch at step {step}",
            )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (128, 256),
        (256, 128),
        (512, 512),
        (768, 3072),
        (1024, 4096),
    ],
)
# @pytest.mark.parametrize("bias", [True, False], ids=["with_bias", "no_bias"])
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("num_steps", [1, 3], ids=["single_step", "multi_step"])
@pytest.mark.parametrize("with_rht", [True, False], ids=["with_rht", "no_rht"])
@pytest.mark.parametrize(
    "with_2d_quantization", [True, False], ids=["with_2d_quantization", "no_2d_quantization"]
)
def test_nvfp4_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int,
    with_rht: bool,
    with_2d_quantization: bool,
):
    """Test NVFP4 Linear module against reference implementation."""
    if with_rht and x_dtype != torch.bfloat16:
        pytest.skip("RHT is only supported for bfloat16 input")

    check_nvfp4_module_versus_reference(
        module_class=te.Linear,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        x_dtype=x_dtype,
        num_steps=num_steps,
        with_rht=with_rht,
        with_2d_quantization=with_2d_quantization,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (128, 256),
        (256, 128),
        (512, 512),
        (768, 3072),
    ],
)
@pytest.mark.parametrize("num_steps", [1, 3], ids=["single_step", "multi_step"])
def test_nvfp4_linear_row_scaled_versus_reference(
    in_features: int,
    out_features: int,
    num_steps: int,
):
    """End-to-end row-scaled (per-token) NVFP4 Linear forward + backward.

    Exercises the row-scaled columnwise (transpose) activation quantization that
    this PR unblocks: the forward activation is quantized row-scaled with both
    rowwise (fprop) and columnwise (wgrad) directions, so ``backward()`` drives the
    new transpose path through the wgrad GEMM. Compared against the dequantized
    reference quantizer by relative norm (both share the same 4-bit error).
    """
    check_nvfp4_module_versus_reference(
        module_class=te.Linear,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        x_dtype=torch.bfloat16,
        num_steps=num_steps,
        with_rht=False,
        with_2d_quantization=False,
        row_scaled=True,
        rel_tol=2e-2,
    )


def check_nvfp4_layernorm_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    normalization: str,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    with_rht: bool = False,
    with_2d_quantization: bool = False,
):
    """
    Compare native NVFP4 LayerNormLinear module against reference implementation,
    including ln_out.
    """
    device = "cuda"
    batch_size = 32
    seq_len = 128

    # Create both modules with identical initialization
    reset_rng_states()

    # Native module
    native_module = te.LayerNormLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=x_dtype,
        normalization=normalization,
        return_layernorm_output=True,
    )

    # Reference module
    reset_rng_states()
    ref_module = te.LayerNormLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=x_dtype,
        normalization=normalization,
        return_layernorm_output=True,
    )

    # Sync weights and LN params
    with torch.no_grad():
        if hasattr(native_module, "weight") and hasattr(ref_module, "weight"):
            ref_module.weight.copy_(native_module.weight)
        if bias and hasattr(native_module, "bias") and hasattr(ref_module, "bias"):
            ref_module.bias.copy_(native_module.bias)
        if hasattr(native_module, "layer_norm_weight") and hasattr(ref_module, "layer_norm_weight"):
            if (
                native_module.layer_norm_weight is not None
                and ref_module.layer_norm_weight is not None
            ):
                ref_module.layer_norm_weight.copy_(native_module.layer_norm_weight)
        if hasattr(native_module, "layer_norm_bias") and hasattr(ref_module, "layer_norm_bias"):
            if native_module.layer_norm_bias is not None and ref_module.layer_norm_bias is not None:
                ref_module.layer_norm_bias.copy_(native_module.layer_norm_bias)

    # Create recipes for native and reference implementations
    nvfp4_recipe = GetRecipes.nvfp4_recipe_to_test(with_rht, with_2d_quantization)
    nvfp4_ref_factory = get_nvfp4_quantizer_factory(with_rht, with_2d_quantization)
    nvfp4_ref_recipe = recipe.CustomRecipe(qfactory=nvfp4_ref_factory)

    native_outputs = []
    ref_outputs = []

    for step in range(num_steps):
        torch.manual_seed(1234 + step)
        torch.cuda.manual_seed(1234 + step)

        x_shape = (batch_size, seq_len, in_features)
        x_val = torch.normal(mean=0.0, std=1.0, size=x_shape, dtype=x_dtype, device=device)
        x_native = x_val.clone().detach().requires_grad_(True)
        x_ref = x_native.clone().detach().requires_grad_(True)

        grad_output_shape = (batch_size, seq_len, out_features)
        grad_output_val = torch.normal(
            mean=0.0, std=1.0, size=grad_output_shape, dtype=x_dtype, device=device
        )
        grad_output = grad_output_val.clone().detach()

        # Native forward/backward
        with te.autocast(enabled=True, recipe=nvfp4_recipe):
            y_native, ln_out_native = native_module(x_native, is_first_microbatch=(step == 0))
        y_native.backward(grad_output)

        # Reference forward/backward
        with te.autocast(enabled=True, recipe=nvfp4_ref_recipe):
            y_ref, ln_out_ref = ref_module(x_ref)
        y_ref.backward(grad_output)

        native_outputs.append(
            {
                "output": y_native.detach().clone(),
                "ln_out": ln_out_native.detach().clone(),
                "input_grad": (
                    x_native.grad.detach().clone() if x_native.grad is not None else None
                ),
                "weight_grad": (
                    native_module.weight.grad.detach().clone()
                    if native_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    native_module.bias.grad.detach().clone()
                    if bias and native_module.bias.grad is not None
                    else None
                ),
            }
        )
        ref_outputs.append(
            {
                "output": y_ref.detach().clone(),
                "ln_out": ln_out_ref.detach().clone(),
                "input_grad": (x_ref.grad.detach().clone() if x_ref.grad is not None else None),
                "weight_grad": (
                    ref_module.weight.grad.detach().clone()
                    if ref_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    ref_module.bias.grad.detach().clone()
                    if bias and ref_module.bias.grad is not None
                    else None
                ),
            }
        )

    # Compare results
    for step in range(num_steps):
        n = native_outputs[step]
        r = ref_outputs[step]
        torch.testing.assert_close(
            n["output"],
            r["output"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Output mismatch at step {step}",
        )
        torch.testing.assert_close(
            n["ln_out"],
            r["ln_out"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"LN output mismatch at step {step}",
        )
        torch.testing.assert_close(
            n["input_grad"],
            r["input_grad"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Input gradient mismatch at step {step}",
        )
        torch.testing.assert_close(
            n["weight_grad"],
            r["weight_grad"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Weight gradient mismatch at step {step}",
        )
        if bias and n["bias_grad"] is not None and r["bias_grad"] is not None:
            torch.testing.assert_close(
                n["bias_grad"],
                r["bias_grad"],
                atol=1e-6,
                rtol=1e-6,
                msg=f"Bias gradient mismatch at step {step}",
            )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (128, 256),
        (256, 128),
    ],
)
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("num_steps", [1], ids=["single_step"])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"], ids=["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("with_rht", [True, False], ids=["with_rht", "no_rht"])
@pytest.mark.parametrize(
    "with_2d_quantization", [True, False], ids=["with_2d_quantization", "no_2d_quantization"]
)
def test_nvfp4_layernorm_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    normalization: str,
    x_dtype: torch.dtype,
    num_steps: int,
    with_rht: bool,
    with_2d_quantization: bool,
):
    if with_rht and x_dtype != torch.bfloat16:
        pytest.skip("RHT is only supported for bfloat16 input")

    check_nvfp4_layernorm_linear_versus_reference(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        normalization=normalization,
        x_dtype=x_dtype,
        num_steps=num_steps,
        with_rht=with_rht,
        with_2d_quantization=with_2d_quantization,
    )
