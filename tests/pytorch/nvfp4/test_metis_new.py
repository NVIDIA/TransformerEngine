# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import sys

sys.path.append("/home/wty/workspace/TransformerEngine")

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# from transformer_engine.pytorch.experimental import quantization_nvfp4
from transformer_engine.pytorch.experimental.quantization_nvfp4 import NVFP4QuantizerRef


from transformer_engine.pytorch.experimental import utils
from transformer_engine.pytorch.module import (
    LinearLowbitContext,
    get_metis_context,
    load_svd_history,
)
import contextlib

# import debugpy
has_init = False
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
    def nvfp4_metis():
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
    def nvfp4_recipe_to_test(with_rht: bool = False, with_2d_quantization: bool = False):
        if with_rht and with_2d_quantization:
            return GetRecipes.nvfp4_rht_and_2d_quantization()
        elif with_rht:
            return GetRecipes.nvfp4_rht_only()
        elif with_2d_quantization:
            return GetRecipes.nvfp4_2d_quantization_only()
        else:
            return GetRecipes.nvfp4_vanilla()


def get_nvfp4_quantizer_factory(with_rht: bool = False, with_2d_quantization: bool = False):
    """
    Create a quantizer factory for NVFP4 reference implementation.

    This factory returns NVFP4QuantizerRef instances based on the role and configuration.
    Used with CustomRecipe to create reference quantizers.

    Args:
        with_rht: Whether to enable random Hadamard transform
        with_2d_quantization: Whether to use 2D quantization (16x16 tiles for weights)

    Returns:
        A factory function that takes a role string and returns a quantizer instance
    """

    def factory(role):
        if role == "linear_input":
            # return quantization_nvfp4.NVFP4QuantizerRef(
            return NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=with_rht,
            )
        elif role == "linear_weight":
            # return quantization_nvfp4.NVFP4QuantizerRef(
            return NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(16, 16) if with_2d_quantization else (1, 16),
                pow_2_scales=False,
                with_rht=False,
            )
        elif role == "linear_output":
            # Output quantization not used
            return None
        elif role == "linear_grad_output":
            # return quantization_nvfp4.NVFP4QuantizerRef(
            return NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=with_rht,
            )
        elif role == "linear_grad_input":
            # Grad input quantization not used
            return None
        else:
            # For any other roles, return None
            return None

    return factory


def reset_rng_states():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def cuda_time_call_backward(fn, x, dx, times=1, **kwargs):
    """Run a callable on CUDA and measure elapsed time in milliseconds.

    Args:
        fn: callable to run.
        *args, **kwargs: forwarded to fn.

    Returns:
        A tuple (result, elapsed_ms). If fn returns multiple values, result
        is whatever fn returned.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Record start, run, record end, synchronize, compute elapsed
    dummy_tensor = torch.rand(4096, 4096, device="cuda")
    dummy_matmul = torch.matmul(dummy_tensor, dummy_tensor)
    start.record()
    for _ in range(times):
        result = fn.backward(x, retain_graph=True, **kwargs)
        x.grad = None
        dx.grad = None
    # Ensure any CUDA kernels launched by fn are recorded before ending
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / times
    return result, elapsed


def cuda_time_call(fn, *args, times=20, **kwargs):
    """Run a callable on CUDA and measure elapsed time in milliseconds.

    Args:
        fn: callable to run.
        *args, **kwargs: forwarded to fn.

    Returns:
        A tuple (result, elapsed_ms). If fn returns multiple values, result
        is whatever fn returned.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Record start, run, record end, synchronize, compute elapsed
    dummy_tensor = torch.rand(4096, 4096, device="cuda")
    dummy_matmul = torch.matmul(dummy_tensor, dummy_tensor)
    start.record()
    for _ in range(times):
        result = fn(*args, **kwargs)
    # Ensure any CUDA kernels launched by fn are recorded before ending
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / times
    return result, elapsed


def check_nvfp4_module_versus_reference(
    module_class,
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    with_rht: bool = False,
    with_2d_quantization: bool = False,
):

    # debugpy.listen(5678)
    # debugpy.wait_for_client()
    """
    Compare native NVFP4 module against reference implementation.

    Args:
        module_class: te.Linear or te.LayerNormLinear
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to use bias
        x_dtype: Input tensor dtype
        num_steps: Number of forward/backward steps to test
    """
    device = "cuda"
    batch_size = 256
    seq_len = 2048

    # Create both modules with identical initialization
    reset_rng_states()

    # Create native module
    print("\nCreate native module")
    if module_class == te.MetisLinear:
        native_module = te.MetisLinear(
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
    print("native_module=", native_module)
    # Create reference module with same weights
    reset_rng_states()

    # Create reference module
    print("Create reference module")
    if module_class == te.MetisLinear:
        ref_module = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            params_dtype=x_dtype,
        )
        baseline_module = te.Linear(
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
            baseline_module.weight.copy_(native_module.weight)
        if bias and hasattr(native_module, "bias") and hasattr(ref_module, "bias"):
            ref_module.bias.copy_(native_module.bias)
            baseline_module.bias.copy_(native_module.bias)

        # Copy layer norm parameters if they exist
        if hasattr(native_module, "layer_norm_weight") and hasattr(ref_module, "layer_norm_weight"):
            ref_module.layer_norm_weight.copy_(native_module.layer_norm_weight)
        if hasattr(native_module, "layer_norm_bias") and hasattr(ref_module, "layer_norm_bias"):
            ref_module.layer_norm_bias.copy_(native_module.layer_norm_bias)

    # Create recipes for native and reference implementations
    nvfp4_recipe = GetRecipes.nvfp4_recipe_to_test(with_rht, with_2d_quantization)
    nvfp4_ref_factory = get_nvfp4_quantizer_factory(with_rht, with_2d_quantization)
    nvfp4_ref_recipe = recipe.CustomRecipe(qfactory=nvfp4_ref_factory)

    # Training loop comparison
    native_outputs = []
    ref_outputs = []
    baseline_outputs = []
    x_shape = (batch_size, seq_len, in_features)
    print(f"x shape= {x_shape}, weight shape={(in_features,out_features)}" + "=" * 50)
    for step in range(num_steps):
        torch.manual_seed(1234 + step)
        torch.cuda.manual_seed(1234 + step)

        x_val = torch.normal(mean=0.0, std=1.0, size=x_shape, dtype=x_dtype, device=device)
        x_native = x_val.clone().detach().requires_grad_(True)
        x_ref = x_val.clone().detach().requires_grad_(True)
        x_base = x_val.clone().detach().requires_grad_(True)

        grad_output_shape = (batch_size, seq_len, out_features)
        grad_output_val = torch.normal(
            mean=0.0, std=1.0, size=grad_output_shape, dtype=x_dtype, device=device
        )
        grad_output_metis = grad_output_val.clone().detach()
        grad_output_nvfp4 = grad_output_val.clone().detach()
        grad_output_baseline = grad_output_val.clone().detach()

        metis_param = {
            "activation_lowrank_niter": 0,
            "backward_lowrank_niter": 0,
            "activation_lowrank_svd": min(in_features // 64, 64),
            "backward_lowrank_svd": min(in_features // 64, 64),
            "forward_svd_rank": min(in_features // 64, 64),
            "enable_activation_svd": True,
            "enable_backward_svd": True,
            "gradacc_broadcast": True,
        }
        # "backward_broadcast_dim":0,
        metis_gradacc_broadcast_ctx = load_svd_history()
        load_history = True
        if step % 2 == 0:
            metis_gradacc_broadcast_ctx = contextlib.nullcontext()
            load_history = False
        y_native = None
        # Native forward/backward
        with metis_gradacc_broadcast_ctx:
            # print("traing context==",LinearLowbitContext())
            with te.autocast(enabled=True, recipe=GetRecipes.nvfp4_vanilla()), get_metis_context(
                **metis_param
            ):
                # enable weight cache by giving is_first_microbatch
                y_native, metis_forward_time = cuda_time_call(
                    native_module, x_native, is_first_microbatch=(step == 0)
                )
        # Backward timing for native
        _, metis_backward_time = cuda_time_call_backward(
            y_native, grad_output_metis, native_module.linear_residual.weight
        )

        # Reference forward/backward
        y_ref = None
        with te.autocast(enabled=True, recipe=nvfp4_recipe):
            y_ref, ref_forward_time = cuda_time_call(ref_module, x_ref)
        # Backward timing for reference
        _, ref_backward_time = cuda_time_call_backward(y_ref, grad_output_nvfp4, ref_module.weight)

        baseline, baseline_time = cuda_time_call(baseline_module, x_base)
        _, baseline_backward_time = cuda_time_call_backward(
            baseline, grad_output_baseline, baseline_module.weight
        )
        print(
            f"=" * 20
            + f" Step {step} time summary begin, load_history == {load_history} "
            + "=" * 20
        )
        print(
            f"baseline forward time (ms): {baseline_time}, backward time (ms):"
            f" {baseline_backward_time}"
        )
        print(
            f"metis forward time (ms): ",
            metis_forward_time,
            ", backward time (ms): ",
            metis_backward_time,
        )
        print(
            f"nvfp4 reference forward time (ms): ",
            ref_forward_time,
            ", backward time (ms): ",
            ref_backward_time,
        )
        print(f"=" * 20 + f" Step {step} time summary end " + "=" * 20)
        # exit()
        # Store results
        # native_outputs.append(
        #     {
        #         "output": y_native.detach().clone(),
        #         "input_grad": (
        #             x_native.grad.detach().clone() if x_native.grad is not None else None
        #         ),
        #         "weight_grad": (
        #             native_module.linear_residual.weight.grad.detach().clone()
        #             if native_module.linear_residual.weight.grad is not None
        #             else None
        #         ),
        #         "bias_grad": (
        #             native_module.linear_residual.bias.grad.detach().clone()
        #             if bias and native_module.linear_residual.bias.grad is not None
        #             else None
        #         ),
        #         # "output_grad": (grad_output_metis.grad.detach().clone() if grad_output_metis is not None else None),
        #     }
        # )

        # ref_outputs.append(
        #     {
        #         "output": y_ref.detach().clone(),
        #         "input_grad": (x_ref.grad.detach().clone() if x_ref.grad is not None else None),
        #         "weight_grad": (
        #             ref_module.weight.grad.detach().clone()
        #             if ref_module.weight.grad is not None
        #             else None
        #         ),
        #         "bias_grad": (
        #             ref_module.bias.grad.detach().clone()
        #             if bias and ref_module.bias.grad is not None
        #             else None
        #         ),
        #         # "output_grad": (grad_output_nvfp4.grad.detach().clone() if grad_output_nvfp4 is not None else None),
        #     }
        # )

        # baseline_outputs.append(
        #     {
        #         "output": baseline.detach().clone(),
        #         "input_grad": (x_base.grad.detach().clone() if x_base.grad is not None else None),
        #         "weight_grad": (
        #             baseline_module.weight.grad.detach().clone()
        #             if baseline_module.weight.grad is not None
        #             else None
        #         ),
        #         "bias_grad": (
        #             baseline_module.bias.grad.detach().clone()
        #             if bias and baseline_module.bias.grad is not None
        #             else None
        #         ),
        #         # "output_grad": (grad_output_baseline.grad.detach().clone() if grad_output_baseline is not None else None)
        #     }
        # )
        x_val = None
        x_native = None
        x_ref = None
        x_base = None

    mse_loss = torch.nn.MSELoss(reduction="mean")
    use_mse = True
    # Compare results across all steps
    # for step in range(num_steps):
    #     native_out = native_outputs[step]
    #     ref_out = ref_outputs[step]
    #     baseline_out = baseline_outputs[step]
    #     print(f"="*20+f" Step {step} result begin"+"="*20)
    #     if use_mse:
    #         print(f"output MSE in metis fp4 pass load_svd_history: {step%2!=0} :", mse_loss(native_out["output"], baseline_out["output"]).item())
    #         print(f"output MSE in nv fp4 pass:", mse_loss(ref_out["output"] , baseline_out["output"]).item())
    #         print(f"input_grad MSE in metis fp4 pass load_svd_history: {step%2!=0}:", mse_loss(native_out["input_grad"] , baseline_out["input_grad"]).item())
    #         print(f"input_grad MSE  in nv fp4 pass:", mse_loss(ref_out["input_grad"] , baseline_out["input_grad"]).item())
    #         print(f"weight_grad MSE in metis fp4 pass  load_svd_history: {step%2!=0}:", mse_loss(native_out["weight_grad"] , baseline_out["weight_grad"]).item())
    #         print(f"weight_grad MSE in nv fp4 pass:", mse_loss(ref_out["weight_grad"] , baseline_out["weight_grad"]).item())
    #     # print(f"output_grad mean error in metis fp4 pass:", torch.mean(torch.abs(native_out["output_grad"] - baseline_out["output_grad"])).item())
    #     # print(f"output_grad mean error in nv fp4 pass:", torch.mean(torch.abs(ref_out["output_grad"] - baseline_out["output_grad"])).item())

    #         if bias and native_out["bias_grad"] is not None and ref_out["bias_grad"] is not None:
    #             print(f"bias_grad MSE in metis fp4 pass:", mse_loss(native_out["bias_grad"] , baseline_out["bias_grad"]).item())
    #             print(f"bias_grad MSE in nv fp4 pass:", mse_loss(ref_out["bias_grad"] , baseline_out["bias_grad"]).item())
    #     else:
    #         print(f"output mean error in metis fp4 pass:", torch.mean(torch.abs(native_out["output"] - baseline_out["output"])).item())
    #         print(f"output mean error in nv fp4 pass:", torch.mean(torch.abs(ref_out["output"] - baseline_out["output"])).item())
    #         print(f"input_grad mean error in metis fp4 pass:", torch.mean(torch.abs(native_out["input_grad"] - baseline_out["input_grad"])).item())
    #         print(f"input_grad mean error in nv fp4 pass:", torch.mean(torch.abs(ref_out["input_grad"] - baseline_out["input_grad"])).item())
    #         print(f"weight_grad mean error in metis fp4 pass:", torch.mean(torch.abs(native_out["weight_grad"] - baseline_out["weight_grad"])).item())
    #         print(f"weight_grad mean error in nv fp4 pass:", torch.mean(torch.abs(ref_out["weight_grad"] - baseline_out["weight_grad"])).item())
    #     # print(f"output_grad mean error in metis fp4 pass:", torch.mean(torch.abs(native_out["output_grad"] - baseline_out["output_grad"])).item())
    #     # print(f"output_grad mean error in nv fp4 pass:", torch.mean(torch.abs(ref_out["output_grad"] - baseline_out["output_grad"])).item())

    #         if bias and native_out["bias_grad"] is not None and ref_out["bias_grad"] is not None:
    #             print(f"bias_grad mean error in metis fp4 pass:", torch.mean(torch.abs(native_out["bias_grad"] - baseline_out["bias_grad"])).item())
    #             print(f"bias_grad mean error in nv fp4 pass:", torch.mean(torch.abs(ref_out["bias_grad"] - baseline_out["bias_grad"])).item())
    #     print(f"="*20+f" Step {step} result end"+"="*20)
    #     # # Compare outputs
    #     # torch.testing.assert_close(
    #     #     native_out["output"],
    #     #     ref_out["output"],
    #     #     atol=1e-6,
    #     #     rtol=1e-6,
    #     #     msg=f"Output mismatch at step {step}",
    #     # )

    #     # # Compare input gradients
    #     # torch.testing.assert_close(
    #     #     native_out["input_grad"],
    #     #     ref_out["input_grad"],
    #     #     atol=1e-6,
    #     #     rtol=1e-6,
    #     #     msg=(
    #     #         f"Input gradient mismatch at step {step}. Native: {native_out['input_grad']}, Ref:"
    #     #         f" {ref_out['input_grad']}"
    #     #     ),
    #     # )

    #     # # Compare weight gradients
    #     # torch.testing.assert_close(
    #     #     native_out["weight_grad"],
    #     #     ref_out["weight_grad"],
    #     #     atol=1e-6,
    #     #     rtol=1e-6,
    #     #     msg=(
    #     #         f"Weight gradient mismatch at step {step}. Native: {native_out['weight_grad']},"
    #     #         f" Ref: {ref_out['weight_grad']}"
    #     #     ),
    #     # )

    #     # # Compare bias gradients
    #     # if bias and native_out["bias_grad"] is not None and ref_out["bias_grad"] is not None:
    #     #     torch.testing.assert_close(
    #     #         native_out["bias_grad"],
    #     #         ref_out["bias_grad"],
    #     #         atol=1e-6,
    #     #         rtol=1e-6,
    #     #         msg=f"Bias gradient mismatch at step {step}",
    #     #     )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "in_features, out_features",
    [
        # (128, 256),
        # (256, 128),
        # (512, 512),
        # (768, 3072),
        (2048, 2048),
    ],
)
# @pytest.mark.parametrize("bias", [True, False], ids=["with_bias", "no_bias"])
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("num_steps", [1, 10], ids=["single_step", "multi_step"])
@pytest.mark.parametrize("with_rht", [False], ids=["no_rht"])
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
        module_class=te.MetisLinear,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        x_dtype=x_dtype,
        num_steps=num_steps,
        with_rht=with_rht,
        with_2d_quantization=with_2d_quantization,
    )
