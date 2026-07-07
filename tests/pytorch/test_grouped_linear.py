# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import random
from typing import Dict, List, Optional, Sequence

import pytest
import torch
import torch.nn as nn
from torch.nn import Parameter

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    Float8Quantizer,
    Fp8Padding,
    Fp8Unpadding,
    GroupedLinear,
    Linear,
    MXFP8Quantizer,
    NVFP4Quantizer,
    autocast,
    is_bf16_available,
    quantized_model_init,
)
from transformer_engine.pytorch.cpp_extensions import (
    general_gemm,
    general_grouped_gemm,
    general_grouped_gemm_for_grouped_tensor,
)
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    get_align_size_for_quantization,
)
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
import transformer_engine_torch as tex
from utils import (
    ModelConfig,
    assert_close,
    recipe_id,
    reset_rng_states,
    skip_unsupported_backward_override,
)

# Only run FP8 tests on supported devices.
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
fp8_block_scaling_available, _ = te.is_fp8_block_scaling_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

seed = 1234
reset_rng_states()

NVTE_TEST_NVINSPECT_ENABLED = int(os.environ.get("NVTE_TEST_NVINSPECT_ENABLED", "0"))

if NVTE_TEST_NVINSPECT_ENABLED:
    import nvdlfw_inspect.api as debug_api

    debug_api.initialize(
        os.environ["NVTE_TEST_NVINSPECT_CONFIG_FILE"],
        feature_dirs=os.environ["NVTE_TEST_NVINSPECT_FEATURE_DIRS"],
    )


model_configs = {
    "126m": ModelConfig(1, 2048, 12, 64, num_layers=12),
}


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


def nvfp4_row_scaled():
    nvfp4_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="high_precision",
    )
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
    return nvfp4_recipe


def nvfp4_4over6():
    nvfp4_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        nvfp4_4over6="all",
    )
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(fp4_2d_quantization=True)
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
    return nvfp4_recipe


def check_rht_usage(recipe: recipe.Recipe) -> bool:
    if recipe.nvfp4():
        if (
            recipe.fp4_quant_fwd_inp.random_hadamard_transform
            or recipe.fp4_quant_fwd_weight.random_hadamard_transform
            or recipe.fp4_quant_bwd_grad.random_hadamard_transform
        ):
            return True
    return False


def get_nvfp4_inp_supported_dtypes(recipe: recipe.Recipe, dtype: torch.dtype) -> bool:
    supported_input_dtypes = []
    if recipe.nvfp4():
        supported_input_dtypes.append(torch.bfloat16)
    if not check_rht_usage(recipe):
        supported_input_dtypes.append(torch.float32)
    return supported_input_dtypes


def dtype_tols(dtype: torch.dtype) -> Dict[str, float]:
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    raise ValueError(f"Unsupported dtype ({dtype})")


param_types = [torch.float32, torch.float16]
if is_bf16_available():
    param_types.append(torch.bfloat16)

batch_sizes = [1, 2]
all_boolean = [True, False]

fp8_recipes = []
if mxfp8_available:
    fp8_recipes.append(recipe.MXFP8BlockScaling())
if fp8_block_scaling_available:
    fp8_recipes.append(recipe.Float8BlockScaling())
if fp8_available:
    fp8_recipes.append(recipe.Float8CurrentScaling())
    fp8_recipes.append(recipe.DelayedScaling())
if nvfp4_available:
    fp8_recipes.append(nvfp4_rht_and_2d_quantization())
    fp8_recipes.append(nvfp4_4over6())
    fp8_recipes.append(nvfp4_row_scaled())

use_cutlass_grouped_gemm = [False]
if torch.cuda.get_device_capability() == (9, 0):
    use_cutlass_grouped_gemm.append(True)


class TorchGroupedLinearWithPadding(nn.Module):

    def __init__(
        self, num_gemms, in_features, out_features, bias, params_dtype, parallel_mode, fp8
    ) -> None:
        super().__init__()

        self.padding = Fp8Padding(num_gemms)
        self.linear_fn = GroupedLinear(
            num_gemms,
            in_features,
            out_features,
            bias=bias,
            params_dtype=params_dtype,
            parallel_mode=parallel_mode,
            device="cuda",
        )
        self.unpadding = Fp8Unpadding(num_gemms)

        self.fp8 = fp8

    def forward(self, inp: torch.Tensor, m_splits: List[int]) -> torch.Tensor:
        if self.fp8:
            orig_m_splits = m_splits
            inp, m_splits = self.padding(inp, m_splits)

        out = self.linear_fn(inp, m_splits)

        if self.fp8:
            out = self.unpadding(out, orig_m_splits)

        return out


def _test_grouped_linear_accuracy(
    block,
    num_gemms,
    bs,
    dtype,
    config,
    recipe,
    fp8,
    fuse_wgrad_accumulation,
    delay_wgrad_compute=False,
):
    reset_rng_states()
    if fp8:
        FP8GlobalStateManager.reset()

    inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    if num_gemms > 1:
        split_size = 1
        if fp8:
            split_size = get_align_size_for_quantization(recipe)
        m = config.max_seqlen_q // split_size
        dist = torch.sort(torch.randint(0, m, (num_gemms - 2,))).values.tolist()
        dist.append(dist[-1])  # Manually add a zero
        m_splits = torch.tensor(dist + [m]) - torch.tensor([0] + dist)
        m_splits = m_splits * split_size
        assert m_splits.sum() == config.max_seqlen_q and len(m_splits) == num_gemms
    else:
        m_splits = torch.tensor([config.max_seqlen_q])

    with autocast(enabled=fp8, recipe=recipe):
        if isinstance(block, GroupedLinear):
            m_splits = m_splits * bs
            out = block(inp_hidden_states, m_splits.tolist())
        else:
            out = torch.cat(
                [
                    block[i](inp)
                    for i, inp in enumerate(torch.split(inp_hidden_states, m_splits.tolist()))
                ]
            )
    loss = out.sum()
    loss.backward()
    if delay_wgrad_compute:
        if isinstance(block, GroupedLinear):
            block.backward_dw()
        else:
            for i in range(num_gemms):
                block[i].backward_dw()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            if getattr(p, "main_grad", None) is not None:
                outputs.append(p.main_grad)
                assert p.grad is None  # grad should be None if fuse_wgrad_accumulation is True
            else:
                outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types, ids=str)
@pytest.mark.parametrize("num_gemms", [3, 6])
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("recipe", fp8_recipes + [None], ids=recipe_id)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
@pytest.mark.parametrize("fuse_wgrad_accumulation", all_boolean)
@pytest.mark.parametrize("bias", all_boolean)
@pytest.mark.parametrize("delay_wgrad_compute", all_boolean)
def test_grouped_linear_accuracy(
    dtype,
    num_gemms,
    bs,
    model,
    recipe,
    fp8_model_params,
    fuse_wgrad_accumulation,
    bias,
    delay_wgrad_compute,
    parallel_mode=None,
    use_cutlass=False,
):
    fp8 = recipe is not None
    if fp8 and fp8_model_params and NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("FP8 parameters are not supported in debug mode.")
    if NVTE_TEST_NVINSPECT_ENABLED and delay_wgrad_compute:
        pytest.skip("Delayed wgrad compute is not supported in debug mode.")
    skip_unsupported_backward_override(
        "grouped_linear", recipe, getattr(recipe, "backward_override", None)
    )

    config = model_configs[model]
    if config.max_seqlen_q % 16 != 0 and fp8:
        pytest.skip("FP8 requires sequence length to be divisible by 16.")

    if recipe is not None and recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe {recipe.__class__.__name__}"
            )

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        grouped_linear = GroupedLinear(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=bias,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            device="cuda",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            delay_wgrad_compute=delay_wgrad_compute,
            save_original_input=False,
        ).eval()
        sequential_linear = torch.nn.ModuleList(
            [
                Linear(
                    config.hidden_size,
                    4 * config.hidden_size,
                    bias=bias,
                    params_dtype=dtype,
                    parallel_mode=parallel_mode,
                    device="cuda",
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                ).eval()
                for _ in range(num_gemms)
            ]
        )

    # Share params
    with torch.no_grad():
        for i in range(num_gemms):
            sequential_linear[i].weight = Parameter(getattr(grouped_linear, f"weight{i}").clone())
            if bias:
                sequential_linear[i].bias = Parameter(getattr(grouped_linear, f"bias{i}").clone())
            if fuse_wgrad_accumulation:
                weight_i = getattr(grouped_linear, f"weight{i}")
                weight_i.main_grad = torch.rand_like(weight_i, dtype=torch.float32)
                sequential_linear[i].weight.main_grad = weight_i.main_grad.clone()

    outputs_ref = _test_grouped_linear_accuracy(
        sequential_linear,
        num_gemms,
        bs,
        dtype,
        config,
        recipe,
        fp8,
        fuse_wgrad_accumulation,
        delay_wgrad_compute,
    )
    outputs = _test_grouped_linear_accuracy(
        grouped_linear,
        num_gemms,
        bs,
        dtype,
        config,
        recipe,
        fp8,
        fuse_wgrad_accumulation,
        delay_wgrad_compute,
    )

    for o, o_ref in zip(outputs, outputs_ref):
        if use_cutlass:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        else:
            # cuBLAS implementation should be bit-wise match
            torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (9, 0),
    reason="Only enable CUTLASS grouped gemm on Hopper",
)
@pytest.mark.parametrize("dtype", param_types, ids=str)
@pytest.mark.parametrize("num_gemms", [3, 6])
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fuse_wgrad_accumulation", all_boolean)
@pytest.mark.parametrize("delay_wgrad_compute", all_boolean)
def test_grouped_linear_accuracy_cutlass(
    dtype,
    num_gemms,
    bs,
    model,
    fuse_wgrad_accumulation,
    delay_wgrad_compute,
    monkeypatch,
):
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
    test_grouped_linear_accuracy(
        dtype,
        num_gemms,
        bs,
        model,
        None,
        False,
        fuse_wgrad_accumulation,
        False,
        delay_wgrad_compute,
        None,
        use_cutlass=True,
    )


@pytest.mark.parametrize("dtype", param_types, ids=str)
@pytest.mark.parametrize("num_gemms", [3])
@pytest.mark.parametrize("bs", [1])
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("recipe", fp8_recipes + [None], ids=recipe_id)
@pytest.mark.parametrize("fp8_model_params", [False])
@pytest.mark.parametrize("fuse_wgrad_accumulation", [True])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("delay_wgrad_compute", [True])
def test_grouped_linear_accuracy_save_original_input(
    dtype,
    num_gemms,
    bs,
    model,
    recipe,
    fp8_model_params,
    fuse_wgrad_accumulation,
    bias,
    delay_wgrad_compute,
    parallel_mode=None,
):
    fp8 = recipe is not None
    if fp8 and fp8_model_params and NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("FP8 parameters are not supported in debug mode.")
    if fp8 and recipe.delayed():
        pytest.skip("DelayedScaling recipe is not supported with save_original_input")
    if NVTE_TEST_NVINSPECT_ENABLED and delay_wgrad_compute:
        pytest.skip("Delayed wgrad compute is not supported in debug mode.")
    skip_unsupported_backward_override(
        "grouped_linear", recipe, getattr(recipe, "backward_override", None)
    )

    config = model_configs[model]
    if config.max_seqlen_q % 16 != 0 and fp8:
        pytest.skip("FP8 requires sequence length to be divisible by 16.")

    if recipe is not None and recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe {recipe.__class__.__name__}"
            )

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        grouped_linear = GroupedLinear(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=bias,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            device="cuda",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            delay_wgrad_compute=delay_wgrad_compute,
            save_original_input=True,
        ).eval()
        sequential_linear = torch.nn.ModuleList(
            [
                Linear(
                    config.hidden_size,
                    4 * config.hidden_size,
                    bias=bias,
                    params_dtype=dtype,
                    parallel_mode=parallel_mode,
                    device="cuda",
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                ).eval()
                for _ in range(num_gemms)
            ]
        )

    # Share params
    with torch.no_grad():
        for i in range(num_gemms):
            sequential_linear[i].weight = Parameter(getattr(grouped_linear, f"weight{i}").clone())
            if bias:
                sequential_linear[i].bias = Parameter(getattr(grouped_linear, f"bias{i}").clone())
            if fuse_wgrad_accumulation:
                weight_i = getattr(grouped_linear, f"weight{i}")
                weight_i.main_grad = torch.rand_like(weight_i, dtype=torch.float32)
                sequential_linear[i].weight.main_grad = weight_i.main_grad.clone()

    outputs_ref = _test_grouped_linear_accuracy(
        sequential_linear,
        num_gemms,
        bs,
        dtype,
        config,
        recipe,
        fp8,
        fuse_wgrad_accumulation,
        delay_wgrad_compute,
    )
    outputs = _test_grouped_linear_accuracy(
        grouped_linear,
        num_gemms,
        bs,
        dtype,
        config,
        recipe,
        fp8,
        fuse_wgrad_accumulation,
        delay_wgrad_compute,
    )

    # Should be bit-wise match
    for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.parametrize("recipe", fp8_recipes + [None], ids=recipe_id)
def test_grouped_linear_accuracy_single_gemm(recipe):
    """Split the tests to save CI time"""
    test_grouped_linear_accuracy(
        dtype=torch.float32,
        num_gemms=1,
        bs=2,
        model="126m",
        recipe=recipe,
        fp8_model_params=True,
        fuse_wgrad_accumulation=True,
        bias=True,
        delay_wgrad_compute=False,
    )


def _test_padding_grouped_linear_accuracy(block, num_gemms, bs, dtype, config, recipe, fp8=False):

    def _pad_tensor_for_fp8(hidden_states, tokens_per_expert):
        align_size = get_align_size_for_quantization(recipe)
        padded_tokens_per_expert = [
            (num_tokens + align_size - 1) // align_size * align_size
            for num_tokens in tokens_per_expert
        ]
        hidden_states = torch.split(hidden_states, tokens_per_expert)
        padded_hidden_states = []
        for hidden_state, actual_num_tokens, padded_num_tokens in zip(
            hidden_states, tokens_per_expert, padded_tokens_per_expert
        ):
            padded_hidden_states.append(hidden_state)
            if padded_num_tokens > actual_num_tokens:
                pad_tensor = torch.zeros(
                    padded_num_tokens - actual_num_tokens,
                    hidden_state.shape[1],
                    dtype=hidden_state.dtype,
                    device=hidden_state.device,
                )
                padded_hidden_states.append(pad_tensor)
        padded_hidden_states = torch.cat(padded_hidden_states, dim=0)
        return padded_hidden_states, padded_tokens_per_expert

    def _unpad_tensor_for_fp8(padded_hidden_states, actual_tokens_per_expert, tokens_per_expert):
        inputmats = torch.split(
            padded_hidden_states.view(-1, padded_hidden_states.shape[-1]), tokens_per_expert
        )
        hidden_states = torch.cat(
            [
                grad_output_mat[: actual_tokens_per_expert[i]]
                for i, grad_output_mat in enumerate(inputmats)
            ],
            dim=0,
        )

        return hidden_states

    def _generate_random_numbers(n, total_sum):
        if n <= 0:
            return []

        # reset seed
        random.seed(seed)

        breaks = sorted(random.sample(range(1, total_sum), n - 1))
        random_numbers = (
            [breaks[0]]
            + [breaks[i] - breaks[i - 1] for i in range(1, n - 1)]
            + [total_sum - breaks[-1]]
        )

        return random_numbers

    reset_rng_states()
    if fp8:
        FP8GlobalStateManager.reset()

    inp_hidden_states = torch.randn(
        (config.max_seqlen_q * bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    m_splits = _generate_random_numbers(num_gemms, config.max_seqlen_q * bs)

    with autocast(enabled=fp8, recipe=recipe):
        if isinstance(block, TorchGroupedLinearWithPadding):
            out = block(inp_hidden_states, m_splits)
        else:
            if fp8:
                padded_inp_hidden_states, padding_m_splits = _pad_tensor_for_fp8(
                    inp_hidden_states, m_splits
                )
                padded_inp_hidden_states = block(padded_inp_hidden_states, padding_m_splits)
                out = _unpad_tensor_for_fp8(padded_inp_hidden_states, m_splits, padding_m_splits)
            else:
                out = block(inp_hidden_states, m_splits)

    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("num_gemms", [3, 6])
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fp8", [True])
@pytest.mark.parametrize("recipe", fp8_recipes, ids=recipe_id)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
def test_padding_grouped_linear_accuracy(
    dtype,
    num_gemms,
    bs,
    model,
    fp8,
    recipe,
    fp8_model_params,
    parallel_mode=None,
):
    if fp8_model_params and NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("FP8 parameters are not supported in debug mode.")
    skip_unsupported_backward_override(
        "grouped_linear", recipe, getattr(recipe, "backward_override", None)
    )

    config = model_configs[model]
    if config.max_seqlen_q % 16 != 0 and fp8:
        pytest.skip("FP8 requires sequence length to be divisible by 16.")

    if recipe is not None and recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe {recipe.__class__.__name__}"
            )

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        grouped_linear = TorchGroupedLinearWithPadding(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            fp8=fp8,
        ).eval()

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        ref_grouped_linear = GroupedLinear(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            device="cuda",
            save_original_input=False,
        ).eval()

    # Share params
    with torch.no_grad():
        inner_grouped_linear = grouped_linear.linear_fn
        for i in range(num_gemms):
            setattr(
                ref_grouped_linear,
                f"weight{i}",
                Parameter(getattr(inner_grouped_linear, f"weight{i}").clone()),
            )

    outputs = _test_padding_grouped_linear_accuracy(
        grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
    )
    outputs_ref = _test_padding_grouped_linear_accuracy(
        ref_grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
    )

    # Should be bit-wise match
    for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("num_gemms", [3])
@pytest.mark.parametrize("bs", [1])
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fp8", [True])
@pytest.mark.parametrize("recipe", fp8_recipes, ids=recipe_id)
@pytest.mark.parametrize("fp8_model_params", [False])
def test_padding_grouped_linear_accuracy_save_original_input(
    dtype,
    num_gemms,
    bs,
    model,
    fp8,
    recipe,
    fp8_model_params,
    parallel_mode=None,
):
    if fp8_model_params and NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("FP8 parameters are not supported in debug mode.")
    if fp8 and recipe.delayed():
        pytest.skip("DelayedScaling recipe is not supported with save_original_input")
    skip_unsupported_backward_override(
        "grouped_linear", recipe, getattr(recipe, "backward_override", None)
    )

    config = model_configs[model]
    if config.max_seqlen_q % 16 != 0 and fp8:
        pytest.skip("FP8 requires sequence length to be divisible by 16.")

    if recipe is not None and recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe {recipe.__class__.__name__}"
            )

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        grouped_linear = TorchGroupedLinearWithPadding(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            fp8=fp8,
        ).eval()

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        ref_grouped_linear = GroupedLinear(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            device="cuda",
            save_original_input=True,
        ).eval()

    # Share params
    with torch.no_grad():
        inner_grouped_linear = grouped_linear.linear_fn
        for i in range(num_gemms):
            setattr(
                ref_grouped_linear,
                f"weight{i}",
                Parameter(getattr(inner_grouped_linear, f"weight{i}").clone()),
            )

    outputs = _test_padding_grouped_linear_accuracy(
        grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
    )
    outputs_ref = _test_padding_grouped_linear_accuracy(
        ref_grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
    )

    # Should be bit-wise match
    for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 127, 128, 512),
        (8, 15, 128, 512),
        (8, 1027, 128, 512),
        (16, 10027, 128, 512),
    ],
)
@pytest.mark.parametrize("dtype", param_types, ids=str)
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("use_cutlass", use_cutlass_grouped_gemm)
def test_grouped_gemm(shape, dtype, layout, accumulate, use_cutlass, monkeypatch):
    torch.manual_seed(0)
    z, m, k, n = shape

    dist = torch.sort(torch.randint(0, m, (z - 1,))).values.tolist()
    m_splits = torch.tensor(dist + [m]) - torch.tensor([0] + dist)
    assert m_splits.sum() == m and len(m_splits) == z
    m_splits = m_splits.tolist()

    if layout == "TN":
        A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
        B = list(torch.split(torch.randn(m, k, dtype=dtype, device="cuda"), m_splits))  # input
        out = [torch.randn(m, n, dtype=dtype, device="cuda")]  # output
        out_ref = [o.clone() for o in torch.split(out[0], m_splits)]
        grad = False
        single_output = True
    elif layout == "NN":
        A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
        B = list(
            torch.split(torch.randn(m, n, dtype=dtype, device="cuda"), m_splits)
        )  # grad_output
        out = [torch.randn(m, k, dtype=dtype, device="cuda")]  # dgrad
        out_ref = [o.clone() for o in torch.split(out[0], m_splits)]
        grad = True
        single_output = True
    else:  # layout == "NT"
        A = list(torch.split(torch.randn(m, k, dtype=dtype, device="cuda"), m_splits))  # input
        B = list(
            torch.split(torch.randn(m, n, dtype=dtype, device="cuda"), m_splits)
        )  # grad_output
        out = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # wgrad
        out_ref = [o.clone() for o in out]
        grad = True
        single_output = False

    if use_cutlass:
        monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")

    for i in range(z):
        general_gemm(
            A[i],
            B[i],
            dtype,
            grad=grad,
            accumulate=accumulate,
            layout=layout,
            out=out_ref[i],
        )
    if single_output:
        out_ref = [torch.cat(out_ref)]

    general_grouped_gemm(
        A,
        B,
        out,
        [None] * z,
        dtype,
        m_splits=m_splits,
        grad=grad,
        accumulate=accumulate,
        layout=layout,
        single_output=single_output,
    )

    for o, o_ref in zip(out, out_ref):
        if not use_cutlass:
            # cublas implementation should be bit-wise match
            torch.testing.assert_close(o, o_ref, rtol=0, atol=0)
        else:
            torch.testing.assert_close(o, o_ref, rtol=1.5e-2, atol=1.5e-2)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (9, 0),
    reason="Only enable CUTLASS grouped gemm on Hopper",
)
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
def test_grouped_gemm_cutlass_empty_groups(layout, monkeypatch):
    dtype = torch.bfloat16
    z, k, n = 1, 2048, 1536
    m_splits = [0] * z

    if layout == "TN":
        A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
        B = [torch.empty(0, k, dtype=dtype, device="cuda") for _ in range(z)]  # input
        out = [torch.empty(0, n, dtype=dtype, device="cuda")]  # output
        grad = False
        single_output = True
    elif layout == "NN":
        A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
        B = [torch.empty(0, n, dtype=dtype, device="cuda") for _ in range(z)]  # grad_output
        out = [torch.empty(0, k, dtype=dtype, device="cuda")]  # dgrad
        grad = True
        single_output = True
    else:  # layout == "NT"
        A = [torch.empty(0, k, dtype=dtype, device="cuda") for _ in range(z)]  # input
        B = [torch.empty(0, n, dtype=dtype, device="cuda") for _ in range(z)]  # grad_output
        out = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # wgrad
        grad = True
        single_output = False

    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
    general_grouped_gemm(
        A,
        B,
        out,
        [None] * z,
        dtype,
        m_splits=m_splits,
        grad=grad,
        layout=layout,
        single_output=single_output,
    )
    torch.cuda.synchronize()

    for tensor in out:
        torch.testing.assert_close(tensor, torch.zeros_like(tensor), rtol=0, atol=0)


# =============================================================================
# NVFP4 single-launch CUTLASS grouped GEMM (Blackwell / SM100). Opt-in via
# NVTE_NVFP4_CUTLASS_GROUPED_GEMM inside the graph-safe grouped-tensor path
# (general_grouped_gemm_for_grouped_tensor). Backend parity and graph-safety are
# covered by the test_nvfp4_grouped_tensor_cutlass_* tests at the end of this
# file; the constants and _diff helper below are shared with them.
# =============================================================================
_NVFP4_CUTLASS_ENV = "NVTE_NVFP4_CUTLASS_GROUPED_GEMM"
nvfp4_cutlass_grouped_available = nvfp4_available and torch.cuda.get_device_capability()[0] == 10


def _diff(ref: torch.Tensor, test: torch.Tensor):
    """(max_abs, global_inf_norm_rel, ref_inf). The global rel (max_abs / ||ref||)
    is robust to the near-zero output elements that make per-element rel explode."""
    ref = ref.float()
    test = test.float()
    max_abs = (ref - test).abs().max().item()
    ref_inf = ref.abs().max().item()
    return max_abs, max_abs / max(ref_inf, 1e-6), ref_inf


def _pack_grouped_tensor(grouped_tensor: GroupedTensor, tensors: List[torch.Tensor]) -> None:
    data = grouped_tensor.rowwise_data
    if data is None:
        data = grouped_tensor.columnwise_data
    if data is None:
        raise ValueError("GroupedTensor has no data buffers to pack.")
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        data[offset : offset + numel].copy_(tensor.reshape(-1))
        offset += numel


def _make_grouped_tensor_from_splits(
    m_sizes: List[int],
    last_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GroupedTensor:
    first_dims = torch.tensor(m_sizes, device=device, dtype=torch.int64)
    return GroupedTensor.make_grouped_tensor(
        num_tensors=len(m_sizes),
        first_dims=first_dims,
        last_dims=None,
        logical_first_dim=sum(m_sizes),
        logical_last_dim=last_dim,
        quantizer=None,
        device=device,
        dtype=dtype,
    )


def _make_grouped_tensor_uniform(
    num_tensors: int,
    first_dim: int,
    last_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GroupedTensor:
    return GroupedTensor.make_grouped_tensor(
        num_tensors=num_tensors,
        first_dims=None,
        last_dims=None,
        logical_first_dim=num_tensors * first_dim,
        logical_last_dim=last_dim,
        quantizer=None,
        device=device,
        dtype=dtype,
    )


def _apply_grouped_bias_ref(
    base_outs: List[torch.Tensor],
    bias: Optional[List[torch.Tensor]],
    bias_scale: Optional[torch.Tensor],
    m_sizes: List[int],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Reference: add (optionally per-row scaled) bias to each group's output, cast to ``dtype``."""
    if bias is None:
        return list(base_outs)
    if bias_scale is None:
        return [(o.float() + b.float()).to(dtype) for o, b in zip(base_outs, bias)]
    out = []
    offset = 0
    for i, ms in enumerate(m_sizes):
        s = bias_scale[offset : offset + ms].unsqueeze(-1)
        out.append((base_outs[i].float() + bias[i].float() * s).to(dtype))
        offset += ms
    return out


@pytest.mark.parametrize(
    "z, m, n, k",
    [
        (4, 256, 256, 256),
        (4, 512, 256, 512),
        (4, 512, 512, 256),
        (8, 512, 256, 512),
    ],
)
@pytest.mark.parametrize("case", ["no_discrete", "discrete_in", "discrete_out"])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("use_bias_scale", [False, True])
def test_grouped_gemm_grouped_tensor(z, m, n, k, case, layout, accumulate, use_bias_scale) -> None:
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Grouped GEMM requires Hopper (SM90) or newer.")
    if torch.cuda.get_device_capability() < (10, 0):
        if tex.get_cublasLt_version() < 130400:
            pytest.skip("Grouped GEMM on Hopper requires cuBLAS 13.4+.")
    if tex.get_cublasLt_version() < 130300:
        pytest.skip("Grouped GEMM requires cuBLAS 13.3+.")
    if not is_bf16_available():
        pytest.skip("bfloat16 is required for grouped GEMM test.")

    torch.manual_seed(0)

    dtype = torch.bfloat16

    split_points = torch.randperm(m - 1)[: z - 1] + 1
    split_points = torch.sort(split_points).values.tolist()
    m_sizes = [split_points[0]]
    m_sizes += [b - a for a, b in zip(split_points[:-1], split_points[1:])]
    m_sizes.append(m - split_points[-1])
    assert sum(m_sizes) == m and len(m_sizes) == z

    if layout == "NT":
        A = [torch.randn(ms, k, dtype=dtype, device="cuda") for ms in m_sizes]  # input
        B = [torch.randn(ms, n, dtype=dtype, device="cuda") for ms in m_sizes]  # grad_output
        out = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # wgrad
        out_ref = [torch.matmul(B[i].transpose(0, 1).float(), A[i].float()) for i in range(z)]
    else:
        A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
        B = [
            torch.randn(ms, k if layout == "TN" else n, dtype=dtype, device="cuda")
            for ms in m_sizes
        ]  # TN --> input, NN --> grad_output
        out = [
            torch.randn(ms, n if layout == "TN" else k, dtype=dtype, device="cuda")
            for ms in m_sizes
        ]  # TN --> output, NN --> dgrad
        if layout == "NN":
            out_ref = [torch.matmul(B[i].float(), A[i].float()) for i in range(z)]
        else:  # layout == "TN"
            out_ref = [torch.matmul(B[i].float(), A[i].transpose(0, 1).float()) for i in range(z)]

    if accumulate:
        out_ref = [out[i].float() + o for i, o in enumerate(out_ref)]

    # Bias is applied after GEMM (broadcasted along rows)
    # Match kernel behavior: GEMM output is already in output dtype when bias is added.
    out_ref_no_bias = [o.to(dtype) for o in out_ref]
    if layout == "TN":
        bias_last_dim = n
    else:  # layout == "NT" or "NN"
        bias_last_dim = k
    bias = (
        [torch.randn(1, bias_last_dim, dtype=dtype, device="cuda") for _ in range(z)]
        if case != "discrete_out"
        else None
    )
    bias_scale = None
    if use_bias_scale and bias is not None and layout != "NT":
        bias_scale = torch.randn(m, device="cuda", dtype=torch.float32)
    # Bias add in grouped kernel accumulates in FP32 for BF16/FP16.
    out_ref = _apply_grouped_bias_ref(out_ref_no_bias, bias, bias_scale, m_sizes, dtype)
    # Create grouped tensors based on case
    device = A[0].device
    grouped_A = A
    grouped_out = out
    grouped_out_bias = [o.clone() for o in out]
    grouped_out_no_bias = [o.clone() for o in out]
    grouped_bias = None
    if layout == "TN":
        grouped_A = (
            _make_grouped_tensor_uniform(z, n, k, device, dtype) if case != "discrete_in" else A
        )  # weight
        grouped_B = _make_grouped_tensor_from_splits(m_sizes, k, device, dtype)  # input
        if case != "discrete_out":
            grouped_out = _make_grouped_tensor_from_splits(m_sizes, n, device, dtype)  # output
            grouped_out_bias = _make_grouped_tensor_from_splits(m_sizes, n, device, dtype)
            grouped_out_no_bias = _make_grouped_tensor_from_splits(m_sizes, n, device, dtype)
    elif layout == "NN":
        grouped_A = (
            _make_grouped_tensor_uniform(z, n, k, device, dtype) if case != "discrete_in" else A
        )  # weight
        grouped_B = _make_grouped_tensor_from_splits(m_sizes, n, device, dtype)  # grad_output
        if case != "discrete_out":
            grouped_out = _make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
            grouped_out_bias = _make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
            grouped_out_no_bias = _make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
    else:  # layout == "NT"
        grouped_A = (
            _make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
            if case != "discrete_in"
            else A
        )  # input
        grouped_B = _make_grouped_tensor_from_splits(m_sizes, n, device, dtype)  # grad_output
        if case != "discrete_out":
            grouped_out = _make_grouped_tensor_uniform(z, n, k, device, dtype)  # wgrad
            grouped_out_bias = _make_grouped_tensor_uniform(z, n, k, device, dtype)
            grouped_out_no_bias = _make_grouped_tensor_uniform(z, n, k, device, dtype)
    _pack_grouped_tensor(grouped_B, B)
    if case != "discrete_out":
        _pack_grouped_tensor(grouped_out, out)
        _pack_grouped_tensor(grouped_out_bias, out)
        _pack_grouped_tensor(grouped_out_no_bias, out)
    if case != "discrete_in":
        _pack_grouped_tensor(grouped_A, A)

    if bias is not None:
        grouped_bias = _make_grouped_tensor_uniform(z, 1, bias_last_dim, device, dtype)
        _pack_grouped_tensor(grouped_bias, bias)

    general_grouped_gemm_for_grouped_tensor(
        grouped_A,
        grouped_B,
        grouped_out_no_bias,
        layout=layout,
        accumulate=accumulate,
        bias=None,
    )
    general_grouped_gemm_for_grouped_tensor(
        grouped_A,
        grouped_B,
        grouped_out_bias,
        layout=layout,
        accumulate=accumulate,
        bias=grouped_bias,
        bias_scale=bias_scale,
    )
    out_grouped_no_bias = (
        grouped_out_no_bias
        if isinstance(grouped_out_no_bias, list)
        else grouped_out_no_bias.split_into_quantized_tensors()
    )
    out_grouped_bias = (
        grouped_out_bias
        if isinstance(grouped_out_bias, list)
        else grouped_out_bias.split_into_quantized_tensors()
    )

    out_grouped_manual_bias = _apply_grouped_bias_ref(
        out_grouped_no_bias, bias, bias_scale, m_sizes, dtype
    )
    tols = dtype_tols(dtype)
    for o, o_ref in zip(out_grouped_no_bias, out_ref_no_bias):
        torch.testing.assert_close(o, o_ref, **tols)
    if bias is not None:
        for o, o_ref in zip(out_grouped_bias, out_grouped_manual_bias):
            torch.testing.assert_close(o, o_ref, **tols)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("quant_type", ["bf16", "mxfp8"])
def test_grouped_gemm_grouped_tensor_zero_work(layout, accumulate, quant_type) -> None:
    """Grouped GEMM with all-zero split sizes (zero total work).

    For wgrad (NT layout) the output should be zero when not accumulating,
    or unchanged when accumulating with beta=1.
    """
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("Grouped GEMM requires Blackwell (SM100) or newer.")
    if not is_bf16_available():
        pytest.skip("bfloat16 is required for grouped GEMM test.")
    if quant_type == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)

    z = 4
    k, n = 256, 256
    dtype = torch.bfloat16
    device = torch.device("cuda")
    use_mxfp8 = quant_type == "mxfp8"

    transa = layout[0] == "T"
    transb = layout[1] == "T"
    zero_first_dims = torch.zeros(z, dtype=torch.int64, device=device)

    def _make_zero_tokens_grouped_tensor(logical_last_dim, is_a):
        """Create a GroupedTensor with non-zero logical_shape but zero first_dims."""
        buf = torch.randn(0, logical_last_dim, dtype=dtype, device=device)
        if use_mxfp8:
            if is_a:
                rowwise, columnwise = transa, not transa
            else:
                rowwise, columnwise = not transb, transb
            quantizer = MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=rowwise,
                columnwise=columnwise,
            )
            quantizer.optimize_for_gemm = True
            return tex.group_quantize(buf, quantizer, z, zero_first_dims)
        return GroupedTensor.make_grouped_tensor(
            num_tensors=z,
            first_dims=zero_first_dims,
            last_dims=None,
            logical_first_dim=k,
            logical_last_dim=logical_last_dim,
            quantizer=None,
            device=device,
            dtype=dtype,
        )

    if layout in ("TN", "NN"):
        weight_tensors = [torch.randn(n, k, dtype=dtype, device=device) for _ in range(z)]
        if use_mxfp8:
            grouped_A = _make_grouped_tensor_quantized_mxfp8(
                weight_tensors,
                rowwise=transa,
                columnwise=not transa,
                device=device,
            )
        else:
            grouped_A = _make_grouped_tensor_uniform(z, n, k, device, dtype)
            _pack_grouped_tensor(grouped_A, weight_tensors)
    else:  # NT
        grouped_A = _make_zero_tokens_grouped_tensor(k, is_a=True)

    b_last_dim = k if layout == "TN" else n
    grouped_B = _make_zero_tokens_grouped_tensor(b_last_dim, is_a=False)

    if layout == "NT":
        out = [torch.randn(n, k, dtype=dtype, device=device) for _ in range(z)]
        grouped_out = _make_grouped_tensor_uniform(z, n, k, device, dtype)
        _pack_grouped_tensor(grouped_out, out)
    else:
        out = [torch.zeros(0, dtype=dtype, device=device) for _ in range(z)]
        out_last_dim = n if layout == "TN" else k
        grouped_out = GroupedTensor.make_grouped_tensor(
            num_tensors=z,
            first_dims=zero_first_dims,
            last_dims=None,
            logical_first_dim=k,
            logical_last_dim=out_last_dim,
            quantizer=None,
            device=device,
            dtype=dtype,
        )

    out_before = [o.clone() for o in out]

    general_grouped_gemm_for_grouped_tensor(
        grouped_A,
        grouped_B,
        grouped_out,
        layout=layout,
        accumulate=accumulate,
    )

    out_result = (
        grouped_out if isinstance(grouped_out, list) else grouped_out.split_into_quantized_tensors()
    )
    for i in range(z):
        if out_result[i].numel() == 0:
            continue
        if accumulate:
            torch.testing.assert_close(out_result[i], out_before[i])
        else:
            torch.testing.assert_close(out_result[i], torch.zeros_like(out_result[i]))


def _make_grouped_tensor_quantized_mxfp8(
    tensors: List[torch.Tensor],
    *,
    rowwise: bool,
    columnwise: bool,
    device: torch.device,
    is_weight: bool = False,
) -> GroupedTensor:
    """Create a quantized MXFP8 GroupedTensor from a list of per-expert tensors.

    For weights (uniform per-expert shape), we generally won't keep it swizzled since we
    might need for future dequantize operations. Swizzling is done internally within
    general_grouped_gemm_for_grouped_tensor call.

    For non-weight tensors (inputs / grad_outputs), we still pass
    ``first_dims`` and keep ``optimize_for_gemm=True``; so the kernel must emit the
    already-swizzled layout up front.
    """
    if not tensors:
        raise ValueError("Expected non-empty tensor list for grouped quantization.")
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    quantizer.optimize_for_gemm = not is_weight
    grouped_input = torch.cat(tensors, dim=0)
    if is_weight:
        first_dims = None
    else:
        first_dims = torch.tensor([t.shape[0] for t in tensors], dtype=torch.int64, device=device)
    return tex.group_quantize(grouped_input, quantizer, len(tensors), first_dims)


def _per_tensor_quantize_mxfp8(
    tensors: List[torch.Tensor],
    *,
    rowwise: bool,
    columnwise: bool,
) -> List:
    """Quantize each tensor individually with MXFP8.
    Used to build reference discrete inputs for grouped GEMM.
    """
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    return [quantizer(t) for t in tensors]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 128, 512),
        (8, 1024, 128, 512),
        (16, 4096, 128, 512),
        (2, 256, 2880, 2880),
    ],
)
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("case", ["no_discrete", "discrete_in", "discrete_out"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_gemm_grouped_tensor_mxfp8(
    shape, accumulate, layout: str, case: str, dtype: torch.dtype
) -> None:
    if tex.get_cublasLt_version() < 130300:
        pytest.skip("Grouped GEMM requires cuBLAS 13.3+.")
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("Grouped GEMM requires Blackwell (SM100) or newer.")
    if dtype == torch.bfloat16 and not is_bf16_available():
        pytest.skip("bfloat16 is required for grouped GEMM test.")

    torch.manual_seed(0)
    z, m, k, n = shape
    m_sizes = [m // z] * z

    if layout == "TN":
        A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
        B = [torch.randn(ms, k, dtype=dtype, device="cuda") for ms in m_sizes]  # input
        out = [torch.randn(ms, n, dtype=dtype, device="cuda") for ms in m_sizes]  # output
        grad = False
    elif layout == "NN":
        A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
        B = [torch.randn(ms, n, dtype=dtype, device="cuda") for ms in m_sizes]  # grad_output
        out = [torch.randn(ms, k, dtype=dtype, device="cuda") for ms in m_sizes]  # dgrad
        grad = True
    else:  # layout == "NT"
        A = [torch.randn(ms, k, dtype=dtype, device="cuda") for ms in m_sizes]  # input
        B = [torch.randn(ms, n, dtype=dtype, device="cuda") for ms in m_sizes]  # grad_output
        out = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # wgrad
        grad = True

    out_ref = [o.clone() for o in out]

    transa = layout[0] == "T"
    transb = layout[1] == "T"
    a_is_weight = all(t.shape == A[0].shape for t in A)
    a_rowwise, a_columnwise = transa, not transa
    b_rowwise, b_columnwise = not transb, transb
    grouped_A = _make_grouped_tensor_quantized_mxfp8(
        A,
        rowwise=a_rowwise,
        columnwise=a_columnwise,
        device="cuda",
        is_weight=a_is_weight,
    )
    grouped_B = _make_grouped_tensor_quantized_mxfp8(
        B, rowwise=b_rowwise, columnwise=b_columnwise, device="cuda"
    )
    A_fp8 = _per_tensor_quantize_mxfp8(A, rowwise=a_rowwise, columnwise=a_columnwise)
    B_fp8 = _per_tensor_quantize_mxfp8(B, rowwise=b_rowwise, columnwise=b_columnwise)

    general_grouped_gemm(
        A_fp8,
        B_fp8,
        out_ref,
        [None] * z,
        dtype,
        m_splits=m_sizes,
        grad=grad,
        accumulate=accumulate,
        layout=layout,
        single_output=False,
    )

    device = A[0].device

    grouped_out = None
    if case != "discrete_out":
        if layout == "TN":
            grouped_out = _make_grouped_tensor_from_splits(m_sizes, n, device, dtype)
        elif layout == "NN":
            grouped_out = _make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
        else:  # layout == "NT"
            grouped_out = _make_grouped_tensor_uniform(z, n, k, device, dtype)
        _pack_grouped_tensor(grouped_out, out)

    grouped_out_input = out if case == "discrete_out" else grouped_out
    grouped_A_input = A_fp8 if case == "discrete_in" else grouped_A
    general_grouped_gemm_for_grouped_tensor(
        grouped_A_input,
        grouped_B,
        grouped_out_input,
        layout=layout,
        accumulate=accumulate,
    )

    out_grouped = out if case == "discrete_out" else grouped_out.split_into_quantized_tensors()
    tols = dict(rtol=0.125, atol=0.0675)  # mxfp8 tolerance

    for o, o_ref in zip(out_grouped, out_ref):
        torch.testing.assert_close(o, o_ref, **tols)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 128, 512),
        (8, 1024, 128, 512),
        (16, 4096, 128, 512),
    ],
)
@pytest.mark.parametrize("accumulate", [False, True])
def test_fp8_grouped_gemm(shape, accumulate):
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    z, m, k, n = shape
    m_splits = [m // z] * z

    dtype = torch.bfloat16
    A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
    B = torch.split(torch.randn(m, k, dtype=dtype, device="cuda"), m_splits)  # input
    out = torch.split(torch.randn(m, n, dtype=dtype, device="cuda"), m_splits)  # output
    out_ref = [o.clone() for o in out]

    # fp8 should be robust enough to this fake scale
    scale = 1 + torch.rand(1, dtype=torch.float32, device="cuda").squeeze()
    amax = torch.zeros(1, 1, dtype=torch.float32, device="cuda")

    a_quantizers = [
        Float8Quantizer(
            scale.clone(),
            amax.clone(),
            tex.DType.kFloat8E4M3,
        )
        for _ in range(z)
    ]
    b_quantizers = [
        Float8Quantizer(
            scale.clone(),
            amax.clone(),
            tex.DType.kFloat8E4M3,
        )
        for _ in range(z)
    ]

    A_fp8 = []
    B_fp8 = []

    for i in range(z):
        A_fp8.append(a_quantizers[i](A[i]))
        B_fp8.append(b_quantizers[i](B[i]))

    # baseline
    for i in range(z):
        general_gemm(
            A_fp8[i],
            B_fp8[i],
            dtype,
            out=out_ref[i],
            accumulate=accumulate,
        )
    general_grouped_gemm(
        A_fp8,
        B_fp8,
        out,
        [None] * z,
        dtype,
        m_splits=m_splits,
        accumulate=accumulate,
    )

    # should be bit-wise match
    for o, o_ref in zip(out, out_ref):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


_FUSED_GROUPED_GEMM_ENV = "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"
_ALL_BOOLEAN = all_boolean
_fp8_available, _reason_for_no_fp8 = fp8_available, reason_for_no_fp8
_mxfp8_available, _reason_for_no_mxfp8 = mxfp8_available, reason_for_no_mxfp8
_nvfp4_available, _reason_for_no_nvfp4 = nvfp4_available, reason_for_no_nvfp4


@pytest.fixture(autouse=True)
def _reset_fp8_state(monkeypatch):
    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "0")
    yield
    FP8GlobalStateManager.reset()
    monkeypatch.delenv(_FUSED_GROUPED_GEMM_ENV, raising=False)


def _clone_outputs(outputs):
    return [None if out is None else out.detach().clone() for out in outputs]


def _run_grouped_linear_path(
    *,
    enable_grouped_tensor_path: bool,
    fp8_recipe,
    bias: bool,
    fp8_model_params: bool,
    delay_wgrad_compute: bool,
    x_base: torch.Tensor,
    dy: torch.Tensor,
    weights,
    biases,
    m_splits,
    monkeypatch,
):
    FP8GlobalStateManager.reset()
    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "1" if enable_grouped_tensor_path else "0")

    dtype = x_base.dtype
    num_gemms = len(m_splits)
    in_features = weights[0].size(1)
    out_features = weights[0].size(0)
    use_fp8 = fp8_recipe is not None

    x = x_base.detach().clone().requires_grad_(True)
    with quantized_model_init(enabled=fp8_model_params, recipe=fp8_recipe):
        grouped_linear = GroupedLinear(
            num_gemms,
            in_features,
            out_features,
            bias=bias,
            params_dtype=dtype,
            device="cuda",
            delay_wgrad_compute=delay_wgrad_compute,
        )
    with torch.no_grad():
        for i in range(num_gemms):
            getattr(grouped_linear, f"weight{i}").copy_(weights[i])
            if bias:
                getattr(grouped_linear, f"bias{i}").copy_(biases[i])

    # The fused path is the graph-safe path and accepts a CUDA tensor for split metadata.
    # The legacy path still expects Python split sections in several places.
    m_splits_arg = (
        torch.tensor(m_splits, dtype=torch.int64, device="cuda")
        if enable_grouped_tensor_path
        else m_splits
    )
    with autocast(enabled=use_fp8, recipe=fp8_recipe):
        y = grouped_linear(x, m_splits_arg)
    y.backward(dy)
    if delay_wgrad_compute:
        grouped_linear.backward_dw()

    outputs = [y, x.grad]
    for i in range(num_gemms):
        outputs.append(getattr(grouped_linear, f"weight{i}").grad)
        if bias:
            outputs.append(getattr(grouped_linear, f"bias{i}").grad)
    return _clone_outputs(outputs)


@pytest.mark.parametrize(
    "fp8_recipe",
    [
        None,
        pytest.param(
            recipe.Float8CurrentScaling(),
            marks=pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8),
        ),
        pytest.param(
            recipe.MXFP8BlockScaling(),
            marks=pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8),
        ),
        pytest.param(
            recipe.NVFP4BlockScaling(disable_stochastic_rounding=True),
            marks=pytest.mark.skipif(not _nvfp4_available, reason=_reason_for_no_nvfp4),
        ),
    ],
    ids=["bf16", "fp8_current_scaling", "mxfp8", "nvfp4"],
)
@pytest.mark.parametrize("bias", _ALL_BOOLEAN)
@pytest.mark.parametrize("fp8_model_params", _ALL_BOOLEAN)
@pytest.mark.parametrize("delay_wgrad_compute", _ALL_BOOLEAN)
def test_grouped_linear_grouped_tensor_path_matches_legacy(
    fp8_recipe, bias, fp8_model_params, delay_wgrad_compute, monkeypatch
):
    use_fp8 = fp8_recipe is not None
    device_capability = torch.cuda.get_device_capability()
    if not (9, 0) <= device_capability <= (11, 0):
        pytest.skip(
            "GroupedTensor grouped GEMM path requires Hopper (SM90) or Blackwell (SM10x and SM110)."
        )
    # MXFP8/NVFP4 grouped quantization kernels require Blackwell, but FP8 per-tensor
    # current scaling also runs on the Hopper grouped GEMM path.
    is_current_scaling = use_fp8 and fp8_recipe.float8_current_scaling()
    if use_fp8 and not is_current_scaling and device_capability < (10, 0):
        pytest.skip(
            "Quantized GroupedTensor grouped GEMM path (MXFP8/NVFP4) requires Blackwell (SM100+)."
        )
    cublaslt_version = tex.get_cublasLt_version()
    if device_capability < (10, 0) and cublaslt_version < 130400:
        pytest.skip("Grouped GEMM on Hopper requires cuBLAS 13.4+.")
    if cublaslt_version < 130300:
        pytest.skip("Grouped GEMM requires cuBLAS 13.3+.")

    if fp8_model_params and not use_fp8:
        pytest.skip("fp8_model_params requires FP8")

    dtype = torch.bfloat16
    num_gemms = 3
    in_features = 128
    out_features = 128
    m_splits = [128, 256, 384]
    total_tokens = sum(m_splits)

    torch.manual_seed(1234)
    x_base = (0.1 * torch.randn(total_tokens, in_features, device="cuda")).to(dtype)
    dy = (0.1 * torch.randn(total_tokens, out_features, device="cuda")).to(dtype)
    weights = [
        (0.1 * torch.randn(out_features, in_features, device="cuda")).to(dtype)
        for _ in range(num_gemms)
    ]
    biases = None
    if bias:
        biases = [
            (0.1 * torch.randn(out_features, device="cuda")).to(dtype) for _ in range(num_gemms)
        ]

    outputs_legacy = _run_grouped_linear_path(
        enable_grouped_tensor_path=False,
        fp8_recipe=fp8_recipe,
        bias=bias,
        fp8_model_params=fp8_model_params,
        delay_wgrad_compute=delay_wgrad_compute,
        x_base=x_base,
        dy=dy,
        weights=weights,
        biases=biases,
        m_splits=m_splits,
        monkeypatch=monkeypatch,
    )
    outputs_grouped_tensor = _run_grouped_linear_path(
        enable_grouped_tensor_path=True,
        fp8_recipe=fp8_recipe,
        bias=bias,
        fp8_model_params=fp8_model_params,
        delay_wgrad_compute=delay_wgrad_compute,
        x_base=x_base,
        dy=dy,
        weights=weights,
        biases=biases,
        m_splits=m_splits,
        monkeypatch=monkeypatch,
    )

    tols = dict(rtol=1e-2, atol=5e-3)
    if use_fp8:
        tols = dict(rtol=0.05, atol=0.05)
    for grouped_tensor_out, legacy_out in zip(outputs_grouped_tensor, outputs_legacy):
        assert grouped_tensor_out is not None
        assert legacy_out is not None
        torch.testing.assert_close(grouped_tensor_out.float(), legacy_out.float(), **tols)


def test_grouped_linear_grouped_tensor_path_single_grouped_bias_delay_wgrad(monkeypatch):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("GroupedTensor grouped GEMM path requires SM100+")

    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "1")

    dtype = torch.bfloat16
    num_gemms = 3
    in_features = 64
    out_features = 64
    total_tokens = 64 + 96 + 128
    m_splits = torch.tensor([64, 96, 128], dtype=torch.int64, device="cuda")
    x = torch.randn(total_tokens, in_features, dtype=dtype, device="cuda").requires_grad_()
    dy = torch.randn(x.size(0), out_features, dtype=dtype, device="cuda")

    grouped_linear = GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=True,
        params_dtype=dtype,
        device="cuda",
        delay_wgrad_compute=True,
        single_grouped_bias=True,
    )

    y = grouped_linear(x, m_splits)
    y.backward(dy)
    grouped_linear.backward_dw()


@pytest.mark.skipif(not _nvfp4_available, reason=_reason_for_no_nvfp4)
def test_grouped_linear_grouped_tensor_path_skips_non_rht_nvfp4(monkeypatch):
    """Non-RHT NVFP4 falls back to the legacy path; check it stays numerically correct.

    Graph-safe grouped quantization currently requires RHT, so requesting NVFP4 with
    ``disable_rht=True`` while the fused grouped-tensor path is enabled falls back to the
    legacy path internally. We verify the output and gradients against a reference built from
    per-GEMM ``te.Linear`` modules that share the same weights and use the same NVFP4 recipe;
    the grouped GEMM should match the loop of single GEMMs.
    """
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("NVFP4 GroupedTensor grouped GEMM path requires SM100+")

    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "1")
    FP8GlobalStateManager.reset()

    dtype = torch.bfloat16
    num_gemms = 3
    in_features = 128
    out_features = 128
    m_splits = [128, 256, 384]
    total_tokens = sum(m_splits)

    torch.manual_seed(1234)
    x_base = (0.1 * torch.randn(total_tokens, in_features, device="cuda")).to(dtype)
    dy = (0.1 * torch.randn(total_tokens, out_features, device="cuda")).to(dtype)
    weights = [
        (0.1 * torch.randn(out_features, in_features, device="cuda")).to(dtype)
        for _ in range(num_gemms)
    ]

    fp8_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
    )

    # Grouped path: fused path enabled, but non-RHT NVFP4 falls back to legacy internally.
    grouped_linear = GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    )
    with torch.no_grad():
        for i in range(num_gemms):
            getattr(grouped_linear, f"weight{i}").copy_(weights[i])

    x = x_base.detach().clone().requires_grad_(True)
    with autocast(enabled=True, recipe=fp8_recipe):
        y = grouped_linear(x, m_splits)
    y.backward(dy)

    # Reference: one te.Linear per GEMM sharing the same weights and NVFP4 recipe.
    ref_linears = torch.nn.ModuleList(
        [
            Linear(in_features, out_features, bias=False, params_dtype=dtype, device="cuda")
            for _ in range(num_gemms)
        ]
    )
    with torch.no_grad():
        for i in range(num_gemms):
            ref_linears[i].weight.copy_(weights[i])

    x_ref = x_base.detach().clone().requires_grad_(True)
    with autocast(enabled=True, recipe=fp8_recipe):
        y_ref = torch.cat(
            [ref_linears[i](x_i) for i, x_i in enumerate(torch.split(x_ref, m_splits))]
        )
    y_ref.backward(dy)

    # cuBLAS grouped GEMM should match the loop of single GEMMs bit-for-bit.
    tols = dict(rtol=0, atol=0)
    torch.testing.assert_close(y.float(), y_ref.float(), **tols)
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), **tols)
    for i in range(num_gemms):
        torch.testing.assert_close(
            getattr(grouped_linear, f"weight{i}").grad.float(),
            ref_linears[i].weight.grad.float(),
            **tols,
        )


@pytest.mark.parametrize(
    "fp8_recipe",
    [
        None,
        pytest.param(
            recipe.Float8CurrentScaling(),
            marks=pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8),
        ),
        pytest.param(
            recipe.MXFP8BlockScaling(),
            marks=pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8),
        ),
        pytest.param(
            recipe.NVFP4BlockScaling(disable_stochastic_rounding=True),
            marks=pytest.mark.skipif(not _nvfp4_available, reason=_reason_for_no_nvfp4),
        ),
    ],
    ids=["bf16", "fp8_current_scaling", "mxfp8", "nvfp4"],
)
@pytest.mark.parametrize("bias", _ALL_BOOLEAN)
def test_grouped_linear_fused_path_cuda_graph_safe(fp8_recipe, bias, monkeypatch):
    """Fused GroupedTensor GEMM path should be CUDA graph capturable."""
    use_fp8 = fp8_recipe is not None
    device_capability = torch.cuda.get_device_capability()
    if not (9, 0) <= device_capability <= (11, 0):
        pytest.skip(
            "GroupedTensor grouped GEMM path requires Hopper (SM90) or Blackwell (SM10x and SM110)."
        )
    # MXFP8/NVFP4 grouped quantization kernels require Blackwell, but FP8 per-tensor
    # current scaling also runs on the Hopper grouped GEMM path.
    is_current_scaling = use_fp8 and fp8_recipe.float8_current_scaling()
    if use_fp8 and not is_current_scaling and device_capability < (10, 0):
        pytest.skip(
            "Quantized GroupedTensor grouped GEMM path (MXFP8/NVFP4) requires Blackwell (SM100+)."
        )
    cublaslt_version = tex.get_cublasLt_version()
    if device_capability < (10, 0) and cublaslt_version < 130400:
        pytest.skip("Grouped GEMM on Hopper requires cuBLAS 13.4+.")
    if cublaslt_version < 130300:
        pytest.skip("Grouped GEMM requires cuBLAS 13.3+.")

    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "1")
    FP8GlobalStateManager.reset()

    dtype = torch.bfloat16
    device = "cuda"
    num_gemms = 3
    in_features = 128
    out_features = 128
    split_sizes = [128, 256, 384]
    total_tokens = sum(split_sizes)
    static_m_splits = torch.tensor(split_sizes, dtype=torch.int64, device=device)

    grouped_linear = GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=bias,
        params_dtype=dtype,
        device=device,
    )

    static_x = torch.randn(total_tokens, in_features, dtype=dtype, device=device)
    static_x.requires_grad_(True)
    static_dy = torch.randn(total_tokens, out_features, dtype=dtype, device=device)
    static_out_buf = torch.empty(total_tokens, out_features, dtype=dtype, device=device)

    def _zero_grads():
        if static_x.grad is not None:
            static_x.grad.zero_()
        for param in grouped_linear.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.zero_()

    def _clone_param_grads():
        return [param.grad.detach().clone() for param in grouped_linear.parameters()]

    def _train_step(x, dy, out_buf, *, use_graphed):
        with autocast(enabled=use_fp8, recipe=fp8_recipe):
            out = (
                graphed_grouped_linear(x, static_m_splits)
                if use_graphed
                else grouped_linear(x, static_m_splits)
            )
        out.backward(dy)
        out_buf.copy_(out)
        return out_buf

    graphed_grouped_linear = te.make_graphed_callables(
        grouped_linear,
        (static_x, static_m_splits),
        num_warmup_iters=3,
        enabled=use_fp8,
        recipe=fp8_recipe,
    )

    fresh_x = torch.randn_like(static_x)
    fresh_dy = torch.randn_like(static_dy)
    with torch.no_grad():
        static_x.copy_(fresh_x)
        static_dy.copy_(fresh_dy)

    _zero_grads()
    graph_out = (
        _train_step(
            static_x,
            static_dy,
            static_out_buf,
            use_graphed=True,
        )
        .detach()
        .clone()
    )
    torch.cuda.synchronize()
    graph_dx = static_x.grad.detach().clone()
    graph_param_grads = _clone_param_grads()

    _zero_grads()
    expected_x = fresh_x.detach().clone().requires_grad_(True)
    expected_dy = fresh_dy.detach().clone()
    with autocast(enabled=use_fp8, recipe=fp8_recipe):
        expected_out = grouped_linear(expected_x, static_m_splits)
    expected_out.backward(expected_dy)

    tols = dict(rtol=1e-2, atol=5e-3)
    if use_fp8:
        tols = dict(rtol=0.05, atol=0.05)
    torch.testing.assert_close(graph_out.float(), expected_out.float(), **tols)
    torch.testing.assert_close(graph_dx.float(), expected_x.grad.float(), **tols)
    for graph_grad, param in zip(graph_param_grads, grouped_linear.parameters()):
        assert param.grad is not None
        torch.testing.assert_close(graph_grad.float(), param.grad.float(), **tols)


@pytest.mark.parametrize("swizzle_type", ["mxfp8_rowwise", "mxfp8_columnwise", "nvfp4"])
def test_swizzle_scales_and_pack_ptrs_for_discrete_weights(
    swizzle_type: str,
    num_tensors: int = 4,
    shape: Sequence[int] = (160, 96),
):
    """Helper function for preparing discrete weights for cuDNN group GEMM kernel"""

    # Skip unsupported configurations
    if not mxfp8_available and swizzle_type in ("mxfp8_rowwise", "mxfp8_columnwise"):
        pytest.skip(reason_for_no_mxfp8)
    if not nvfp4_available and swizzle_type == "nvfp4":
        pytest.skip(reason_for_no_nvfp4)

    # Construct quantizer
    quantizer = None
    if swizzle_type in ("mxfp8_rowwise", "mxfp8_columnwise"):
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=swizzle_type == "mxfp8_rowwise",
            columnwise=swizzle_type == "mxfp8_columnwise",
        )
    elif swizzle_type == "nvfp4":
        quantizer = NVFP4Quantizer(
            columnwise=False,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )

    # Per-expert tensors: unquantized, quantized with compact scales,
    # quantized with swizzled scales
    device = torch.device("cuda")
    unquantized_tensors = [
        torch.randn(shape, dtype=torch.bfloat16, device=device) for _ in range(num_tensors)
    ]
    quantizer.optimize_for_gemm = False
    tensors_with_compact_scales = [quantizer(t) for t in unquantized_tensors]
    quantizer.optimize_for_gemm = True
    tensors_with_swizzled_scales = [quantizer(t) for t in unquantized_tensors]

    # Extract data and scale buffers
    if swizzle_type in ("mxfp8_rowwise", "nvfp4"):
        data_tensors = [qx._rowwise_data for qx in tensors_with_compact_scales]
        scale_tensors = [qx._rowwise_scale_inv for qx in tensors_with_compact_scales]
        ref_scale_tensors = [qx._rowwise_scale_inv for qx in tensors_with_swizzled_scales]
    elif swizzle_type == "mxfp8_columnwise":
        data_tensors = [qx._columnwise_data for qx in tensors_with_compact_scales]
        scale_tensors = [qx._columnwise_scale_inv for qx in tensors_with_compact_scales]
        ref_scale_tensors = [qx._columnwise_scale_inv for qx in tensors_with_swizzled_scales]
    else:
        raise ValueError("Unrecogized swizzle type")

    # Call the helper function
    data_ptrs, scale_ptrs, swizzled_scales_buffer = (
        tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
            data_tensors,
            scale_tensors,
            swizzle_type,
            device,
        )
    )

    # Check data pointer values
    expected_data_ptrs = torch.tensor(
        [t.data_ptr() for t in data_tensors],
        dtype=torch.int64,
        device="cpu",
    )
    assert_close(data_ptrs, expected_data_ptrs)

    # Check scale pointer values
    scale_bytes = scale_tensors[0].numel() * scale_tensors[0].element_size()
    expected_scale_ptrs = torch.tensor(
        [swizzled_scales_buffer.data_ptr() + i * scale_bytes for i in range(num_tensors)],
        dtype=torch.int64,
        device="cpu",
    )
    assert_close(scale_ptrs, expected_scale_ptrs)

    # Check swizzled scale values
    swizzled_scales_buffer = swizzled_scales_buffer.view(torch.uint8)
    expected_swizzled_scales_buffer = (
        torch.cat(ref_scale_tensors).view(torch.uint8).view_as(swizzled_scales_buffer)
    )
    assert_close(
        swizzled_scales_buffer,
        expected_swizzled_scales_buffer,
    )

    # Poison the padded compact scales
    if swizzle_type == "mxfp8_rowwise":
        unpadded_scale_shape = (shape[0], shape[1] // 32)
    elif swizzle_type == "mxfp8_columnwise":
        unpadded_scale_shape = (shape[0] // 32, shape[1])
    elif swizzle_type == "nvfp4":
        unpadded_scale_shape = (shape[0], shape[1] // 16)
    for scale in scale_tensors:
        scale[unpadded_scale_shape[0] :, :].view(torch.uint8).fill_(-1)
        scale[:, unpadded_scale_shape[1] :].view(torch.uint8).fill_(-1)

    # Check that swizzling removes poisoned pad scales
    _, _, swizzled_scales_buffer = (
        tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights(
            data_tensors,
            scale_tensors,
            swizzle_type,
            device,
        )
    )
    assert_close(
        swizzled_scales_buffer,
        expected_swizzled_scales_buffer,
    )


# =============================================================================
# NVFP4 single-launch CUTLASS grouped GEMM through the graph-safe grouped-tensor
# path. Both backends run through general_grouped_gemm_for_grouped_tensor;
# NVTE_NVFP4_CUTLASS_GROUPED_GEMM only flips which single-launch kernel it
# dispatches to (cuBLAS baseline vs CUTLASS). NVTE_GROUPED_LINEAR_USE_FUSED_
# GROUPED_GEMM routes GroupedLinear onto that path. SM100-only.
# =============================================================================
def _nvfp4_grouped_m_splits(num_gemms, dist):
    """128-aligned per-group token counts (per-tensor NVFP4 CUTLASS is %128)."""
    if dist == "balanced":
        return [256] * num_gemms
    return [128 * ((i % 3) + 1) for i in range(num_gemms)]  # unequal, 128-aligned


@pytest.mark.skipif(
    not nvfp4_cutlass_grouped_available,
    reason="NVFP4 CUTLASS grouped GEMM requires Blackwell (SM100) + NVFP4",
)
@pytest.mark.parametrize("num_gemms", [3, 8])
@pytest.mark.parametrize("dist", ["balanced", "imbalanced"])
@pytest.mark.parametrize("fuse_wgrad_accumulation", all_boolean)
def test_nvfp4_grouped_tensor_cutlass_matches_cublas(
    num_gemms, dist, fuse_wgrad_accumulation, monkeypatch
):
    """End-to-end GroupedLinear fwd+bwd on the grouped-tensor path: CUTLASS
    backend (env=1) vs cuBLAS single-launch (env=0). Identical weights / x / dy
    (RNG reset before each run) feed both, so any divergence in output, dgrad or
    wgrad is a backend bug. fuse_wgrad_accumulation=True (fp32 main_grad) is what
    routes wgrad through CUTLASS, mirroring Megatron training."""
    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "1")
    # Default recipe (RHT on): the graph-safe grouped NVFP4 quant kernel is only
    # implemented for the RHT path, so disable_rht=True is not supported here.
    nvfp4_recipe = recipe.NVFP4BlockScaling()
    K, N = 512, 512  # all dims %128 -> path eligible
    m_splits = _nvfp4_grouped_m_splits(num_gemms, dist)
    total_m = sum(m_splits)
    m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device="cuda")

    torch.manual_seed(0)
    model = GroupedLinear(
        num_gemms,
        K,
        N,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    ).eval()
    x = torch.randn(total_m, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn(total_m, N, dtype=torch.bfloat16, device="cuda")
    init_mg = (
        [torch.randn(N, K, dtype=torch.float32, device="cuda") for _ in range(num_gemms)]
        if fuse_wgrad_accumulation
        else None
    )

    def run(cutlass: bool):
        monkeypatch.setenv(_NVFP4_CUTLASS_ENV, "1" if cutlass else "0")
        reset_rng_states()  # identical quantization randoms in both runs
        FP8GlobalStateManager.reset()
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        if fuse_wgrad_accumulation:
            for i in range(num_gemms):
                getattr(model, f"weight{i}").main_grad = init_mg[i].clone()
        with autocast(enabled=True, recipe=nvfp4_recipe):
            out = model(x, m_splits_t)
        out.backward(dy)
        snap = {"out": out.detach().float().clone(), "dgrad": x.grad.detach().float().clone()}
        for i in range(num_gemms):
            w = getattr(model, f"weight{i}")
            g = w.main_grad if fuse_wgrad_accumulation else w.grad
            snap[f"wgrad{i}"] = g.detach().float().clone()
        return snap

    ref = run(cutlass=False)
    test = run(cutlass=True)
    for key in ref:
        abs_d, rel_d, _ = _diff(ref[key], test[key])
        assert (
            abs_d <= 1e-2 or rel_d <= 5e-3
        ), f"{key}: cutlass vs cuBLAS diverged (max_abs={abs_d:.4g}, rel={rel_d:.4g})"


@pytest.mark.skipif(
    not nvfp4_cutlass_grouped_available,
    reason="NVFP4 CUTLASS grouped GEMM requires Blackwell (SM100) + NVFP4",
)
def test_nvfp4_grouped_tensor_cutlass_cuda_graph_safe(monkeypatch):
    """The CUTLASS backend on the grouped-tensor path is CUDA-graph capturable.
    Split metadata is passed as a device int64 tensor (no host->device copy to
    capture) and the captured forward must match eager."""
    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "1")
    monkeypatch.setenv(_NVFP4_CUTLASS_ENV, "1")
    FP8GlobalStateManager.reset()
    # Default recipe (RHT on): the graph-safe grouped NVFP4 quant kernel is only
    # implemented for the RHT path, so disable_rht=True is not supported here.
    nvfp4_recipe = recipe.NVFP4BlockScaling()
    num_gemms, K, N = 3, 512, 512
    m_splits = [256, 128, 384]  # 128-aligned, unequal
    total_m = sum(m_splits)
    m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device="cuda")

    torch.manual_seed(0)
    model = GroupedLinear(
        num_gemms, K, N, bias=False, params_dtype=torch.bfloat16, device="cuda"
    ).eval()
    x = torch.randn(total_m, K, dtype=torch.bfloat16, device="cuda")

    reset_rng_states()
    with torch.no_grad(), autocast(enabled=True, recipe=nvfp4_recipe):
        out_eager = model(x, m_splits_t).detach().float().clone()
    torch.cuda.synchronize()

    # Warmup on a side stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad(), autocast(enabled=True, recipe=nvfp4_recipe):
                _ = model(x, m_splits_t)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad(), autocast(enabled=True, recipe=nvfp4_recipe):
            out_graph = model(x, m_splits_t)
    g.replay()
    torch.cuda.synchronize()

    abs_d, rel_d, _ = _diff(out_eager, out_graph.float())
    assert (
        abs_d <= 1e-2 or rel_d <= 5e-3
    ), f"graph vs eager diverged (max_abs={abs_d:.4g}, rel={rel_d:.4g})"
