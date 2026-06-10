# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import random
from typing import List, Optional, Sequence

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
    QuantizerRole,
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
from transformer_engine.pytorch.ops._common import (
    _cudnn_frontend_supports_grouped_gemm_srelu,
    _cudnn_frontend_version_supported,
    is_glu_activation,
)
from utils import (
    MegatronTrainingHelper,
    ModelConfig,
    assert_close,
    assert_close_grads,
    dtype_tols,
    make_recipe,
    make_reference_and_test_tensors,
    maybe_skip_quantization,
    quantization_tols,
    recipe_id,
    reset_rng_states,
    skip_unsupported_backward_override,
)

# Check supported quantization schemes
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

_quantization_list: list = [None]
if fp8_available:
    _quantization_list.extend(("fp8_delayed_scaling", "fp8_current_scaling"))
if mxfp8_available:
    _quantization_list.append("mxfp8")
if nvfp4_available:
    _quantization_list.extend(("nvfp4", "nvfp4_4over6", "nvfp4_rht"))


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


_FUSED_GROUPED_GEMM_ENV = "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"


def _force_legacy_grouped_linear_path(monkeypatch) -> None:
    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "0")


def _enable_fused_grouped_linear_path(monkeypatch) -> None:
    monkeypatch.setenv(_FUSED_GROUPED_GEMM_ENV, "1")


def _enable_single_grouped_param(monkeypatch) -> None:
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")


def _enable_fused_grouped_mlp(monkeypatch) -> None:
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")


def _make_grouped_split_sizes(
    group_size: int,
    split_alignment: int,
    *,
    start: int = 0,
    dtype: torch.dtype = torch.int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Construct shuffled per-group token counts."""
    split_sizes = [split_alignment * (i + start) for i in range(group_size)]
    random.shuffle(split_sizes)
    return torch.tensor(split_sizes, dtype=dtype, device=device)


def _grouped_weight_params(
    op: torch.nn.Module,
    group_size: int,
    *,
    single_grouped_weight: bool,
) -> list[torch.nn.Parameter]:
    """Extract weight parameters from grouped linear module or op."""
    if single_grouped_weight:
        return [op.weight]
    return [getattr(op, f"weight{i}") for i in range(group_size)]


def _grouped_bias_params(
    op: torch.nn.Module,
    group_size: int,
    *,
    single_grouped_bias: bool,
) -> list[torch.nn.Parameter]:
    """Extract bias parameters from grouped linear module or op."""
    if single_grouped_bias:
        return [op.bias]
    return [getattr(op, f"bias{i}") for i in range(group_size)]


def _copy_grouped_linear_params(
    op: torch.nn.Module,
    weights: Sequence[torch.Tensor],
    biases: Optional[Sequence[Optional[torch.Tensor]]] = None,
    *,
    single_grouped_weight: bool = False,
    single_grouped_bias: bool = False,
) -> None:
    """Copy values into grouped linear params"""

    # Copy into weights
    if single_grouped_weight:
        weight_parts = op.weight.quantized_tensors
        if weight_parts is None:
            weight_parts = op.weight.split_into_quantized_tensors()
        for dst, src in zip(weight_parts, weights):
            dst.copy_(src)
    else:
        for group_idx, weight in enumerate(weights):
            getattr(op, f"weight{group_idx}").copy_(weight)

    # Copy into biases
    if biases is None:
        pass
    elif single_grouped_bias:
        bias_parts = op.bias.split_into_quantized_tensors()
        for dst, src in zip(bias_parts, biases):
            dst.reshape(-1).copy_(src)
    else:
        for group_idx, bias in enumerate(biases):
            getattr(op, f"bias{group_idx}").copy_(bias)


def _fill_main_grads(
    params: Sequence[torch.nn.Parameter],
    value: float,
    *,
    device: torch.device | str,
) -> None:
    """Construct param main_grad if needed and fill with value"""
    with torch.no_grad():
        for param in params:
            if getattr(param, "main_grad", None) is None:
                param.main_grad = torch.empty(param.size(), device=device, dtype=torch.float32)
            param.main_grad.fill_(value)


def _clone_grads(params: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.grad.detach().clone() for param in params]


def _clone_main_grads(params: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.main_grad.detach().clone() for param in params]


def _stack_cloned_attr(params: Sequence[torch.nn.Parameter], attr: str) -> torch.Tensor:
    values = [getattr(param, attr).detach().clone() for param in params]
    if len(values) == 1:
        return values[0]
    return torch.stack(values, dim=0)


def _make_scaled_grouped_mlp_activation(
    activation: str,
    *,
    glu_interleave_size: Optional[int],
    geglu_limit: float = 7.0,
    geglu_alpha: float = 1.702,
    geglu_offset: float = 1.0,
) -> torch.nn.Module:
    if activation == "scaled_swiglu":
        return te.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
    if activation == "scaled_clamped_qgeglu_custom":
        return te.ops.ScaledClampedQGeGLU(
            glu_interleave_size=glu_interleave_size,
            limit=geglu_limit,
            alpha=geglu_alpha,
            glu_linear_offset=geglu_offset,
        )
    if activation.startswith("scaled_clamped_qgeglu"):
        return te.ops.ScaledClampedQGeGLU(glu_interleave_size=glu_interleave_size)
    if activation == "scaled_srelu":
        return te.ops.ScaledSReLU()
    raise ValueError(f"Unexpected activation ({activation})")


def _skip_invalid_grouped_mlp_case(
    *,
    activation: str,
    activation_is_glu: bool,
    bias: bool,
    dtype: torch.dtype,
    quantization: Optional[str],
    single_grouped_weight: bool,
    single_grouped_bias: bool,
    glu_interleave_size: Optional[int],
    device: torch.device | str,
) -> None:
    with_quantization = quantization is not None
    maybe_skip_quantization(quantization, device=device, dtype=dtype)
    if single_grouped_weight and quantization != "mxfp8":
        pytest.skip("single_grouped_weight is only supported for MXFP8 quantization")
    if single_grouped_bias and not bias:
        pytest.skip("single_grouped_bias requires bias=True")
    if with_quantization and dtype not in (torch.bfloat16, torch.float16):
        pytest.skip("Quantized group GEMM is only supported with BF16/FP16")
    if not activation_is_glu and quantization not in ("mxfp8", "nvfp4", "nvfp4_rht"):
        pytest.skip("Scaled unary grouped MLP is only supported with MXFP8 or NVFP4")
    if not activation_is_glu and glu_interleave_size is not None:
        pytest.skip("Unary activations do not use GLU interleaving")
    if quantization == "nvfp4_4over6":
        pytest.skip("NVFP4 4over6 grouped quantization is not supported")
    if activation == "scaled_srelu" and quantization in ("nvfp4", "nvfp4_rht") and bias:
        pytest.skip("NVFP4 SReLU grouped MLP coverage is limited to no-bias")
    if quantization == "nvfp4_rht":
        if activation == "scaled_swiglu" and (bias or glu_interleave_size != 32):
            pytest.skip("NVFP4 RHT SwiGLU grouped MLP coverage is limited to no-bias")
        if activation not in ("scaled_swiglu", "scaled_srelu"):
            pytest.skip("NVFP4 RHT grouped MLP coverage is limited to SwiGLU and SReLU")
    if (
        with_quantization
        and quantization in ("nvfp4", "nvfp4_row_scaled", "nvfp4_4over6", "nvfp4_rht")
        and activation.startswith("scaled_clamped_qgeglu")
        and bias
    ):
        # TODO: ksivaman: Need to debug numerics for this case.
        pytest.skip("Bias/dbias not yet supported in NVFP4 fused grouped MLP with GeGLU")


class TestGroupedLinearModule:
    """Tests for te.GroupedLinear module API.

    Reference: sequential te.Linear modules with shared weights.
    """

    @pytest.fixture(autouse=True)
    def _use_legacy_grouped_linear_path(self, monkeypatch):
        _force_legacy_grouped_linear_path(monkeypatch)
        yield
        monkeypatch.delenv(_FUSED_GROUPED_GEMM_ENV, raising=False)

    @pytest.fixture(autouse=True)
    def _use_single_grouped_param(self, monkeypatch):
        _enable_single_grouped_param(monkeypatch)


    @pytest.mark.parametrize("dtype", param_types, ids=str)
    @pytest.mark.parametrize("num_gemms", [1, 3, 6])
    @pytest.mark.parametrize("bs", batch_sizes)
    @pytest.mark.parametrize("model", ["126m"])
    @pytest.mark.parametrize("recipe", fp8_recipes + [None], ids=recipe_id)
    @pytest.mark.parametrize("fp8_model_params", all_boolean)
    @pytest.mark.parametrize("fuse_wgrad_accumulation", all_boolean)
    @pytest.mark.parametrize("bias", all_boolean)
    @pytest.mark.parametrize("delay_wgrad_compute", all_boolean)
    def test_grouped_linear_accuracy(
        self,
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
        self,
        dtype,
        num_gemms,
        bs,
        model,
        fuse_wgrad_accumulation,
        delay_wgrad_compute,
        monkeypatch,
    ):
        monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
        self.test_grouped_linear_accuracy(
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
        self,
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


    @pytest.mark.parametrize("save_original_input", [False, True])
    @pytest.mark.parametrize("dtype", param_types)
    @pytest.mark.parametrize("num_gemms", [3, 6])
    @pytest.mark.parametrize("bs", batch_sizes)
    @pytest.mark.parametrize("model", ["126m"])
    @pytest.mark.parametrize("fp8", [True])
    @pytest.mark.parametrize("recipe", fp8_recipes, ids=recipe_id)
    @pytest.mark.parametrize("fp8_model_params", all_boolean)
    def test_padding_grouped_linear_accuracy(
        self,
        dtype,
        num_gemms,
        bs,
        model,
        fp8,
        recipe,
        fp8_model_params,
        save_original_input,
        parallel_mode=None,
    ):
        if fp8_model_params and NVTE_TEST_NVINSPECT_ENABLED:
            pytest.skip("FP8 parameters are not supported in debug mode.")
        if save_original_input and recipe.delayed():
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
                save_original_input=save_original_input,
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


    @staticmethod
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
            _copy_grouped_linear_params(
                grouped_linear,
                weights,
                biases if bias else None,
            )

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
        outputs.extend(getattr(grouped_linear, f"weight{i}").grad for i in range(num_gemms))
        if bias:
            outputs.extend(getattr(grouped_linear, f"bias{i}").grad for i in range(num_gemms))
        return _clone_outputs(outputs)


    @pytest.mark.parametrize(
        "fp8_recipe",
        [
            None,
            pytest.param(
                recipe.MXFP8BlockScaling(),
                marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
            ),
        ],
        ids=["bf16", "mxfp8"],
    )
    @pytest.mark.parametrize("bias", all_boolean)
    @pytest.mark.parametrize("fp8_model_params", all_boolean)
    @pytest.mark.parametrize("delay_wgrad_compute", all_boolean)
    def test_grouped_linear_grouped_tensor_path_matches_legacy(
        self,
        fp8_recipe, bias, fp8_model_params, delay_wgrad_compute, monkeypatch
    ):
        if torch.cuda.get_device_capability() < (10, 0):
            pytest.skip("GroupedTensor grouped GEMM path requires SM100+")

        use_fp8 = fp8_recipe is not None
        if fp8_model_params and not use_fp8:
            pytest.skip("fp8_model_params requires FP8")

        dtype = torch.bfloat16
        num_gemms = 3
        in_features = 64
        out_features = 64
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

        outputs_legacy = self._run_grouped_linear_path(
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
        outputs_grouped_tensor = self._run_grouped_linear_path(
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
            assert_close(grouped_tensor_out, legacy_out, **tols)


    @pytest.mark.parametrize(
        "fp8_recipe",
        [
            None,
            pytest.param(
                recipe.MXFP8BlockScaling(),
                marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
            ),
        ],
        ids=["bf16", "mxfp8"],
    )
    @pytest.mark.parametrize("bias", all_boolean)
    def test_grouped_linear_fused_path_cuda_graph_safe(self, fp8_recipe, bias, monkeypatch):
        """Fused GroupedTensor GEMM path should be CUDA graph capturable."""
        if torch.cuda.get_device_capability() < (10, 0):
            pytest.skip("GroupedTensor grouped GEMM path requires SM100+")

        _enable_fused_grouped_linear_path(monkeypatch)
        FP8GlobalStateManager.reset()

        use_fp8 = fp8_recipe is not None
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
        assert_close(graph_out, expected_out, **tols)
        assert_close(graph_dx, expected_x.grad, **tols)
        for graph_grad, param in zip(graph_param_grads, grouped_linear.parameters()):
            assert param.grad is not None
            assert_close(graph_grad, param.grad, **tols)


def _clone_outputs(outputs):
    return [None if out is None else out.detach().clone() for out in outputs]


@pytest.fixture(autouse=True)
def _reset_fp8_state():
    yield
    FP8GlobalStateManager.reset()


class TestGroupedGemm:
    """Tests for raw grouped GEMM kernels (cpp_extensions)."""

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
    def test_grouped_gemm(self, shape, dtype, layout, accumulate, use_cutlass, monkeypatch):
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
    def test_grouped_gemm_cutlass_empty_groups(self, layout, monkeypatch):
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


    @staticmethod
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


    @staticmethod
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


    @staticmethod
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


    @staticmethod
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
    def test_grouped_gemm_grouped_tensor(self, z, m, n, k, case, layout, accumulate, use_bias_scale) -> None:
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
        out_ref = self._apply_grouped_bias_ref(out_ref_no_bias, bias, bias_scale, m_sizes, dtype)
        # Create grouped tensors based on case
        device = A[0].device
        grouped_A = A
        grouped_out = out
        grouped_out_bias = [o.clone() for o in out]
        grouped_out_no_bias = [o.clone() for o in out]
        grouped_bias = None
        if layout == "TN":
            grouped_A = (
                self._make_grouped_tensor_uniform(z, n, k, device, dtype) if case != "discrete_in" else A
            )  # weight
            grouped_B = self._make_grouped_tensor_from_splits(m_sizes, k, device, dtype)  # input
            if case != "discrete_out":
                grouped_out = self._make_grouped_tensor_from_splits(m_sizes, n, device, dtype)  # output
                grouped_out_bias = self._make_grouped_tensor_from_splits(m_sizes, n, device, dtype)
                grouped_out_no_bias = self._make_grouped_tensor_from_splits(m_sizes, n, device, dtype)
        elif layout == "NN":
            grouped_A = (
                self._make_grouped_tensor_uniform(z, n, k, device, dtype) if case != "discrete_in" else A
            )  # weight
            grouped_B = self._make_grouped_tensor_from_splits(m_sizes, n, device, dtype)  # grad_output
            if case != "discrete_out":
                grouped_out = self._make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
                grouped_out_bias = self._make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
                grouped_out_no_bias = self._make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
        else:  # layout == "NT"
            grouped_A = (
                self._make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
                if case != "discrete_in"
                else A
            )  # input
            grouped_B = self._make_grouped_tensor_from_splits(m_sizes, n, device, dtype)  # grad_output
            if case != "discrete_out":
                grouped_out = self._make_grouped_tensor_uniform(z, n, k, device, dtype)  # wgrad
                grouped_out_bias = self._make_grouped_tensor_uniform(z, n, k, device, dtype)
                grouped_out_no_bias = self._make_grouped_tensor_uniform(z, n, k, device, dtype)
        self._pack_grouped_tensor(grouped_B, B)
        if case != "discrete_out":
            self._pack_grouped_tensor(grouped_out, out)
            self._pack_grouped_tensor(grouped_out_bias, out)
            self._pack_grouped_tensor(grouped_out_no_bias, out)
        if case != "discrete_in":
            self._pack_grouped_tensor(grouped_A, A)

        if bias is not None:
            grouped_bias = self._make_grouped_tensor_uniform(z, 1, bias_last_dim, device, dtype)
            self._pack_grouped_tensor(grouped_bias, bias)

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

        out_grouped_manual_bias = self._apply_grouped_bias_ref(
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
    def test_grouped_gemm_grouped_tensor_zero_work(self, layout, accumulate, quant_type) -> None:
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
                grouped_A = self._make_grouped_tensor_quantized_mxfp8(
                    weight_tensors,
                    rowwise=transa,
                    columnwise=not transa,
                    device=device,
                )
            else:
                grouped_A = self._make_grouped_tensor_uniform(z, n, k, device, dtype)
                self._pack_grouped_tensor(grouped_A, weight_tensors)
        else:  # NT
            grouped_A = _make_zero_tokens_grouped_tensor(k, is_a=True)

        b_last_dim = k if layout == "TN" else n
        grouped_B = _make_zero_tokens_grouped_tensor(b_last_dim, is_a=False)

        if layout == "NT":
            out = [torch.randn(n, k, dtype=dtype, device=device) for _ in range(z)]
            grouped_out = self._make_grouped_tensor_uniform(z, n, k, device, dtype)
            self._pack_grouped_tensor(grouped_out, out)
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


    @staticmethod
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


    @staticmethod
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
        self,
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
        grouped_A = self._make_grouped_tensor_quantized_mxfp8(
            A,
            rowwise=a_rowwise,
            columnwise=a_columnwise,
            device="cuda",
            is_weight=a_is_weight,
        )
        grouped_B = self._make_grouped_tensor_quantized_mxfp8(
            B, rowwise=b_rowwise, columnwise=b_columnwise, device="cuda"
        )
        A_fp8 = self._per_tensor_quantize_mxfp8(A, rowwise=a_rowwise, columnwise=a_columnwise)
        B_fp8 = self._per_tensor_quantize_mxfp8(B, rowwise=b_rowwise, columnwise=b_columnwise)

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
                grouped_out = self._make_grouped_tensor_from_splits(m_sizes, n, device, dtype)
            elif layout == "NN":
                grouped_out = self._make_grouped_tensor_from_splits(m_sizes, k, device, dtype)
            else:  # layout == "NT"
                grouped_out = self._make_grouped_tensor_uniform(z, n, k, device, dtype)
            self._pack_grouped_tensor(grouped_out, out)

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
    def test_fp8_grouped_gemm(self, shape, accumulate):
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


class TestGroupedLinearOps:
    """Tests for te.ops.GroupedLinear (ops/fuser API)."""

    @pytest.fixture(autouse=True)
    def _use_single_grouped_param(self, monkeypatch):
        _enable_single_grouped_param(monkeypatch)

    @pytest.mark.parametrize("swizzle_type", ["mxfp8_rowwise", "mxfp8_columnwise", "nvfp4"])
    def test_swizzle_scales_and_pack_ptrs_for_discrete_weights(
        self,
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


    @pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
    @pytest.mark.parametrize(
        "quantization",
        [None] + (["mxfp8"] if mxfp8_available else []),
    )
    @pytest.mark.parametrize("quantized_weight", (False, True))
    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("single_grouped_bias", (False, True))
    @pytest.mark.parametrize("accumulate_into_main_grad", (False, True))
    def test_grouped_linear_cuda_graph_safe(
        self,
        *,
        dtype: torch.dtype,
        quantization: Optional[str],
        quantized_weight: bool,
        bias: bool,
        single_grouped_weight: bool,
        single_grouped_bias: bool,
        accumulate_into_main_grad: bool,
        device: torch.device = "cuda",
        group_size: int = 4,
        in_features: int = 128,
        out_features: int = 128,
        split_alignment: int = 128,
        token_padding: int = 256,
    ) -> None:
        """GroupedLinear forward+backward should be CUDA graph capturable.

        Exercises the grouped-tensor / cublas-grouped-gemm path which uses
        GPU-resident split offsets and is the only flow safe to capture.
        """
        if torch.cuda.get_device_capability() < (10, 0):
            pytest.skip("Grouped GEMM CUDA-graph-safe path requires SM100+ (Blackwell)")
        # Skip invalid configurations
        if quantization is None and quantized_weight:
            pytest.skip("quantized_weight requires a quantization recipe")
        if single_grouped_bias and not bias:
            pytest.skip("single_grouped_bias requires bias=True")

        # Split sizes (statically pinned for graph capture)
        split_sizes = _make_grouped_split_sizes(
            group_size,
            split_alignment,
            start=1,
            dtype=torch.int,
            device=device,
        )
        # Pad input tokens to validate the sync-free flow
        in_shape = (split_sizes.sum().item() + token_padding, in_features)
        out_shape = (in_shape[0], out_features)

        recipe = make_recipe(quantization)
        with te.quantized_model_init(enabled=quantized_weight, recipe=recipe):
            op = te.ops.GroupedLinear(
                group_size,
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype,
                accumulate_into_main_grad=accumulate_into_main_grad,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
            )
        weight_params = _grouped_weight_params(
            op,
            group_size,
            single_grouped_weight=single_grouped_weight,
        )

        def _init_main_grads(value: float = 0.0) -> None:
            if not accumulate_into_main_grad:
                return
            _fill_main_grads(weight_params, value, device=device)

        def _collect_main_grads() -> list[torch.Tensor]:
            return _clone_main_grads(weight_params)

        def _zero_param_grads() -> None:
            for param in op.parameters():
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                else:
                    param.grad.zero_()

        static_split_sizes = split_sizes.clone()

        def train_step(
            x: torch.Tensor,
            dy: torch.Tensor,
            out_buf: torch.Tensor,
            *,
            use_graphed: bool,
        ) -> torch.Tensor:
            with te.autocast(enabled=quantization is not None, recipe=recipe):
                out = (
                    graphed_module(x, static_split_sizes)
                    if use_graphed
                    else op(x, static_split_sizes)
                )
            out.backward(dy)
            out_buf.copy_(out)
            return out_buf

        _init_main_grads(0.0)

        static_x = torch.randn(in_shape, device=device, dtype=dtype, requires_grad=True)
        static_dy = torch.randn(out_shape, device=device, dtype=dtype)
        static_out_buf = torch.empty(out_shape, device=device, dtype=dtype)

        graphed_module = te.make_graphed_callables(
            op,
            (static_x, static_split_sizes),
            num_warmup_iters=3,
            enabled=quantization is not None,
            recipe=recipe,
        )

        # Replace static buffers with fresh data (graph captures must replay
        # against new inputs without re-recording).
        fresh_x = torch.randn_like(static_x)
        fresh_dy = torch.randn_like(static_dy)
        with torch.no_grad():
            static_x.copy_(fresh_x)
            static_dy.copy_(fresh_dy)

        # Reset grads & main_grads so the captured iteration starts fresh.
        _zero_param_grads()
        _init_main_grads(0.5)
        if static_x.grad is not None:
            static_x.grad.zero_()

        # Replay the graph
        graph_out = (
            train_step(static_x, static_dy, static_out_buf, use_graphed=True).detach().clone()
        )
        torch.cuda.synchronize()
        graph_dx = static_x.grad.detach().clone()
        if accumulate_into_main_grad:
            graph_main_grads = _collect_main_grads()
            graph_param_grads: list[torch.Tensor] = []
        else:
            graph_main_grads = []
            graph_param_grads = [param.grad.detach().clone() for param in op.parameters()]

        # Reference: same op invoked eagerly with the same fresh inputs and
        # the same starting grad/main_grad state.
        _zero_param_grads()
        _init_main_grads(0.5)
        static_x.grad.zero_()

        expected_x = fresh_x.detach().clone().requires_grad_(True)
        expected_dy = fresh_dy.detach().clone()
        with te.autocast(enabled=quantization is not None, recipe=recipe):
            expected_out = op(expected_x, static_split_sizes)
        expected_out.backward(expected_dy)

        tols = dtype_tols(dtype)
        if quantization is not None:
            tols = quantization_tols(quantization)

        assert_close(graph_out, expected_out, **tols)
        assert_close(graph_dx, expected_x.grad, **tols)
        if accumulate_into_main_grad:
            for g, w in zip(graph_main_grads, weight_params):
                assert_close(g, w.main_grad, **tols)
        else:
            for g, param in zip(graph_param_grads, op.parameters()):
                assert_close(g, param.grad, **tols)


    @pytest.mark.parametrize("delay_wgrad_compute", (False, True))
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("single_grouped_bias", (False, True))
    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("dtype", param_types, ids=str)
    @pytest.mark.parametrize("quantization", _quantization_list)
    @pytest.mark.parametrize("quantized_compute", (False, True))
    @pytest.mark.parametrize("quantized_weight", (False, True))
    @pytest.mark.parametrize("input_requires_grad", (False, True))
    @pytest.mark.parametrize("weight_requires_grad", (False, True))
    def test_grouped_linear(
        self,
        *,
        group_size: int = 4,
        bias: bool,
        weight_shape: tuple = (128, 128),
        split_alignment: int = 128,
        dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
        quantized_compute: bool,
        quantized_weight: bool,
        input_requires_grad: bool,
        weight_requires_grad: bool,
        delay_wgrad_compute: bool,
        single_grouped_weight: bool,
        single_grouped_bias: bool,
    ) -> None:
        """te.ops.GroupedLinear forward+backward accuracy"""

        # Split sizes
        split_sizes = _make_grouped_split_sizes(
            group_size,
            split_alignment,
            dtype=torch.int,
            device=device,
        )

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = (split_sizes.sum().item(), in_features)
        out_shape = (in_shape[0], out_features)

        # Skip invalid configurations
        maybe_skip_quantization(quantization, dims=in_shape, device=device, dtype=dtype)
        maybe_skip_quantization(quantization, dims=out_shape)
        if quantization is None and (quantized_compute or quantized_weight):
            pytest.skip("Quantization scheme is not specified")
        if quantization is not None and not (quantized_compute or quantized_weight):
            pytest.skip("Quantization scheme is not used")
        if quantization is not None and dtype not in (torch.bfloat16, torch.float16):
            pytest.skip("Quantized group GEMM is only supported with BF16/FP16")
        if single_grouped_bias and not bias:
            pytest.skip("single_grouped_bias requires bias=True")
        if (
            single_grouped_weight
            and quantized_weight
            and quantization in ("fp8_delayed_scaling", "fp8_current_scaling")
        ):
            pytest.skip(
                "single_grouped_weight does not support FP8 delayed/current scaling "
                "with quantized_model_init"
            )

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            quantization=quantization,
            test_dtype=dtype,
            test_device=device,
            requires_grad=input_requires_grad,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            quantization=quantization,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )
        ws_ref, ws_test = [], []
        bs_ref, bs_test = [], []
        for _ in range(group_size):
            w_ref, w_test = make_reference_and_test_tensors(
                (out_features, in_features),
                quantization=quantization,
                test_dtype=dtype,
                test_device=device,
                quantizer_role=QuantizerRole(tensor_type="weight"),
                requires_grad=weight_requires_grad,
            )
            b_ref, b_test = None, None
            if bias:
                b_ref, b_test = make_reference_and_test_tensors(
                    out_features,
                    test_dtype=dtype,
                    test_device=device,
                    requires_grad=weight_requires_grad,
                )
            ws_ref.append(w_ref)
            ws_test.append(w_test)
            bs_ref.append(b_ref)
            bs_test.append(b_test)

        # Plain PyTorch reference implementation
        xs_ref = torch.split(x_ref, split_sizes.tolist())
        ys_ref = []
        for x, w, b in zip(xs_ref, ws_ref, bs_ref):
            ys_ref.append(torch.nn.functional.linear(x, w, bias=b))
        y_ref = torch.cat(ys_ref)
        if input_requires_grad or weight_requires_grad:
            y_ref.backward(dy_ref)

        # Construct te.ops.GroupedLinear
        recipe = make_recipe(quantization)
        with te.quantized_model_init(enabled=quantized_weight, recipe=recipe):
            op = te.ops.GroupedLinear(
                group_size,
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype,
                delay_wgrad_compute=delay_wgrad_compute,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
            )
        with torch.no_grad():
            _copy_grouped_linear_params(
                op,
                ws_test,
                bs_test if bias else None,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
            )
            del ws_test, bs_test
            for param in op.parameters():
                param.requires_grad_(requires_grad=weight_requires_grad)

        # Forward and backward pass
        with te.autocast(enabled=quantized_compute, recipe=recipe):
            y_test = op(x_test, split_sizes)
        if input_requires_grad or weight_requires_grad:
            y_test.backward(dy_test)
            if delay_wgrad_compute and weight_requires_grad:
                op.backward_dw()

        # Expected numerical tolerances
        tols = dtype_tols(dtype)
        if dtype == torch.float32:
            tols = dtype_tols(torch.float16)  # TF32 GEMM
        if quantized_compute:
            tols = quantization_tols(quantization)

        # Check results
        assert_close(y_test, y_ref, **tols)
        assert_close_grads(x_test, x_ref, **tols)
        if single_grouped_weight:
            if weight_requires_grad:
                w_ref_grad = torch.stack([w.grad for w in ws_ref], dim=0)
                assert_close(op.weight.grad, w_ref_grad, **tols)
            else:
                assert op.weight.grad is None
        else:
            for group_idx in range(group_size):
                w_test = getattr(op, f"weight{group_idx}")
                assert_close_grads(w_test, ws_ref[group_idx], **tols)
        if bias:
            if single_grouped_bias:
                if weight_requires_grad:
                    b_ref_grad = torch.stack([b.grad for b in bs_ref], dim=0)
                    assert_close(op.bias.grad, b_ref_grad, **tols)
                else:
                    assert op.bias.grad is None
            else:
                for group_idx in range(group_size):
                    b_test = getattr(op, f"bias{group_idx}")
                    assert_close_grads(b_test, bs_ref[group_idx], **tols)


class TestGroupedMLP:
    """Tests for grouped MLP patterns (te.ops.GroupedLinear + activation)."""

    @pytest.fixture(autouse=True)
    def _enable_grouped_mlp_envvars(self, monkeypatch):
        _enable_single_grouped_param(monkeypatch)
        _enable_fused_grouped_mlp(monkeypatch)

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("quantization", _quantization_list)
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize(
        "activation",
        ("scaled_swiglu", "scaled_clamped_qgeglu", "scaled_clamped_qgeglu_custom", "scaled_srelu"),
    )
    def test_grouped_mlp(
        self,
        *,
        group_size: int = 4,
        hidden_size: int = 256,
        bias: bool,
        dtype: torch.dtype = torch.bfloat16,
        quantization: Optional[str],
        single_grouped_weight: bool,
        accumulate_into_main_grad: bool = False,
        device: torch.device = "cuda",
        split_alignment: int = 256,
        delay_wgrad_compute: bool = False,
        activation: str,
    ) -> None:
        """GroupedLinear + scaled activation + GroupedLinear"""

        # Grouped MLP fused op requires GLU interleaving
        activation_is_glu = activation in (
            "scaled_swiglu", "scaled_clamped_qgeglu", "scaled_clamped_qgeglu_custom"
        )
        glu_interleave_size = 32 if activation_is_glu else None

        # Enable grouped bias if weights are grouped
        single_grouped_bias = bias and single_grouped_weight

        _skip_invalid_grouped_mlp_case(
            activation=activation,
            activation_is_glu=activation_is_glu,
            bias=bias,
            dtype=dtype,
            quantization=quantization,
            single_grouped_weight=single_grouped_weight,
            single_grouped_bias=single_grouped_bias,
            glu_interleave_size=glu_interleave_size,
            device=device,
        )
        with_quantization = quantization is not None

        fc1_out_features = 2 * hidden_size if activation_is_glu else hidden_size
        if activation == "scaled_clamped_qgeglu_custom":
            geglu_limit, geglu_alpha, geglu_offset = 5.0, 1.5, 0.5
        else:
            geglu_limit, geglu_alpha, geglu_offset = 7.0, 1.702, 1.0

        # Split sizes (one group intentionally empty to test the zero-token case)
        split_sizes = _make_grouped_split_sizes(
            group_size,
            split_alignment,
            dtype=torch.int,
            device=device,
        )
        in_shape = (split_sizes.sum().item(), hidden_size)
        out_shape = in_shape

        # Reference tensors: float64 CPU; test tensors: target dtype on CUDA
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            min=-0.25, max=0.25,
            quantization=quantization,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            min=-0.25, max=0.25,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )
        probs_ref, probs_test = make_reference_and_test_tensors(
            (in_shape[0],),
            min=0.1, max=1.0,
            test_dtype=dtype,
            test_device=device,
        )

        fc1_ws_ref, fc1_ws_test = [], []
        fc1_bs_ref, fc1_bs_test = [], []
        fc2_ws_ref, fc2_ws_test = [], []
        fc2_bs_ref, fc2_bs_test = [], []
        for _ in range(group_size):
            w1_ref, w1_test = make_reference_and_test_tensors(
                (fc1_out_features, hidden_size),
                min=-0.125, max=0.125,
                quantization=quantization,
                test_dtype=dtype,
                test_device=device,
                quantizer_role=QuantizerRole(tensor_type="weight"),
            )
            fc1_ws_ref.append(w1_ref)
            fc1_ws_test.append(w1_test)
            w2_ref, w2_test = make_reference_and_test_tensors(
                (hidden_size, hidden_size),
                min=-0.125, max=0.125,
                quantization=quantization,
                test_dtype=dtype,
                test_device=device,
                quantizer_role=QuantizerRole(tensor_type="weight"),
            )
            fc2_ws_ref.append(w2_ref)
            fc2_ws_test.append(w2_test)
            if bias:
                b1_ref, b1_test = make_reference_and_test_tensors(
                    (fc1_out_features,),
                    min=-0.5, max=0.5,
                    test_dtype=dtype,
                    test_device=device,
                )
                fc1_bs_ref.append(b1_ref)
                fc1_bs_test.append(b1_test)
                b2_ref, b2_test = make_reference_and_test_tensors(
                    (hidden_size,),
                    min=-0.5, max=0.5,
                    test_dtype=dtype,
                    test_device=device,
                )
                fc2_bs_ref.append(b2_ref)
                fc2_bs_test.append(b2_test)
            else:
                fc1_bs_ref.append(None)
                fc1_bs_test.append(None)
                fc2_bs_ref.append(None)
                fc2_bs_test.append(None)

        def _apply_activation(x: torch.Tensor) -> torch.Tensor:
            if glu_interleave_size is not None:
                x = x.reshape(-1, 2 * hidden_size // (2 * glu_interleave_size), 2, glu_interleave_size)
                x = x.transpose(1, 2).reshape(-1, 2 * hidden_size)
            if activation == "scaled_swiglu":
                x1, x2 = x.chunk(2, dim=-1)
                return torch.nn.functional.silu(x1) * x2
            if activation.startswith("scaled_clamped_qgeglu"):
                x1, x2 = x.chunk(2, dim=-1)
                lim = torch.tensor(geglu_limit, device=x1.device, dtype=x1.dtype)
                x1c = torch.minimum(x1, lim)
                x2c = torch.clamp(x2, -lim, lim)
                return (x2c + geglu_offset) * (x1c * torch.sigmoid(geglu_alpha * x1c))
            if activation == "scaled_srelu":
                return torch.nn.functional.relu(x).square()
            raise ValueError(f"Unexpected activation ({activation})")

        # Reference implementation (float64 CPU PyTorch)
        xs = torch.split(x_ref, split_sizes.tolist())
        probs = torch.split(probs_ref, split_sizes.tolist())
        ys = []
        for group_idx in range(group_size):
            x = xs[group_idx]
            fc1_out = torch.nn.functional.linear(x, fc1_ws_ref[group_idx], bias=fc1_bs_ref[group_idx])
            fc2_in = _apply_activation(fc1_out) * probs[group_idx].unsqueeze(-1)
            y = torch.nn.functional.linear(fc2_in, fc2_ws_ref[group_idx])
            if bias:
                y = y + fc2_bs_ref[group_idx] * probs[group_idx].unsqueeze(-1)
            ys.append(y)
        y_ref = torch.cat(ys)
        y_ref.backward(dy_ref)

        # Construct TE module
        recipe = make_recipe(quantization)

        with te.quantized_model_init(enabled=with_quantization, recipe=recipe):
            fc1 = te.ops.GroupedLinear(
                group_size, hidden_size, fc1_out_features,
                bias=bias, device=device, dtype=dtype,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
                accumulate_into_main_grad=accumulate_into_main_grad,
                delay_wgrad_compute=delay_wgrad_compute,
            )
            fc2 = te.ops.GroupedLinear(
                group_size, hidden_size, hidden_size,
                bias=bias, device=device, dtype=dtype,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
                accumulate_into_main_grad=accumulate_into_main_grad,
                delay_wgrad_compute=delay_wgrad_compute,
                scale_bias=bias,
            )
        module = te.ops.Sequential(
            fc1,
            _make_scaled_grouped_mlp_activation(
                activation,
                glu_interleave_size=glu_interleave_size,
                geglu_limit=geglu_limit,
                geglu_alpha=geglu_alpha,
                geglu_offset=geglu_offset,
            ),
            fc2,
        )

        # Copy weights
        with torch.no_grad():
            _copy_grouped_linear_params(
                fc1,
                fc1_ws_test,
                fc1_bs_test if bias else None,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
            )
            _copy_grouped_linear_params(
                fc2,
                fc2_ws_test,
                fc2_bs_test if bias else None,
                single_grouped_weight=single_grouped_weight,
                single_grouped_bias=single_grouped_bias,
            )
            if accumulate_into_main_grad:
                main_grad_sentinel = 0.5
                weight_params_for_main_grad = (
                    _grouped_weight_params(
                        fc1,
                        group_size,
                        single_grouped_weight=single_grouped_weight,
                    )
                    + _grouped_weight_params(
                        fc2,
                        group_size,
                        single_grouped_weight=single_grouped_weight,
                    )
                )
                MegatronTrainingHelper.init_main_grad_buffers(
                    weight_params_for_main_grad,
                    fill_value=main_grad_sentinel,
                    overwrite_main_grad=False,
                )
        del fc1_ws_test, fc1_bs_test, fc2_ws_test, fc2_bs_test

        # Forward and backward pass
        with te.autocast(enabled=with_quantization, recipe=recipe):
            fc2_extra = (split_sizes, probs_test) if bias else (split_sizes,)
            y_test = module(x_test, split_sizes, probs_test, *fc2_extra)
        y_test.backward(dy_test)
        if delay_wgrad_compute:
            fc1.backward_dw()
            fc2.backward_dw()

        # Determine whether op fusion is expected
        is_fusion_expected = False
        if quantization == "mxfp8":
            is_fusion_expected = (
                dtype in (torch.bfloat16, torch.float16)
                and (
                    (not activation_is_glu and glu_interleave_size is None)
                    or (activation_is_glu and glu_interleave_size == 32)
                )
            )
        if quantization == "nvfp4_rht":
            is_fusion_expected = (
                dtype == torch.bfloat16
                and activation == "scaled_srelu"
                and glu_interleave_size is None
            )
        if is_fusion_expected:
            is_fusion_expected = (
                _cudnn_frontend_supports_grouped_gemm_srelu()
                if activation == "scaled_srelu"
                else _cudnn_frontend_version_supported()
            )

        # Check that fusion is applied if expected
        if is_fusion_expected:
            if activation_is_glu:
                forward_cls = te.ops.fused.ForwardGroupedMLP_CuTeGEMMGLU
                backward_cls = te.ops.fused.BackwardGroupedMLP_CuTeGEMMDGLU
            else:
                forward_cls = te.ops.fused.ForwardGroupedMLP_CuTeGEMMUnary
                backward_cls = te.ops.fused.BackwardGroupedMLP_CuTeGEMMDUnary
            if forward_cls.is_supported():
                forward_ops = module._module_groups[0]._forward_ops
                assert len(forward_ops) == 1
                assert isinstance(forward_ops[0][0], forward_cls)
            if backward_cls is not None and backward_cls.is_supported():
                backward_ops = module._module_groups[0]._backward_ops
                assert len(backward_ops) == 1
                assert isinstance(backward_ops[0][0], backward_cls)

        # Loose tols for sanity checking
        tols = {"rtol": 0.125, "atol": 0.25}
        if quantization in ("nvfp4", "nvfp4_row_scaled", "nvfp4_4over6", "nvfp4_rht"):
            tols = {"rtol": 0.25, "atol": 0.5}

        # Check values
        assert_close(y_test, y_ref, **tols)
        assert_close_grads(x_test, x_ref, **tols)
        assert_close_grads(probs_test, probs_ref, **tols)
        for group_idx in range(group_size):
            if bias:
                if single_grouped_bias:
                    assert_close(fc2.bias.grad[group_idx], fc2_bs_ref[group_idx].grad, **tols)
                    assert_close(fc1.bias.grad[group_idx], fc1_bs_ref[group_idx].grad, **tols)
                else:
                    assert_close_grads(getattr(fc2, f"bias{group_idx}"), fc2_bs_ref[group_idx], **tols)
                    assert_close_grads(getattr(fc1, f"bias{group_idx}"), fc1_bs_ref[group_idx], **tols)
            if not single_grouped_weight and not accumulate_into_main_grad:
                assert_close_grads(getattr(fc2, f"weight{group_idx}"), fc2_ws_ref[group_idx], **tols)
                assert_close_grads(getattr(fc1, f"weight{group_idx}"), fc1_ws_ref[group_idx], **tols)
        fc1_w_ref_grad = torch.stack([w.grad for w in fc1_ws_ref], dim=0)
        fc2_w_ref_grad = torch.stack([w.grad for w in fc2_ws_ref], dim=0)
        if accumulate_into_main_grad:
            fc1_expected = (
                [fc1_w_ref_grad + main_grad_sentinel]
                if single_grouped_weight
                else [g + main_grad_sentinel for g in fc1_w_ref_grad]
            )
            fc2_expected = (
                [fc2_w_ref_grad + main_grad_sentinel]
                if single_grouped_weight
                else [g + main_grad_sentinel for g in fc2_w_ref_grad]
            )
            MegatronTrainingHelper.verify_main_grad_accumulation(
                weight_params_for_main_grad,
                expected_main_grads=fc1_expected + fc2_expected,
                **tols,
            )
        elif single_grouped_weight:
            assert_close(fc1.weight.grad, fc1_w_ref_grad, **tols)
            assert_close(fc2.weight.grad, fc2_w_ref_grad, **tols)

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("quantization", ("mxfp8", "nvfp4_rht"))
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("accumulate_into_main_grad", (False, True))
    @pytest.mark.parametrize("delay_wgrad_compute", (False, True))
    @pytest.mark.parametrize("activation", ("scaled_swiglu", "scaled_srelu"))
    def test_grouped_mlp_mcore_integrations(
        self,
        *,
        bias: bool,
        quantization: Optional[str],
        single_grouped_weight: bool,
        accumulate_into_main_grad: bool,
        delay_wgrad_compute: bool,
        activation: str,
    ) -> None:
        """Grouped MLP with advanced Mcore integrations"""
        if not (accumulate_into_main_grad or delay_wgrad_compute):
            pytest.skip("Repeated test case in test_grouped_mlp")
        self.test_grouped_mlp(
            bias=bias,
            quantization=quantization,
            single_grouped_weight=single_grouped_weight,
            accumulate_into_main_grad=accumulate_into_main_grad,
            delay_wgrad_compute=delay_wgrad_compute,
            activation=activation,
        )

    @pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("activation", ("scaled_swiglu", "scaled_clamped_qgeglu"))
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_grouped_mlp_single_weight_numerics(
        self,
        *,
        dtype: torch.dtype,
        bias: bool,
        activation: str,
        device: torch.device = "cuda",
        group_size: int = 4,
        hidden_size: int = 256,
        split_alignment: int = 256,
        glu_interleave_size: int = 32,
    ) -> None:
        """single_grouped_weight=True/False should match exactly for fused MXFP8 grouped MLP."""

        if not te.ops.fused.ForwardGroupedMLP_CuTeGEMMGLU.is_supported():
            pytest.skip("MXFP8 fused grouped MLP forward is not supported on this system")
        if not te.ops.fused.BackwardGroupedMLP_CuTeGEMMDGLU.is_supported():
            pytest.skip("MXFP8 fused grouped MLP backward is not supported on this system")

        split_sizes = _make_grouped_split_sizes(
            group_size,
            split_alignment,
            start=1,
            dtype=torch.int64,
            device=device,
        )
        in_shape = (split_sizes.sum().item(), hidden_size)
        recipe = make_recipe("mxfp8")

        x_base = torch.empty(in_shape, device=device, dtype=dtype).uniform_(-0.25, 0.25)
        probs_base = torch.empty((in_shape[0],), device=device, dtype=dtype).uniform_(-0.25, 0.25)
        dy_base = torch.empty(in_shape, device=device, dtype=dtype).uniform_(-0.25, 0.25)
        fc1_ws_base = [
            torch.empty((2 * hidden_size, hidden_size), device=device, dtype=dtype).uniform_(
                -0.25, 0.25
            )
            for _ in range(group_size)
        ]
        fc2_ws_base = [
            torch.empty((hidden_size, hidden_size), device=device, dtype=dtype).uniform_(
                -0.25, 0.25
            )
            for _ in range(group_size)
        ]
        fc1_bs_base = (
            [
                torch.empty((2 * hidden_size,), device=device, dtype=dtype).uniform_(-0.5, 0.5)
                for _ in range(group_size)
            ]
            if bias
            else None
        )
        fc2_bs_base = (
            [
                torch.empty((hidden_size,), device=device, dtype=dtype).uniform_(-0.5, 0.5)
                for _ in range(group_size)
            ]
            if bias
            else None
        )

        def _run_case(single_grouped_weight: bool) -> tuple[torch.Tensor, ...]:
            with te.quantized_model_init(enabled=True, recipe=recipe):
                scaled_act = _make_scaled_grouped_mlp_activation(
                    activation,
                    glu_interleave_size=glu_interleave_size,
                )
                fc1 = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    2 * hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                )
                fc2 = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                    scale_bias=bias,
                )
                module = te.ops.Sequential(fc1, scaled_act, fc2)

            with torch.no_grad():
                _copy_grouped_linear_params(
                    fc1,
                    fc1_ws_base,
                    fc1_bs_base if bias else None,
                    single_grouped_weight=single_grouped_weight,
                )
                _copy_grouped_linear_params(
                    fc2,
                    fc2_ws_base,
                    fc2_bs_base if bias else None,
                    single_grouped_weight=single_grouped_weight,
                )

            x = x_base.detach().clone().requires_grad_(True)
            probs = probs_base.detach().clone().requires_grad_(True)
            dy = dy_base.detach().clone()

            with te.autocast(enabled=True, recipe=recipe):
                fc2_extra = (split_sizes, probs) if bias else (split_sizes,)
                y = module(x, split_sizes, probs, *fc2_extra)
            y.backward(dy)

            forward_ops = module._module_groups[0]._forward_ops
            backward_ops = module._module_groups[0]._backward_ops
            assert len(forward_ops) == 1
            assert isinstance(
                forward_ops[0][0],
                te.ops.fused.ForwardGroupedMLP_CuTeGEMMGLU,
            )
            assert len(backward_ops) == 1
            assert isinstance(
                backward_ops[0][0],
                te.ops.fused.BackwardGroupedMLP_CuTeGEMMDGLU,
            )

            fc1_dw = _stack_cloned_attr(
                _grouped_weight_params(
                    fc1,
                    group_size,
                    single_grouped_weight=single_grouped_weight,
                ),
                "grad",
            )
            fc2_dw = _stack_cloned_attr(
                _grouped_weight_params(
                    fc2,
                    group_size,
                    single_grouped_weight=single_grouped_weight,
                ),
                "grad",
            )

            fc1_db = None
            fc2_db = None
            if bias:
                fc1_db = _stack_cloned_attr(
                    _grouped_bias_params(
                        fc1,
                        group_size,
                        single_grouped_bias=False,
                    ),
                    "grad",
                )
                fc2_db = _stack_cloned_attr(
                    _grouped_bias_params(
                        fc2,
                        group_size,
                        single_grouped_bias=False,
                    ),
                    "grad",
                )

            return (
                y.detach().clone(),
                x.grad.detach().clone(),
                probs.grad.detach().clone(),
                fc1_dw,
                fc2_dw,
                fc1_db,
                fc2_db,
            )

        (
            y_false,
            dx_false,
            dprobs_false,
            fc1_dw_false,
            fc2_dw_false,
            fc1_db_false,
            fc2_db_false,
        ) = _run_case(False)
        (
            y_true,
            dx_true,
            dprobs_true,
            fc1_dw_true,
            fc2_dw_true,
            fc1_db_true,
            fc2_db_true,
        ) = _run_case(True)

        torch.testing.assert_close(y_false, y_true, rtol=0, atol=0)
        torch.testing.assert_close(dx_false, dx_true, rtol=0, atol=0)
        torch.testing.assert_close(dprobs_false, dprobs_true, rtol=0, atol=0)
        torch.testing.assert_close(fc1_dw_false, fc1_dw_true, rtol=0, atol=0)
        torch.testing.assert_close(fc2_dw_false, fc2_dw_true, rtol=0, atol=0)
        if bias:
            bias_tols = {"rtol": 0.05, "atol": 0.015625}
            torch.testing.assert_close(fc1_db_false, fc1_db_true, **bias_tols)
            torch.testing.assert_close(fc2_db_false, fc2_db_true, **bias_tols)


    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("delay_wgrad_compute", (False, True))
    @pytest.mark.parametrize("zero_out_wgrad", (False, True))
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_grouped_mlp_overwrite_main_grad(
        self,
        *,
        single_grouped_weight: bool,
        delay_wgrad_compute: bool,
        zero_out_wgrad: bool,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda",
        group_size: int = 4,
        hidden_size: int = 256,
        split_alignment: int = 256,
        glu_interleave_size: int = 32,
    ) -> None:
        """End-to-end check that the fused grouped-MLP backward writes the
        wgrad into ``weight.main_grad`` correctly under the MegatronFSDP
        ``overwrite_main_grad=True`` convention.
        ``test_grouped_mlp`` already covers the standard Megatron-LM
        ``fuse_wgrad_accumulation`` (DDP) path where the wgrad GEMM
        *accumulates* into ``main_grad``. This test focuses exclusively on
        the MegatronFSDP variant where the wgrad GEMM must *overwrite*
        ``main_grad`` (because FSDP has already ReduceScattered the previous
        accumulation), so ``main_grad`` after backward equals ``wgrad``
        regardless of the prior contents.

        Also exercises the MegatronFSDP ``zero_out_wgrad`` flag, which is
        independent of ``main_grad`` and only controls whether the dummy
        ``param.grad`` returned to autograd is zeroed (so downstream hooks
        that read ``.grad`` don't see stale bytes from the cached dummy).
        """

        if not te.ops.fused.ForwardGroupedMLP_CuTeGEMMGLU.is_supported():
            pytest.skip("MXFP8 fused grouped MLP forward is not supported on this system")
        if not te.ops.fused.BackwardGroupedMLP_CuTeGEMMDGLU.is_supported():
            pytest.skip("MXFP8 fused grouped MLP backward is not supported on this system")

        recipe = make_recipe("mxfp8")
        split_sizes = _make_grouped_split_sizes(
            group_size,
            split_alignment,
            start=1,
            dtype=torch.int64,
            device=device,
        )
        in_shape = (split_sizes.sum().item(), hidden_size)
        x_base = torch.empty(in_shape, device=device, dtype=dtype).uniform_(-0.25, 0.25)
        probs_base = torch.empty((in_shape[0],), device=device, dtype=dtype).uniform_(-0.25, 0.25)
        dy_base = torch.empty(in_shape, device=device, dtype=dtype).uniform_(-0.25, 0.25)
        fc1_ws_base = [
            torch.empty((2 * hidden_size, hidden_size), device=device, dtype=dtype).uniform_(
                -0.25, 0.25
            )
            for _ in range(group_size)
        ]
        fc2_ws_base = [
            torch.empty((hidden_size, hidden_size), device=device, dtype=dtype).uniform_(
                -0.25, 0.25
            )
            for _ in range(group_size)
        ]

        def _build_module(*, accumulate_into_main_grad: bool):
            with te.quantized_model_init(enabled=True, recipe=recipe):
                fc1 = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    2 * hidden_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                    accumulate_into_main_grad=accumulate_into_main_grad,
                    delay_wgrad_compute=delay_wgrad_compute,
                )
                fc2 = te.ops.GroupedLinear(
                    group_size,
                    hidden_size,
                    hidden_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                    single_grouped_weight=single_grouped_weight,
                    accumulate_into_main_grad=accumulate_into_main_grad,
                    delay_wgrad_compute=delay_wgrad_compute,
                )
                scaled_act = te.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
                module = te.ops.Sequential(fc1, scaled_act, fc2)

            with torch.no_grad():
                _copy_grouped_linear_params(
                    fc1,
                    fc1_ws_base,
                    single_grouped_weight=single_grouped_weight,
                )
                _copy_grouped_linear_params(
                    fc2,
                    fc2_ws_base,
                    single_grouped_weight=single_grouped_weight,
                )
            return module, fc1, fc2

        def _weight_params(fc):
            return _grouped_weight_params(
                fc,
                group_size,
                single_grouped_weight=single_grouped_weight,
            )

        def _run_backward(module, fc1, fc2):
            x = x_base.detach().clone().requires_grad_(True)
            probs = probs_base.detach().clone().requires_grad_(True)
            with te.autocast(enabled=True, recipe=recipe):
                y = module(x, split_sizes, probs, split_sizes)
            y.backward(dy_base)
            if delay_wgrad_compute:
                fc1.backward_dw()
                fc2.backward_dw()

        # Reference run: vanilla autograd, no Megatron protocol.
        ref_module, ref_fc1, ref_fc2 = _build_module(accumulate_into_main_grad=False)
        _run_backward(ref_module, ref_fc1, ref_fc2)
        ref_fc1_grads = _clone_grads(_weight_params(ref_fc1))
        ref_fc2_grads = _clone_grads(_weight_params(ref_fc2))

        # Test run: main_grad fusion with overwrite_main_grad=True (MegatronFSDP).
        # NaN sentinel makes a missed write loud (would surface as NaN diff).
        test_module, test_fc1, test_fc2 = _build_module(accumulate_into_main_grad=True)
        for fc in (test_fc1, test_fc2):
            MegatronTrainingHelper.init_main_grad_buffers(
                _weight_params(fc),
                fill_value=float("nan"),
                overwrite_main_grad=True,
                zero_out_wgrad=zero_out_wgrad,
            )
        _run_backward(test_module, test_fc1, test_fc2)

        # main_grad must be overwritten to exactly the ref wgrad (bitwise:
        # the wgrad GEMM is deterministic across the two runs because the
        # quantized weights and inputs are identical).
        MegatronTrainingHelper.verify_main_grad_accumulation(
            _weight_params(test_fc1), expected_main_grads=ref_fc1_grads
        )
        MegatronTrainingHelper.verify_main_grad_accumulation(
            _weight_params(test_fc2), expected_main_grads=ref_fc2_grads
        )


    @pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
    @pytest.mark.parametrize("single_grouped_weight", (False, True))
    @pytest.mark.parametrize("accumulate_into_main_grad", (False, True))
    @pytest.mark.parametrize("activation", ("scaled_swiglu", "scaled_clamped_qgeglu"))
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_grouped_mlp_cuda_graph_safe_mxfp8(
        self,
        *,
        dtype: torch.dtype,
        single_grouped_weight: bool,
        accumulate_into_main_grad: bool,
        activation: str,
        device: torch.device = "cuda",
        group_size: int = 4,
        hidden_size: int = 256,
        split_alignment: int = 256,
        glu_interleave_size: int = 32,
        token_padding: int = 2048,
    ) -> None:
        """Grouped MLP forward+backward should be CUDA graph capturable (MXFP8)."""

        if not te.ops.fused.ForwardGroupedMLP_CuTeGEMMGLU.is_supported():
            pytest.skip("MXFP8 fused grouped MLP is not supported on this system")
        if dtype not in (torch.bfloat16, torch.float16):
            pytest.skip("MXFP8 fused grouped MLP is only supported with BF16/FP16")

        split_sizes = _make_grouped_split_sizes(
            group_size,
            split_alignment,
            start=1,
            dtype=torch.int64,
            device=device,
        )
        # Pad the input tokens to validate the sync-free MOE
        in_shape = (split_sizes.sum().item() + token_padding, hidden_size)
        recipe = make_recipe("mxfp8")
        with te.quantized_model_init(enabled=True, recipe=recipe):
            fc1 = te.ops.GroupedLinear(
                group_size,
                hidden_size,
                2 * hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
                single_grouped_weight=single_grouped_weight,
                accumulate_into_main_grad=accumulate_into_main_grad,
            )
            fc2 = te.ops.GroupedLinear(
                group_size,
                hidden_size,
                hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
                single_grouped_weight=single_grouped_weight,
                accumulate_into_main_grad=accumulate_into_main_grad,
            )
            scaled_act = _make_scaled_grouped_mlp_activation(
                activation,
                glu_interleave_size=glu_interleave_size,
            )
            module = te.ops.Sequential(
                fc1,
                scaled_act,
                fc2,
            )
        fc1_weight_params = _grouped_weight_params(
            fc1,
            group_size,
            single_grouped_weight=single_grouped_weight,
        )
        fc2_weight_params = _grouped_weight_params(
            fc2,
            group_size,
            single_grouped_weight=single_grouped_weight,
        )

        def _init_main_grads(value: float = 0.0) -> None:
            if not accumulate_into_main_grad:
                return
            _fill_main_grads(fc1_weight_params + fc2_weight_params, value, device=device)

        def _collect_main_grads() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                _stack_cloned_attr(fc1_weight_params, "main_grad"),
                _stack_cloned_attr(fc2_weight_params, "main_grad"),
            )

        static_split_sizes = split_sizes.clone()

        def train_step(
            x: torch.Tensor,
            probs: torch.Tensor,
            dy: torch.Tensor,
            out_buf: torch.Tensor,
            *,
            use_graphed: bool,
        ) -> torch.Tensor:
            with te.autocast(enabled=True, recipe=recipe):
                out = (
                    graphed_module(x, static_split_sizes, probs, static_split_sizes)
                    if use_graphed
                    else module(x, static_split_sizes, probs, static_split_sizes)
                )
            out.backward(dy)
            out_buf.copy_(out)
            return out_buf

        _init_main_grads(0.0)

        static_x = torch.randn(in_shape, device=device, dtype=dtype, requires_grad=True)
        static_probs = torch.randn((in_shape[0],), device=device, dtype=dtype, requires_grad=True)
        static_dy = torch.randn(in_shape, device=device, dtype=dtype)
        static_out_buf = torch.empty((in_shape[0], hidden_size), device=device, dtype=dtype)

        graphed_module = te.make_graphed_callables(
            module,
            (static_x, static_split_sizes, static_probs, static_split_sizes),
            num_warmup_iters=3,
            enabled=True,
            recipe=recipe,
        )

        forward_ops = module._module_groups[0]._forward_ops
        backward_ops = module._module_groups[0]._backward_ops
        assert len(forward_ops) == 1
        assert isinstance(
            forward_ops[0][0],
            te.ops.fused.ForwardGroupedMLP_CuTeGEMMGLU,
        )
        assert len(backward_ops) == 1
        assert isinstance(
            backward_ops[0][0],
            te.ops.fused.BackwardGroupedMLP_CuTeGEMMDGLU,
        )

        fresh_x = torch.randn_like(static_x)
        fresh_probs = torch.randn_like(static_probs)
        fresh_dy = torch.randn_like(static_dy)
        with torch.no_grad():
            static_x.copy_(fresh_x)
            static_probs.copy_(fresh_probs)
            static_dy.copy_(fresh_dy)

        for param in module.parameters():
            param.grad = torch.zeros_like(param)
        _init_main_grads(0.5)
        if static_x.grad is not None:
            static_x.grad.zero_()
        if static_probs.grad is not None:
            static_probs.grad.zero_()

        graph_out = (
            train_step(static_x, static_probs, static_dy, static_out_buf, use_graphed=True)
            .detach()
            .clone()
        )
        torch.cuda.synchronize()
        graph_dx = static_x.grad.detach().clone()
        graph_dprobs = static_probs.grad.detach().clone()
        if accumulate_into_main_grad:
            graph_fc1_main_grad, graph_fc2_main_grad = _collect_main_grads()
        else:
            graph_param_grads = [param.grad.detach().clone() for param in module.parameters()]

        for param in module.parameters():
            param.grad.zero_()
        _init_main_grads(0.5)
        static_x.grad.zero_()
        static_probs.grad.zero_()

        expected_x = fresh_x.detach().clone().requires_grad_(True)
        expected_probs = fresh_probs.detach().clone().requires_grad_(True)
        expected_dy = fresh_dy.detach().clone()
        with te.autocast(enabled=True, recipe=recipe):
            expected_out = module(
                expected_x,
                static_split_sizes,
                expected_probs,
                static_split_sizes,
            )
        expected_out.backward(expected_dy)

        tols = dtype_tols(dtype)
        assert_close(graph_out, expected_out, **tols)
        assert_close(graph_dx, expected_x.grad, **tols)
        assert_close(graph_dprobs, expected_probs.grad, **tols)
        if accumulate_into_main_grad:
            expected_fc1_main_grad, expected_fc2_main_grad = _collect_main_grads()
            assert_close(graph_fc1_main_grad, expected_fc1_main_grad, **tols)
            assert_close(graph_fc2_main_grad, expected_fc2_main_grad, **tols)
        else:
            for graph_grad, param in zip(graph_param_grads, module.parameters()):
                assert_close(graph_grad, param.grad, **tols)


    def test_grouped_gemm_quant_cute_matches_mxfp8_quantized(self) -> None:
        if not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
        if torch.cuda.get_device_capability() < (10, 0):
            pytest.skip("Requires SM100+ for grouped GEMM quant kernel.")

        try:
            from cudnn import grouped_gemm_quant_wrapper_sm100  # pylint: disable=no-name-in-module
        except ImportError as exc:
            pytest.skip(f"grouped_gemm_quant_wrapper_sm100 unavailable: {exc}")

        device = torch.device("cuda")
        dtype = torch.bfloat16 if is_bf16_available() else torch.float16
        num_groups = 4
        m = 256
        n = 512
        k = 512
        total_m = num_groups * m
        split_sizes = torch.full((num_groups,), m, device=device, dtype=torch.int64)

        q = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=False)
        q.optimize_for_gemm = False

        torch.manual_seed(0)
        a_full = torch.randn(total_m, k, device=device, dtype=dtype)
        weights = [torch.randn(n, k, device=device, dtype=dtype) for _ in range(num_groups)]

        grouped_a = tex.group_quantize(a_full, q, num_groups, split_sizes)
        a_groups = grouped_a.split_into_quantized_tensors()
        b_groups = [q(w) for w in weights]

        # Reference GEMM on dequantized tensors.
        ref = torch.empty((total_m, n), device=device, dtype=torch.float32)
        start = 0
        for group_idx in range(num_groups):
            end = start + m
            a_deq = a_groups[group_idx].dequantize(dtype=torch.float32)
            b_deq = b_groups[group_idx].dequantize(dtype=torch.float32)
            ref[start:end, :] = a_deq @ b_deq.t()
            start = end
        ref = ref.to(dtype=torch.bfloat16).to(torch.float32)

        # Allocate empty input tensors needed for cuTE DSL kernel
        padded_offsets = torch.tensor(
            [m * (i + 1) for i in range(num_groups)],
            dtype=torch.int32,
            device=device,
        )
        inputs = {
            "a_tensor": torch.empty(1, total_m, k, dtype=torch.float8_e4m3fn, device=device).permute(
                1, 2, 0
            ),
            "b_tensor": torch.empty(num_groups, n, k, dtype=torch.float8_e4m3fn, device=device).permute(
                1, 2, 0
            ),
            "sfa_tensor": torch.empty(
                1,
                total_m // 128,
                k // 128,
                32,
                4,
                4,
                dtype=torch.float8_e8m0fnu,
                device=device,
            ).permute(3, 4, 1, 5, 2, 0),
            "sfb_tensor": torch.empty(
                num_groups,
                n // 128,
                k // 128,
                32,
                4,
                4,
                dtype=torch.float8_e8m0fnu,
                device=device,
            ).permute(3, 4, 1, 5, 2, 0),
            "alpha_tensor": torch.empty(num_groups, dtype=torch.float32, device=device),
            "prob_tensor": torch.empty(total_m, 1, 1, dtype=torch.float32, device=device),
            "padded_offsets_tensor": padded_offsets,
        }
        # Overwrite inputs with quantized data/scales from MXFP8 quantizer.
        a_data = grouped_a.rowwise_data.view(total_m, k).view(dtype=torch.float8_e4m3fn)
        a_data = a_data.unsqueeze(0).permute(1, 2, 0).contiguous()
        inputs["a_tensor"].copy_(a_data)

        a_scales = grouped_a.scale_inv.view(dtype=torch.float8_e8m0fnu)
        a_scales = a_scales.view(1, total_m // 128, 4, 32, k // 128, 4)
        a_scales = a_scales.permute(0, 1, 4, 3, 2, 5).contiguous()
        a_scales = a_scales.permute(3, 4, 1, 5, 2, 0).contiguous()
        inputs["sfa_tensor"].copy_(a_scales)

        b_data = torch.cat([w._rowwise_data.reshape(-1) for w in b_groups])
        b_data = b_data.view(dtype=torch.float8_e4m3fn)
        b_data = b_data.view(num_groups, n, k).permute(1, 2, 0).contiguous()
        inputs["b_tensor"].copy_(b_data)

        b_scales = torch.cat([w._rowwise_scale_inv for w in b_groups])
        b_scales = b_scales.view(dtype=torch.float8_e8m0fnu)
        b_scales = b_scales.view(num_groups, n // 128, 4, 32, k // 128, 4)
        b_scales = b_scales.permute(0, 1, 4, 3, 2, 5).contiguous()
        b_scales = b_scales.permute(3, 4, 1, 5, 2, 0).contiguous()
        inputs["sfb_tensor"].copy_(b_scales)

        inputs["alpha_tensor"].fill_(1.0)
        inputs["prob_tensor"].fill_(1.0)

        cute_out = grouped_gemm_quant_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            b_tensor=inputs["b_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            norm_const_tensor=None,
            prob_tensor=inputs["prob_tensor"],
            acc_dtype=torch.float32,
            d_dtype=torch.bfloat16,
            cd_major="n",
            sf_vec_size=32,
            discrete_col_sfd=True,
            current_stream=None,
        )

        if isinstance(cute_out, dict):
            outputs = cute_out
        else:
            d_tensor, d_col_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor = cute_out
            outputs = {
                "d_tensor": d_tensor,
                "d_col_tensor": d_col_tensor,
                "amax_tensor": amax_tensor,
                "sfd_row_tensor": sfd_row_tensor,
                "sfd_col_tensor": sfd_col_tensor,
            }

        d_cute = outputs["d_tensor"]
        if d_cute.dim() == 3:
            d_cute = d_cute.squeeze(-1)
        tols = dtype_tols(torch.bfloat16)
        assert_close(d_cute[:total_m].float(), ref, **tols)
