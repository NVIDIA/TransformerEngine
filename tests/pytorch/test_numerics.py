# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
import os
from typing import Dict, List, Tuple, Optional
import pytest

import torch
import torch.nn as nn
from torch.nn import Parameter

from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
)
from transformer_engine.pytorch._extra_state import UNSAFE_PICKLE_EXTRA_STATE_ENV
from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
    attention_mask_func,
)
from transformer_engine.pytorch import (
    autocast,
    quantized_model_init,
    DotProductAttention,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    MultiheadAttention,
    RMSNorm,
    TransformerLayer,
    LayerNorm,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
    get_device_compute_capability,
    is_fp8_available,
    is_mxfp8_available,
    is_fp8_block_scaling_available,
    is_bf16_available,
    is_nvfp4_available,
)
from transformer_engine.pytorch import checkpoint as te_checkpoint
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.common import recipe
from transformer_engine.pytorch import DType
import transformer_engine_torch as tex
from utils import ModelConfig, recipe_id, reset_rng_states, skip_unsupported_backward_override

# Only run FP8 tests on supported devices.
fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
fp8_block_scaling_available = is_fp8_block_scaling_available()
nvfp4_available = is_nvfp4_available()

sm_80plus = get_device_compute_capability() >= (8, 0)

seed = 1234
# Reset RNG states.
reset_rng_states()

torch._dynamo.config.recompile_limit = 16


model_configs = {
    "small": ModelConfig(1, 128, 8, 16, num_layers=4),
    "126m": ModelConfig(1, 2048, 12, 64, num_layers=12),
}
model_configs_inference = {
    "126m": ModelConfig(1, 256, 12, 64, num_layers=12),
}
backends_inference = ["FlashAttention", "UnfusedAttention", "FusedAttention"]
module_inference = ["TransformerLayer", "MultiheadAttention"]
input_formats_inference = ["sbhd", "bshd"]

param_types = [torch.float32, torch.float16]
if is_bf16_available():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

batch_sizes = [1, 2]

all_boolean = [True, False]

all_activations = [
    "gelu",
    "geglu",
    "glu",
    "qgelu",
    "qgeglu",
    "relu",
    "reglu",
    "srelu",
    "sreglu",
    "silu",
    "swiglu",
]

all_normalizations = ["LayerNorm", "RMSNorm"]

mask_types = ["causal", "no_mask"]

NVTE_TEST_NVINSPECT_ENABLED = int(os.environ.get("NVTE_TEST_NVINSPECT_ENABLED", "0"))

if NVTE_TEST_NVINSPECT_ENABLED:
    # The numerics of all the layers should work the same,
    # when debug=True. I fed them with dummy feature
    # to prevent switching off debug, which can happen if
    # no feature is active.
    import nvdlfw_inspect.api as debug_api

    debug_api.initialize(
        os.environ["NVTE_TEST_NVINSPECT_CONFIG_FILE"],
        feature_dirs=os.environ["NVTE_TEST_NVINSPECT_FEATURE_DIRS"],
    )


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
    # if using RHT, we can only support bf16
    # check fp4_quant_fwd_inp, fp4_quant_fwd_weight, fp4_quant_bwd_grad
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
        # if not using RHT, we can add fp32 as well
    if not check_rht_usage(recipe):
        supported_input_dtypes.append(torch.float32)
    return supported_input_dtypes


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


def get_causal_attn_mask(sq: int) -> torch.Tensor:
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()


def dtype_tols(dtype: torch.dtype) -> Dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    raise ValueError(f"Unsupported dtype ({dtype})")


def assert_allclose(
    l1: List[torch.Tensor], l2: List[torch.Tensor], atol: float = None, rtol: float = None
) -> bool:
    """Ensures two lists are equal."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    for i, (t1, t2) in enumerate(zip(l1, l2)):
        tols = dtype_tols(t2.dtype)
        if rtol is not None:
            tols["rtol"] = rtol
        if atol is not None:
            tols["atol"] = atol
        result = torch.allclose(t1, t2, **tols)
        if not result:
            diff = torch.abs(t1 - t2)
            tol = tols["atol"] + (tols["rtol"] * torch.abs(t2))
            exceed_mask = diff > tol
            if exceed_mask.any():
                indices = torch.nonzero(exceed_mask, as_tuple=True)
                max_diff = diff[exceed_mask].max()
                max_idx = (diff[exceed_mask] == max_diff).nonzero(as_tuple=True)[0][0]
                max_location = [idx[max_idx].item() for idx in indices]
                msg = (
                    f"Outputs not close enough in tensor at idx={i}. "
                    f"Maximum difference at location {max_location} "
                    f"with {t1[exceed_mask][max_idx].item()} vs {t2[exceed_mask][max_idx].item()} "
                    f"(diff {max_diff.item()})."
                )
            raise AssertionError(msg)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


class TorchScaledMaskedSoftmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, inp: torch.Tensor, mask: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        dtype = inp.dtype
        inp = inp.float()

        if scale is not None:
            inp = inp * scale
        mask_output = attention_mask_func(inp, mask) if mask is not None else inp

        probs = torch.nn.Softmax(dim=-1)(mask_output)
        probs = probs.to(dtype)
        return probs


class TorchDotProductAttention(torch.nn.Module):
    def __init__(
        self,
        kv_channels: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm_factor = math.sqrt(kv_channels)
        self.scale_mask_softmax = TorchScaledMaskedSoftmax()
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        attention_probs = self.attention_dropout(attention_probs)

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context_layer = context_layer.view(seqlen, batch_size, -1)

        return context_layer


class TorchLayerNorm(nn.Module):
    def __init__(self, in_features: int, eps: float, zero_centered_gamma: bool):
        super().__init__()
        self.eps = eps
        self.in_features = in_features
        self.zero_centered_gamma = zero_centered_gamma

        initial_value = torch.ones(in_features) if zero_centered_gamma else torch.zeros(in_features)
        self.weight = nn.Parameter(initial_value)
        self.bias = nn.Parameter(torch.zeros(in_features))
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight if not self.zero_centered_gamma else 1 + self.weight
        w = w.to(torch.float32)
        b = self.bias.to(torch.float32)
        inp = x.to(torch.float32)
        out = torch.nn.functional.layer_norm(
            inp, (self.in_features,), weight=w, bias=b, eps=self.eps
        )
        return out.to(x.dtype)


# Adapted from https://github.com/bzhangGo/rmsnorm/blob/c6691f20ec0af4128c8159c903071f7575404295/rmsnorm_torch.py
class TorchRMSNorm(nn.Module):
    def __init__(self, in_features, zero_centered_gamma, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.in_features = in_features
        self.zero_centered_gamma = zero_centered_gamma

        initial_value = torch.ones(in_features) if zero_centered_gamma else torch.zeros(in_features)
        self.weight = nn.Parameter(initial_value)
        self.register_parameter("weight", self.weight)

    def forward(self, x):
        norm_x2 = torch.sum(x.float() ** 2, dim=-1, keepdim=True)
        d_x = self.in_features

        rms_x2 = norm_x2 / d_x + self.eps
        r_rms_x = rms_x2 ** (-1.0 / 2)
        x_normed = x * r_rms_x

        w = self.weight.float()
        if self.zero_centered_gamma:
            w = 1 + w
        return (w * x_normed).to(x.dtype)


class TorchLayerNormLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float,
        normalization: str = "LayerNorm",
        zero_centered_gamma: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if normalization == "LayerNorm":
            self.layernorm = TorchLayerNorm(
                in_features, eps=eps, zero_centered_gamma=zero_centered_gamma
            )
        elif normalization == "RMSNorm":
            self.layernorm = TorchRMSNorm(
                in_features, eps=eps, zero_centered_gamma=zero_centered_gamma
            )
        else:
            raise RuntimeError("Unsupported normalization")

        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layernorm(x))


class TorchMHA(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=0.1,
            bias=True,
            batch_first=False,
        )

    def forward(self, x, attention_mask=None):
        output = self.mhsa(x, x, x, attn_mask=attention_mask, need_weights=False)
        if isinstance(output, tuple):
            output = output[0]
        return output


class TorchQuickGELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class TorchSquaredRELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input > 0) * input * input


_supported_act = {
    "gelu": nn.GELU(approximate="tanh"),
    "geglu": nn.GELU(approximate="tanh"),
    "glu": nn.Sigmoid(),
    "qgelu": TorchQuickGELU(),
    "qgeglu": TorchQuickGELU(),
    "relu": nn.ReLU(),
    "reglu": nn.ReLU(),
    "srelu": TorchSquaredRELU(),
    "sreglu": TorchSquaredRELU(),
    "silu": nn.SiLU(),
    "swiglu": nn.SiLU(),
}


class TorchGLU(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        self.act = _supported_act[activation]

    def forward(self, x):
        shape = x.size(-1)
        a = x[..., : shape // 2]
        b = x[..., (shape // 2) :]
        a = self.act(a)
        return a * b


class TorchLayerNormMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        activation="gelu",
        normalization: str = "LayerNorm",
        bias: bool = True,
    ):
        super().__init__()
        if normalization == "LayerNorm":
            self.ln = TorchLayerNorm(hidden_size, eps=eps, zero_centered_gamma=False)
        elif normalization == "RMSNorm":
            self.ln = TorchRMSNorm(hidden_size, eps=eps, zero_centered_gamma=False)
        else:
            raise RuntimeError("Unsupported normalization")
        if "glu" in activation:
            fc1_output_features = 2 * ffn_hidden_size
            self.gelu = TorchGLU(activation)
        else:
            fc1_output_features = ffn_hidden_size
            self.gelu = _supported_act[activation]

        self.fc1 = nn.Linear(hidden_size, fc1_output_features, bias=bias)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)

    def forward(self, x):
        t = self.gelu(self.fc1(self.ln(x)))
        return self.fc2(t)


class TorchGPT(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, num_attention_heads: int, parallel_attention_mlp: bool
    ):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps)
        self.causal_attn = TorchMHA(hidden_size, num_attention_heads)
        self.ln_mlp = TorchLayerNormMLP(hidden_size, 4 * hidden_size, eps)
        self.parallel_attention_mlp = parallel_attention_mlp

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = self.ln(x)
        b = self.causal_attn(a, attention_mask)
        if self.parallel_attention_mlp:
            n = self.ln_mlp(x)
            x = x + nn.functional.dropout(b + n, p=0.1, training=self.training)
        else:
            x = x + nn.functional.dropout(b, p=0.1, training=self.training)
            n = self.ln_mlp(x)
            x = x + nn.functional.dropout(n, p=0.1, training=self.training)
        return x


def _test_e2e_selective_recompute(
    bs, dtype, config, fp8, recipe, fp8_model_params=False, recompute=False
):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.max_seqlen_q)

    with autocast(enabled=fp8, recipe=recipe):
        te_out = block(
            te_inp_hidden_states,
            attention_mask=te_inp_attn_mask,
            checkpoint_core_attention=recompute,
        )
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

    outputs = [te_out, te_inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("recipe", fp8_recipes, ids=recipe_id)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
def test_gpt_selective_activation_recompute(dtype, bs, model, fp8, recipe, fp8_model_params):
    if fp8_model_params and NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("FP8 parameters are not supported in debug mode.")
    if fp8 or fp8_model_params:
        skip_unsupported_backward_override(
            "transformer_layer", recipe, getattr(recipe, "backward_override", None)
        )
    if fp8 and recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe {recipe.__class__.__name__}"
            )

    config = model_configs[model]

    outputs = _test_e2e_selective_recompute(
        bs, dtype, config, fp8, recipe, fp8_model_params, recompute=False
    )
    outputs_recompute = _test_e2e_selective_recompute(
        bs, dtype, config, fp8, recipe, fp8_model_params, recompute=True
    )

    # Check that results match
    tols = dtype_tols(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        tols["atol"] = 1e-4
    if fp8 or fp8_model_params:
        tols.update(dict(rtol=0.125, atol=0.0675))

    for i, (ref, test) in enumerate(zip(outputs, outputs_recompute)):
        torch.testing.assert_close(
            test,
            ref,
            msg=f"Mismatch in tensor {i}",
            **tols,
        )


def _test_e2e_full_recompute(
    bs, dtype, config, fp8, recipe, fp8_model_params=False, recompute=False, use_reentrant=True
):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=use_reentrant,
    )
    if use_reentrant:
        te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.max_seqlen_q)

    with autocast(enabled=fp8, recipe=recipe):
        if recompute:
            te_out = te_checkpoint(
                block,
                te_inp_hidden_states,
                attention_mask=te_inp_attn_mask,
                checkpoint_core_attention=False,
                distribute_saved_activations=False,
                tp_group=None,
                use_reentrant=use_reentrant,
            )
        else:
            te_out = block(
                te_inp_hidden_states,
                attention_mask=te_inp_attn_mask,
                checkpoint_core_attention=False,
            )
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

    outputs = [te_out]
    names = ["output"]
    if use_reentrant:
        outputs.append(te_inp_hidden_states.grad)
        names.append("input")
    for name, p in block.named_parameters():
        if p.requires_grad:
            outputs.append(p.grad)
            names.append(name)

    return outputs, names


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("recipe", fp8_recipes, ids=recipe_id)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
@pytest.mark.parametrize("use_reentrant", all_boolean)
def test_gpt_full_activation_recompute(
    dtype, bs, model, fp8, recipe, fp8_model_params, use_reentrant
):
    if fp8_model_params and NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("FP8 parameters are not supported in debug mode.")
    if fp8 or fp8_model_params:
        skip_unsupported_backward_override(
            "transformer_layer", recipe, getattr(recipe, "backward_override", None)
        )
    if fp8 and recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe {recipe.__class__.__name__}"
            )

    config = model_configs[model]

    if not use_reentrant:
        # Non-reentrant checkpoint becomes non-deterministic with bias+GELU fusion
        os.environ["NVTE_BIAS_GELU_NVFUSION"] = "0"

    outputs, names = _test_e2e_full_recompute(
        bs,
        dtype,
        config,
        fp8,
        recipe,
        fp8_model_params,
        recompute=False,
        use_reentrant=use_reentrant,
    )
    outputs_recompute, _ = _test_e2e_full_recompute(
        bs,
        dtype,
        config,
        fp8,
        recipe,
        fp8_model_params,
        recompute=True,
        use_reentrant=use_reentrant,
    )

    if not use_reentrant:
        # Reset bias+GELU fusion flag to avoid contaminating other tests
        del os.environ["NVTE_BIAS_GELU_NVFUSION"]

    # Check that results match
    tols = dtype_tols(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        tols["atol"] = 1e-3
    if fp8 or fp8_model_params:
        tols.update(dict(rtol=0.125, atol=0.0675))
    for i, (ref, test) in enumerate(zip(outputs, outputs_recompute)):
        torch.testing.assert_close(
            test,
            ref,
            msg=f"Mismatch in tensor {i}",
            **tols,
        )


def _test_e2e_checkpointing_get_model(config, dtype):
    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    return TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        params_dtype=dtype,
        device="cuda",
    )


def _test_e2e_checkpointing(bs, dtype, config, checkpoint=False, steps=10, path="checkpoint.pt"):
    reset_rng_states()

    te_inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()

    block = _test_e2e_checkpointing_get_model(config, dtype)

    for _ in range(steps // 2):
        te_out = block(
            te_inp_hidden_states,
            None,
        )
        loss = te_out.sum()
        loss.backward()

    if checkpoint:
        # This process is necessary so that we can start afresh with
        # a new model while erasing all internal state to ensure that
        # loading from a checkpoint gives bitwise identical results.
        # Since gradients are being accumulated, it is important to
        # restore them post loading the checkpoint.
        torch.save(block.state_dict(), path)

        param_grads = []
        for p in block.parameters():
            if p.requires_grad:
                param_grads.append(p.grad.clone())

        _cpu_rng_state = torch.get_rng_state()
        _cuda_rng_state = torch.cuda.get_rng_state()

        del block
        block = _test_e2e_checkpointing_get_model(config, dtype)
        loaded_state_dict = torch.load(path, weights_only=False)
        old_unsafe_extra_state = os.environ.get(UNSAFE_PICKLE_EXTRA_STATE_ENV)
        try:
            block.load_state_dict(loaded_state_dict)
        finally:
            if old_unsafe_extra_state is None:
                os.environ.pop(UNSAFE_PICKLE_EXTRA_STATE_ENV, None)
            else:
                os.environ[UNSAFE_PICKLE_EXTRA_STATE_ENV] = old_unsafe_extra_state
        torch.set_rng_state(_cpu_rng_state)
        torch.cuda.set_rng_state(_cuda_rng_state)

        for p in block.parameters():
            if p.requires_grad:
                p.grad = param_grads.pop(0)

        assert not param_grads, "Oops!"

    for _ in range(steps // 2):
        te_out = block(
            te_inp_hidden_states,
            None,
        )
        loss = te_out.sum()
        loss.backward()

    torch.cuda.synchronize()

    if os.path.exists(path):
        os.remove(path)

    outputs = [te_out, te_inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
def test_gpt_checkpointing(dtype, bs, model):
    config = model_configs[model]
    outputs = _test_e2e_checkpointing(bs, dtype, config, checkpoint=False)
    outputs_checkpoint = _test_e2e_checkpointing(bs, dtype, config, checkpoint=True)

    # Check that results match
    tols = dtype_tols(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        tols.update(dict(rtol=2e-2, atol=2e-3))
    for i, (ref, test) in enumerate(zip(outputs, outputs_checkpoint)):
        torch.testing.assert_close(
            test,
            ref,
            msg=f"Mismatch in tensor {i}",
            **tols,
        )


def _test_e2e_gpt_accuracy(block, bs, dtype, config):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()
    inp_attn_mask = get_causal_attn_mask(config.max_seqlen_q)

    out = block(inp_hidden_states, attention_mask=inp_attn_mask)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("parallel_attention_mlp", all_boolean)
def test_gpt_accuracy(dtype, bs, model, parallel_attention_mlp):
    config = model_configs[model]

    te_gpt = TransformerLayer(
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        num_attention_heads=config.num_heads,
        layernorm_epsilon=config.eps,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        params_dtype=dtype,
        fuse_qkv_params=True,
        qkv_weight_interleaved=False,
        parallel_attention_mlp=parallel_attention_mlp,
        device="cuda",
    ).eval()

    torch_gpt = (
        TorchGPT(
            config.hidden_size,
            config.eps,
            config.num_heads,
            parallel_attention_mlp=parallel_attention_mlp,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_gpt.ln.weight = Parameter(
            te_gpt.self_attention.layernorm_qkv.layer_norm_weight.clone()
        )
        torch_gpt.ln.bias = Parameter(te_gpt.self_attention.layernorm_qkv.layer_norm_bias.clone())
        torch_gpt.causal_attn.mhsa.in_proj_weight = Parameter(
            te_gpt.self_attention.layernorm_qkv.weight.clone()
        )
        torch_gpt.causal_attn.mhsa.in_proj_bias = Parameter(
            te_gpt.self_attention.layernorm_qkv.bias.clone()
        )
        torch_gpt.causal_attn.mhsa.out_proj.weight = Parameter(
            te_gpt.self_attention.proj.weight.clone()
        )
        torch_gpt.causal_attn.mhsa.out_proj.bias = Parameter(
            te_gpt.self_attention.proj.bias.clone()
        )
        torch_gpt.ln_mlp.ln.weight = Parameter(te_gpt.layernorm_mlp.layer_norm_weight.clone())
        torch_gpt.ln_mlp.ln.bias = Parameter(te_gpt.layernorm_mlp.layer_norm_bias.clone())
        torch_gpt.ln_mlp.fc1.weight = Parameter(te_gpt.layernorm_mlp.fc1_weight.clone())
        torch_gpt.ln_mlp.fc1.bias = Parameter(te_gpt.layernorm_mlp.fc1_bias.clone())
        torch_gpt.ln_mlp.fc2.weight = Parameter(te_gpt.layernorm_mlp.fc2_weight.clone())
        torch_gpt.ln_mlp.fc2.bias = Parameter(te_gpt.layernorm_mlp.fc2_bias.clone())

    te_outputs = _test_e2e_gpt_accuracy(te_gpt, bs, dtype, config)
    torch_outputs = _test_e2e_gpt_accuracy(torch_gpt, bs, dtype, config)

    atol = {
        torch.float32: 5e-3,
        torch.half: 5e-2,
        torch.bfloat16: 1e-1,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])

    # Check gradients, only for small model
    if model == "small":
        atol[torch.float32] = 5e-2
        rtol = {
            torch.float32: 1e-2,
            torch.half: 1e-2,
            torch.bfloat16: 1e-2,
        }
        for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
            assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


def _test_mha_accuracy(block, bs, dtype, config, mask_type, te=True):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()
    inp_attn_mask = get_causal_attn_mask(config.max_seqlen_q) if mask_type == "causal" else None

    forward_kwargs = {}
    if te:
        forward_kwargs["attn_mask_type"] = mask_type
    forward_kwargs["attention_mask"] = inp_attn_mask

    out = block(inp_hidden_states, **forward_kwargs)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("mask_type", mask_types)
def test_mha_accuracy(dtype, bs, model, mask_type):
    config = model_configs[model]

    te_mha = MultiheadAttention(
        config.hidden_size,
        config.num_heads,
        fuse_qkv_params=True,
        params_dtype=dtype,
        qkv_weight_interleaved=False,
        input_layernorm=False,
        device="cuda",
    ).eval()

    torch_mha = (
        TorchMHA(
            config.hidden_size,
            config.num_heads,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_mha.mhsa.in_proj_weight = Parameter(te_mha.qkv.weight.clone())
        torch_mha.mhsa.in_proj_bias = Parameter(te_mha.qkv.bias.clone())
        torch_mha.mhsa.out_proj.weight = Parameter(te_mha.proj.weight.clone())
        torch_mha.mhsa.out_proj.bias = Parameter(te_mha.proj.bias.clone())

    te_outputs = _test_mha_accuracy(te_mha, bs, dtype, config, mask_type, te=True)
    torch_outputs = _test_mha_accuracy(torch_mha, bs, dtype, config, mask_type, te=False)

    # Check output.
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)

    # Check gradients, only for small model
    if model == "small":
        atol = {
            torch.float32: 5e-2,
            torch.half: 5e-2,
            torch.bfloat16: 5e-2,
        }
        rtol = {
            torch.float32: 1e-2,
            torch.half: 1e-2,
            torch.bfloat16: 1e-2,
        }
        for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
            assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


def _test_granular_accuracy(block, bs, dtype, config, delay_wgrad_compute=False, recipe=None):
    reset_rng_states()
    fp8 = recipe is not None
    if fp8:
        FP8GlobalStateManager.reset()

    inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    with autocast(enabled=fp8, recipe=recipe):
        out = block(inp_hidden_states)
        if isinstance(out, (List, Tuple)):
            out = out[0]
    loss = out.sum()
    loss.backward()
    if delay_wgrad_compute:
        block.backward_dw()

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


def _test_dpa_accuracy(block, bs, dtype, config):
    reset_rng_states()

    mask = torch.triu(
        torch.ones(config.max_seqlen_q, config.max_seqlen_kv, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    query, key, value = [
        torch.randn(
            (config.max_seqlen_q, bs, config.num_heads, config.kv_channels),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        for _ in range(3)
    ]

    query.retain_grad()
    key.retain_grad()
    value.retain_grad()

    out = block(query, key, value, attention_mask=mask)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()

    return [out, query.grad, key.grad, value.grad]


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
def test_dpa_accuracy(dtype, bs, model):
    config = model_configs[model]

    te_dpa = (
        DotProductAttention(
            config.num_heads,
            config.kv_channels,
            attention_dropout=0.0,  # disable dropout, FU uses rng differently
        )
        .to(dtype=dtype)
        .cuda()
    )

    torch_dpa = (
        TorchDotProductAttention(
            config.kv_channels,
            0.0,  # dropout
        )
        .to(dtype=dtype)
        .cuda()
    )

    te_outputs = _test_dpa_accuracy(te_dpa, bs, dtype, config)
    torch_outputs = _test_dpa_accuracy(torch_dpa, bs, dtype, config)

    # Check output.
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)

    for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
        assert_allclose(te_output, torch_output, atol=5e-2, rtol=1e-2)


class TestReturnBiasModule(nn.Module):
    def __init__(self, mod, **kwargs):
        super().__init__()
        self.te_module = mod(**kwargs)
        self.return_bias = kwargs["return_bias"]
        self.bias = kwargs["bias"]

    def forward(self, x):
        if self.return_bias:
            out, bias = self.te_module(x)
            if self.bias:
                out = out + bias
            return out
        return self.te_module(x)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("return_bias", all_boolean)
@pytest.mark.parametrize("bias", all_boolean)
def test_linear_accuracy(dtype, bs, model, return_bias, bias):
    config = model_configs[model]

    te_linear = TestReturnBiasModule(
        Linear,
        in_features=config.hidden_size,
        out_features=4 * config.hidden_size,
        params_dtype=dtype,
        return_bias=return_bias,
        bias=bias,
        device="cuda",
    )

    torch_linear = torch.nn.Linear(
        config.hidden_size,
        4 * config.hidden_size,
        bias=bias,
        device="cuda",
        dtype=dtype,
    )

    # Share params
    with torch.no_grad():
        torch_linear.weight = Parameter(te_linear.te_module.weight.clone())
        if bias:
            torch_linear.bias = Parameter(te_linear.te_module.bias.clone())

    te_outputs = _test_granular_accuracy(te_linear, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_linear, bs, dtype, config)

    # Check output.
    if model == "small":
        tolerance = 5e-3 if dtype == torch.float32 else 5e-2
        rtol = {
            torch.float32: 1.3e-6,
            torch.half: 1e-2,
            torch.bfloat16: 2e-2,
        }
        for te_output, torch_output in zip(te_outputs, torch_outputs):
            assert_allclose(te_output, torch_output, tolerance, rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("bias", all_boolean)
@pytest.mark.parametrize("fuse_wgrad_accumulation", all_boolean)
def test_linear_accuracy_delay_wgrad_compute(dtype, bs, model, bias, fuse_wgrad_accumulation):
    if NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("Delayed wgrad compute is not supported in debug mode.")

    config = model_configs[model]

    te_linear_ref = Linear(
        config.hidden_size,
        4 * config.hidden_size,
        bias=bias,
        params_dtype=dtype,
        device="cuda",
        delay_wgrad_compute=False,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    ).eval()

    te_linear = Linear(
        config.hidden_size,
        4 * config.hidden_size,
        bias=bias,
        params_dtype=dtype,
        device="cuda",
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    ).eval()

    # Share params
    with torch.no_grad():
        te_linear_ref.weight = Parameter(te_linear.weight.clone())
        if bias:
            te_linear_ref.bias = Parameter(te_linear.bias.clone())
        if fuse_wgrad_accumulation:
            weight = getattr(te_linear, f"weight")
            weight.main_grad = torch.rand_like(weight, dtype=torch.float32)
            te_linear_ref.weight.main_grad = weight.main_grad.clone()

    te_outputs = _test_granular_accuracy(te_linear, bs, dtype, config, delay_wgrad_compute=True)
    te_outputs_ref = _test_granular_accuracy(
        te_linear_ref, bs, dtype, config, delay_wgrad_compute=False
    )

    # Should be bit-wise match
    for _, (o, o_ref) in enumerate(zip(te_outputs, te_outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("recipe", fp8_recipes + [None], ids=recipe_id)
def test_linear_accuracy_save_original_input(dtype, model, recipe):
    bs = 1
    fuse_wgrad_accumulation = True
    fp8_model_params = False
    fp8 = recipe is not None

    if fp8 and recipe.delayed():
        pytest.skip("DelayedScaling recipe is not supported with save_original_input")
    skip_unsupported_backward_override("linear", recipe, getattr(recipe, "backward_override", None))

    config = model_configs[model]
    if config.max_seqlen_q % 16 != 0 and fp8:
        pytest.skip("FP8 requires sequence length to be divisible by 16.")

    if recipe is not None and recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe {recipe.__class__.__name__}"
            )

    with quantized_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        te_linear_ref = Linear(
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=dtype,
            device="cuda",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            save_original_input=False,
        ).eval()

        te_linear = Linear(
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=dtype,
            device="cuda",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            save_original_input=True,
        ).eval()

    # Share params
    with torch.no_grad():
        te_linear_ref.weight = Parameter(te_linear.weight.clone())
        if fuse_wgrad_accumulation:
            weight = getattr(te_linear, f"weight")
            weight.main_grad = torch.rand_like(weight, dtype=torch.float32)
            te_linear_ref.weight.main_grad = weight.main_grad.clone()

    te_outputs = _test_granular_accuracy(te_linear, bs, dtype, config, recipe=recipe)
    te_outputs_ref = _test_granular_accuracy(te_linear_ref, bs, dtype, config, recipe=recipe)

    # Should be bit-wise match
    for i, (o, o_ref) in enumerate(zip(te_outputs, te_outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5, 1e-7])
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_accuracy(dtype, bs, model, eps, zero_centered_gamma):
    config = model_configs[model]

    te_rmsnorm = RMSNorm(
        config.hidden_size,
        eps=eps,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
    ).eval()

    torch_rmsnorm = (
        TorchRMSNorm(config.hidden_size, eps=eps, zero_centered_gamma=zero_centered_gamma)
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_rmsnorm.weight = Parameter(te_rmsnorm.weight.clone())

    te_outputs = _test_granular_accuracy(te_rmsnorm, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_rmsnorm, bs, dtype, config)

    atol = {
        torch.float32: 1e-7,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])

    atol[torch.float32] = 2e-3
    rtol = {
        torch.float32: 1.3e-6,
        torch.half: 1e-3,
        torch.bfloat16: 1.6e-2,
    }
    # Check gradients
    for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
        assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5, 1e-7])
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_accuracy(dtype, bs, model, eps, zero_centered_gamma):
    config = model_configs[model]

    te_layernorm = LayerNorm(
        config.hidden_size,
        eps=eps,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
    ).eval()

    torch_layernorm = (
        TorchLayerNorm(config.hidden_size, eps=eps, zero_centered_gamma=zero_centered_gamma)
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_layernorm.weight = Parameter(te_layernorm.weight.clone())
        torch_layernorm.bias = Parameter(te_layernorm.bias.clone())

    te_outputs = _test_granular_accuracy(te_layernorm, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_layernorm, bs, dtype, config)

    atol = {
        torch.float32: 1e-7,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])

    rtol = {
        torch.float32: 1.3e-6,
        torch.half: 1e-3,
        torch.bfloat16: 1.6e-2,
    }
    atol[torch.float32] = 1e-4
    # Check gradients
    for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
        assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("normalization", all_normalizations)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("return_bias", all_boolean)
@pytest.mark.parametrize("bias", all_boolean)
def test_layernorm_linear_accuracy(
    dtype, bs, model, normalization, zero_centered_gamma, return_bias, bias
):
    config = model_configs[model]

    te_ln_linear = TestReturnBiasModule(
        LayerNormLinear,
        in_features=config.hidden_size,
        out_features=4 * config.hidden_size,
        eps=config.eps,
        normalization=normalization,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        return_bias=return_bias,
        bias=bias,
        device="cuda",
    )

    torch_ln_linear = (
        TorchLayerNormLinear(
            config.hidden_size,
            4 * config.hidden_size,
            config.eps,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
            bias=bias,
        )
        .to(dtype=dtype)
        .cuda()
    )

    # Share params
    with torch.no_grad():
        torch_ln_linear.layernorm.weight = Parameter(
            te_ln_linear.te_module.layer_norm_weight.clone()
        )
        if normalization != "RMSNorm":
            torch_ln_linear.layernorm.bias = Parameter(
                te_ln_linear.te_module.layer_norm_bias.clone()
            )
        torch_ln_linear.linear.weight = Parameter(te_ln_linear.te_module.weight.clone())
        if bias:
            torch_ln_linear.linear.bias = Parameter(te_ln_linear.te_module.bias.clone())

    te_outputs = _test_granular_accuracy(te_ln_linear, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_ln_linear, bs, dtype, config)

    atol = {
        torch.float32: 2.5e-4,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }
    rtol = {
        torch.float32: 1e-3,
        torch.half: 4e-2,
        torch.bfloat16: 4e-2,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype], rtol[dtype])

    if model == "small":
        atol = {
            torch.float32: 1e-3,
            torch.half: 5e-2,
            torch.bfloat16: 5e-2,
        }
        rtol = {
            torch.float32: 1e-3,
            torch.half: 4e-2,
            torch.bfloat16: 4e-2,
        }
        # Check gradients
        for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
            assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("normalization", all_normalizations)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("bias", all_boolean)
@pytest.mark.parametrize("fuse_wgrad_accumulation", all_boolean)
def test_layernorm_linear_accuracy_delay_wgrad_compute(
    dtype, bs, model, normalization, zero_centered_gamma, bias, fuse_wgrad_accumulation
):
    if NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("Delayed wgrad compute is not supported in debug mode.")

    config = model_configs[model]

    ln_linear_ref = LayerNormLinear(
        config.hidden_size,
        4 * config.hidden_size,
        config.eps,
        bias=bias,
        normalization=normalization,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
        delay_wgrad_compute=False,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    ).eval()

    ln_linear = LayerNormLinear(
        config.hidden_size,
        4 * config.hidden_size,
        config.eps,
        bias=bias,
        normalization=normalization,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    ).eval()

    # Share params
    with torch.no_grad():
        ln_linear_ref.layer_norm_weight = Parameter(ln_linear.layer_norm_weight.clone())
        if normalization != "RMSNorm":
            ln_linear_ref.layer_norm_bias = Parameter(ln_linear.layer_norm_bias.clone())
        ln_linear_ref.weight = Parameter(ln_linear.weight.clone())
        if bias:
            ln_linear_ref.bias = Parameter(ln_linear.bias.clone())
        if fuse_wgrad_accumulation:
            weight = getattr(ln_linear, f"weight")
            weight.main_grad = torch.rand_like(weight, dtype=torch.float32)
            ln_linear_ref.weight.main_grad = weight.main_grad.clone()

    te_outputs = _test_granular_accuracy(ln_linear, bs, dtype, config, delay_wgrad_compute=True)
    te_outputs_ref = _test_granular_accuracy(
        ln_linear_ref, bs, dtype, config, delay_wgrad_compute=False
    )

    # Shoule be bit-wise match
    for i, (o, o_ref) in enumerate(zip(te_outputs, te_outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
@pytest.mark.parametrize("return_bias", all_boolean)
@pytest.mark.parametrize("bias", all_boolean)
def test_layernorm_mlp_accuracy(dtype, bs, model, activation, normalization, return_bias, bias):
    config = model_configs[model]

    te_ln_mlp = TestReturnBiasModule(
        LayerNormMLP,
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        activation=activation,
        normalization=normalization,
        params_dtype=dtype,
        return_bias=return_bias,
        bias=bias,
        device="cuda",
    )

    torch_ln_mlp = (
        TorchLayerNormMLP(
            config.hidden_size,
            4 * config.hidden_size,
            activation=activation,
            normalization=normalization,
            bias=bias,
        )
        .to(dtype=dtype)
        .cuda()
    )

    # Share params
    with torch.no_grad():
        torch_ln_mlp.ln.weight = Parameter(te_ln_mlp.te_module.layer_norm_weight.clone())
        if normalization != "RMSNorm":
            torch_ln_mlp.ln.bias = Parameter(te_ln_mlp.te_module.layer_norm_bias.clone())
        torch_ln_mlp.fc1.weight = Parameter(te_ln_mlp.te_module.fc1_weight.clone())
        torch_ln_mlp.fc2.weight = Parameter(te_ln_mlp.te_module.fc2_weight.clone())
        if bias:
            torch_ln_mlp.fc1.bias = Parameter(te_ln_mlp.te_module.fc1_bias.clone())
            torch_ln_mlp.fc2.bias = Parameter(te_ln_mlp.te_module.fc2_bias.clone())

    te_outputs = _test_granular_accuracy(te_ln_mlp, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_ln_mlp, bs, dtype, config)

    atol = {
        torch.float32: 2e-2,
        torch.half: 5e-2,
        torch.bfloat16: 5e-2,
    }

    rtol = {
        torch.float32: 1e-3,
        torch.half: 4e-2,
        torch.bfloat16: 4e-2,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype], rtol[dtype])

    # Check gradients, only for small model
    rtol = {
        torch.float32: 1e-3,
        torch.half: 1e-2,
        torch.bfloat16: 4e-2,
    }
    atol[torch.half] = 2e-1
    atol[torch.bfloat16] = 2e-1
    if model == "small":
        for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
            assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", [2])
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("bias", all_boolean)
@pytest.mark.parametrize("fuse_wgrad_accumulation", all_boolean)
def test_layernorm_mlp_accuracy_delay_wgrad_compute(
    dtype,
    bs,
    model,
    bias,
    fuse_wgrad_accumulation,
):
    if NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("Delayed wgrad compute is not supported in debug mode.")

    config = model_configs[model]

    ln_mlp = LayerNormMLP(
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        eps=config.eps,
        bias=bias,
        params_dtype=dtype,
        device="cuda",
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    ).eval()

    ln_mlp_ref = LayerNormMLP(
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        eps=config.eps,
        bias=bias,
        params_dtype=dtype,
        device="cuda",
        delay_wgrad_compute=False,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    ).eval()

    # Share params
    with torch.no_grad():
        ln_mlp_ref.layer_norm_weight = Parameter(ln_mlp.layer_norm_weight.clone())
        ln_mlp_ref.layer_norm_bias = Parameter(ln_mlp.layer_norm_bias.clone())
        ln_mlp_ref.fc1_weight = Parameter(ln_mlp.fc1_weight.clone())
        ln_mlp_ref.fc2_weight = Parameter(ln_mlp.fc2_weight.clone())
        if bias:
            ln_mlp_ref.fc1_bias = Parameter(ln_mlp.fc1_bias.clone())
            ln_mlp_ref.fc2_bias = Parameter(ln_mlp.fc2_bias.clone())
        if fuse_wgrad_accumulation:
            ln_mlp.fc1_weight.main_grad = torch.rand_like(ln_mlp.fc1_weight, dtype=torch.float32)
            ln_mlp_ref.fc1_weight.main_grad = ln_mlp.fc1_weight.main_grad.clone()
            ln_mlp.fc2_weight.main_grad = torch.rand_like(ln_mlp.fc2_weight, dtype=torch.float32)
            ln_mlp_ref.fc2_weight.main_grad = ln_mlp.fc2_weight.main_grad.clone()

    te_outputs = _test_granular_accuracy(ln_mlp, bs, dtype, config, delay_wgrad_compute=True)
    te_outputs_ref = _test_granular_accuracy(
        ln_mlp_ref, bs, dtype, config, delay_wgrad_compute=False
    )

    # Shoule be bit-wise match
    for i, (o, o_ref) in enumerate(zip(te_outputs, te_outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", [2])
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("bias", all_boolean)
def test_layernorm_mlp_accuracy_checkpoint(
    dtype,
    bs,
    model,
    bias,
):
    config = model_configs[model]

    ln_mlp = LayerNormMLP(
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        eps=config.eps,
        bias=bias,
        params_dtype=dtype,
        device="cuda",
        checkpoint=True,
    ).eval()

    ln_mlp_ref = LayerNormMLP(
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        eps=config.eps,
        bias=bias,
        params_dtype=dtype,
        device="cuda",
        checkpoint=False,
    ).eval()

    # Share params
    with torch.no_grad():
        ln_mlp_ref.layer_norm_weight = Parameter(ln_mlp.layer_norm_weight.clone())
        ln_mlp_ref.layer_norm_bias = Parameter(ln_mlp.layer_norm_bias.clone())
        ln_mlp_ref.fc1_weight = Parameter(ln_mlp.fc1_weight.clone())
        ln_mlp_ref.fc2_weight = Parameter(ln_mlp.fc2_weight.clone())
        if bias:
            ln_mlp_ref.fc1_bias = Parameter(ln_mlp.fc1_bias.clone())
            ln_mlp_ref.fc2_bias = Parameter(ln_mlp.fc2_bias.clone())

    te_outputs = _test_granular_accuracy(ln_mlp, bs, dtype, config, delay_wgrad_compute=False)
    te_outputs_ref = _test_granular_accuracy(
        ln_mlp_ref, bs, dtype, config, delay_wgrad_compute=False
    )

    # Shoule be bit-wise match
    for i, (o, o_ref) in enumerate(zip(te_outputs, te_outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


def _test_gpt_e2e_cuda_graph(block, bs, dtype, config, graph):
    reset_rng_states()

    # Initialize loss function and optimizer.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)

    # Placeholders used for graph capture.
    static_input = torch.randn(
        config.max_seqlen_q, bs, config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    static_target = torch.randn(
        config.max_seqlen_q, bs, config.hidden_size, device="cuda", dtype=dtype
    )

    real_input = torch.rand_like(static_input)
    real_target = torch.rand_like(static_target)

    # Basic training loop.
    def train_step():
        optimizer.zero_grad(set_to_none=False)
        out = block(static_input)
        loss = loss_fn(out, static_target)
        loss.backward()
        optimizer.step()
        return out

    # Warmup steps in a separate stream.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            train_step()
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph.
    g = None
    static_output = None
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = train_step()

    # Run with new data.
    with torch.no_grad():
        static_input.copy_(real_input)
        static_target.copy_(real_target)
    if graph:
        g.replay()
    else:
        static_output = train_step()

    grads = [static_input.grad]
    for p in block.parameters():
        if p.requires_grad:
            grads.append(p.grad)

    with torch.no_grad():
        output = static_output.clone()
    return output, grads


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
def test_gpt_cuda_graph(dtype, bs, model):
    if NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("Cuda Graphs are not supported in debug mode.")
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block_args = (
        config.hidden_size,
        4 * config.hidden_size,
        config.num_heads,
    )
    block_kwargs = dict(
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
    )
    block = TransformerLayer(*block_args, **block_kwargs)
    graphed_block = TransformerLayer(*block_args, **block_kwargs)
    with torch.no_grad():
        for param1, param2 in zip(block.parameters(), graphed_block.parameters()):
            param2.copy_(param1)

    out, grads = _test_gpt_e2e_cuda_graph(block, bs, dtype, config, False)
    graphed_out, graphed_grads = _test_gpt_e2e_cuda_graph(graphed_block, bs, dtype, config, True)
    params = list(block.parameters())
    graphed_params = list(graphed_block.parameters())

    # Check that results match
    assert_allclose(out, graphed_out, 1e-3)
    assert_allclose(params, graphed_params, 1e-3)
    assert_allclose(grads, graphed_grads, 1e-3)


def _test_gpt_fp8_parameters(bs, dtype, config, fp8_model_params, recipe):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with quantized_model_init(enabled=fp8_model_params, recipe=recipe):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.max_seqlen_q)

    with autocast(enabled=True, recipe=recipe):
        te_out = block(te_inp_hidden_states, attention_mask=te_inp_attn_mask)
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

    outputs = [te_out, te_inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("recipe", fp8_recipes, ids=recipe_id)
def test_gpt_fp8_parameters(dtype, bs, model, recipe):
    if NVTE_TEST_NVINSPECT_ENABLED:
        pytest.skip("FP8 parameters are not supported in debug mode.")
    skip_unsupported_backward_override(
        "transformer_layer", recipe, getattr(recipe, "backward_override", None)
    )

    if recipe.nvfp4():
        if dtype not in get_nvfp4_inp_supported_dtypes(recipe, dtype):
            pytest.skip(
                f"Input dtype {dtype} not supported for NVFP4 Recipe {recipe.__class__.__name__}"
            )

    config = model_configs[model]

    outputs = _test_gpt_fp8_parameters(bs, dtype, config, False, recipe)
    outputs_fp8_params = _test_gpt_fp8_parameters(bs, dtype, config, True, recipe)

    # Check that results match
    tols = dict(rtol=0.125, atol=0.0675)
    for i, (ref, test) in enumerate(zip(outputs, outputs_fp8_params)):
        torch.testing.assert_close(
            test,
            ref,
            msg=f"Mismatch in tensor {i}",
            rtol=0.125,
            atol=0.0675,
        )


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
def test_transformer_layer_hidden_states_format(dtype, bs, model):
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    # Set `torch.manual_seed` to make sure the weights are identical to the
    # other layer. Set `*dropout` values to 0 to make sure the forward pass
    # is identical to the other layer.
    torch.manual_seed(0)
    block_sbhd = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0,
        attention_dropout=0,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
        attn_input_format="sbhd",
    )

    # Set `torch.manual_seed` to make sure the weights are identical to the
    # other layer. Set `*dropout` values to 0 to make sure the forward pass
    # is identical to the other layer.
    torch.manual_seed(0)
    block_bshd = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0,
        attention_dropout=0,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
        attn_input_format="bshd",
    )

    torch.manual_seed(0)
    block_thd = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0,
        attention_dropout=0,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
        attn_input_format="thd",
        self_attn_mask_type="padding_causal",
    )

    for (n1, p1), (n2, p2), (n3, p3) in zip(
        block_bshd.named_parameters(), block_sbhd.named_parameters(), block_thd.named_parameters()
    ):
        assert torch.all(torch.eq(p1, p2) & torch.eq(p1, p3)), f"{n1}, {n2} and {n3} not identical"

    x_sbhd = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    x_bshd = x_sbhd.transpose(0, 1).contiguous()
    x_thd = x_bshd.reshape(bs * config.max_seqlen_q, config.hidden_size).contiguous()
    x_thd_cumsum = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * config.max_seqlen_q

    # To make sure forward is also identical (just in case some module decides
    # to act fancy)
    torch.manual_seed(0)
    y_sbhd = block_sbhd(x_sbhd)

    # To make sure forward is also identical (just in case some module decides
    # to act fancy)
    torch.manual_seed(0)
    y_bshd = block_bshd(x_bshd)

    # Check that results match
    torch.testing.assert_close(
        y_bshd,
        y_sbhd.transpose(0, 1).contiguous(),
    )

    # THD is not supported in float32 and on GPUs older than Ampere, skip the test here
    if dtype != torch.float32 and sm_80plus:
        # To make sure forward is also identical (just in case some module decides
        # to act fancy)
        torch.manual_seed(0)
        y_thd = block_thd(
            x_thd,
            cu_seqlens_q=x_thd_cumsum,
            cu_seqlens_kv=x_thd_cumsum,
            max_seqlen_q=config.max_seqlen_q,
            max_seqlen_kv=config.max_seqlen_kv,
        )

        torch.testing.assert_close(
            y_bshd,
            y_thd.reshape(bs, config.max_seqlen_q, config.hidden_size).contiguous(),
        )


@pytest.mark.parametrize("N", [32])
@pytest.mark.parametrize("datatype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "input_quantizer",
    [
        Float8CurrentScalingQuantizer(fp8_dtype=DType.kFloat8E4M3, device="cuda"),
        MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3),
    ],
)
@pytest.mark.parametrize(
    "out_quantizer",
    [
        Float8CurrentScalingQuantizer(fp8_dtype=DType.kFloat8E4M3, device="cuda"),
        MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3),
        Float8Quantizer(
            torch.ones(1).cuda().squeeze(),
            torch.ones(1).cuda().squeeze(),
            DType.kFloat8E4M3,
        ),
    ],
)
def test_fp8gemm_with_unfused_quantization(N, datatype, input_quantizer, out_quantizer):
    # For MXFP8 and CurrentScaling, below unfused quantization should happen
    # FP8 input --> cublas GEMM --> BF16 output --> Quantize to FP8 --> fp8 Output
    # Skip invalid configurations
    is_mxfp8_needed = isinstance(input_quantizer, MXFP8Quantizer) or isinstance(
        out_quantizer, MXFP8Quantizer
    )
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if is_mxfp8_needed and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    inp_fp8 = input_quantizer(torch.randn(N, N, device="cuda", dtype=datatype))
    weight_fp8 = input_quantizer(torch.randn(N, N, device="cuda", dtype=datatype))
    outp_type = torch.float32
    quantized_out, *_ = general_gemm(
        weight_fp8,
        inp_fp8,
        outp_type,
        quantization_params=out_quantizer,
        bias=None,
        use_split_accumulator=False,
    )

    out, *_ = general_gemm(
        weight_fp8,
        inp_fp8,
        outp_type,
        quantization_params=None,
        bias=None,
        use_split_accumulator=False,
    )
    expected_quantized_out = out_quantizer(out)

    # Match results again Pytorch GEMM and allow for quantization tolerance
    pytorch_out = torch.matmul(
        inp_fp8.dequantize().to(torch.float64),
        torch.transpose(weight_fp8.dequantize().to(torch.float64), 0, 1),
    )
    fp8_tols = dict(rtol=0.125, atol=0.0675)
    torch.testing.assert_close(
        pytorch_out.to(outp_type), expected_quantized_out.dequantize(), **fp8_tols
    )
    # Match results between quantization happening inside vs outside general_gemm
    torch.testing.assert_close(expected_quantized_out.dequantize(), quantized_out.dequantize())


def test_noncontiguous():
    def _create2modules(m, params):
        mod1 = m(*params)
        mod2 = m(*params)
        for p1, p2 in zip(mod1.parameters(), mod2.parameters()):
            p2.data = p1.data.clone()

        return mod1, mod2

    def _run_module(m, inp):
        out = m(inp)
        out.sum().backward()
        ret = [out]
        if inp.grad is not None:
            ret.append(inp.grad)

        for p in m.parameters():
            if p.requires_grad:
                ret.append(p.grad)
        return ret

    a = torch.randn((128, 256), device="cuda", requires_grad=True)
    a = a.T
    assert not a.is_contiguous(), "The test is supposed to test noncontiguous input."

    b = a.contiguous()

    # LayerNorm
    ln1, ln2 = _create2modules(LayerNorm, [128])
    outT = _run_module(ln1, a)
    out = _run_module(ln2, b)

    assert_allclose(out, outT, 1e-7)

    # RMSNorm
    ln1, ln2 = _create2modules(RMSNorm, [128])
    outT = _run_module(ln1, a)
    out = _run_module(ln2, b)

    assert_allclose(out, outT, 1e-7)

    # GEMM
    g1, g2 = _create2modules(Linear, [128, 128])
    outT = _run_module(g1, a)
    out = _run_module(g2, b)

    assert_allclose(out, outT, 1e-7)
