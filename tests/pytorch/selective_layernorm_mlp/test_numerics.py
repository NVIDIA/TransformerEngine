# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os

# disable tf32 for numerics test
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS", "0")
os.environ.setdefault("TORCH_ALLOW_TF32_CUDNN", "0")
os.environ.setdefault("PYTORCH_CUDA_MATMUL_ALLOW_TF32", "0")
os.environ.setdefault("PYTORCH_CUDNN_ALLOW_TF32", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from typing import Dict, List, Tuple
import pytest

import torch
import torch.nn as nn
from torch.nn import Parameter

from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch import (
    autocast,
    SelectiveLayerNormMLP,
    get_device_compute_capability,
    is_fp8_available,
    is_mxfp8_available,
    is_fp8_block_scaling_available,
    is_bf16_available,
    is_nvfp4_available,
)
from transformer_engine.common import recipe
from utils import ModelConfig, reset_rng_states


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


class TorchLayerNorm(nn.Module):
    def __init__(self, in_features: int, eps: float, zero_centered_gamma: bool):
        super().__init__()
        self.eps = eps
        self.in_features = in_features
        self.zero_centered_gamma = zero_centered_gamma

        initial_value = torch.zeros(in_features) if zero_centered_gamma else torch.ones(in_features)
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

        initial_value = torch.zeros(in_features) if zero_centered_gamma else torch.ones(in_features)
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


class TorchQuickGELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class TorchSquaredRELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input > 0) * input * input


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


_supported_act = {
    "gelu": nn.GELU(approximate="tanh"),
    "geglu": nn.GELU(approximate="tanh"),
    "qgelu": TorchQuickGELU(),
    "qgeglu": TorchQuickGELU(),
    "relu": nn.ReLU(),
    "reglu": nn.ReLU(),
    "srelu": TorchSquaredRELU(),
    "sreglu": TorchSquaredRELU(),
    "silu": nn.SiLU(),
    "swiglu": nn.SiLU(),
}


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

use_cutlass_grouped_gemm = [False]
# Only enable cutlass grouped gemm on Hopper
if torch.cuda.get_device_capability() == (9, 0):
    use_cutlass_grouped_gemm.append(True)


def get_causal_attn_mask(sq: int) -> torch.Tensor:
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()


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
    raise ValueError(f"Unsuppored dtype ({dtype})")


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


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
@pytest.mark.parametrize("return_bias", all_boolean)
@pytest.mark.parametrize("bias", all_boolean)
@pytest.mark.parametrize("checkpoint", all_boolean)
def test_selective_layernorm_mlp_accuracy(
    dtype, bs, model, activation, normalization, return_bias, bias, checkpoint
):
    config = model_configs[model]

    te_ln_mlp = TestReturnBiasModule(
        SelectiveLayerNormMLP,
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        activation=activation,
        normalization=normalization,
        params_dtype=dtype,
        return_bias=return_bias,
        bias=bias,
        checkpoint=checkpoint,
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
@pytest.mark.parametrize("checkpoint", all_boolean)
def test_selective_layernorm_mlp_accuracy_delay_wgrad_compute(
    dtype, bs, model, bias, fuse_wgrad_accumulation, checkpoint
):
    config = model_configs[model]

    ln_mlp = SelectiveLayerNormMLP(
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        eps=config.eps,
        bias=bias,
        params_dtype=dtype,
        checkpoint=checkpoint,
        device="cuda",
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    ).eval()

    ln_mlp_ref = SelectiveLayerNormMLP(
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
