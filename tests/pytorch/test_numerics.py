# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
import os
from typing import Dict, List, Optional
import pytest
import copy

import torch
import torch.nn as nn
from torch.nn import Parameter

from transformer_engine.pytorch.fp8 import fp8_autocast, FP8GlobalStateManager, fp8_model_init
from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
    attention_mask_func,
    is_bf16_compatible,
)
from transformer_engine.pytorch import (
    DotProductAttention,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    MultiheadAttention,
    RMSNorm,
    TransformerLayer,
    LayerNorm,
    InferenceParams,
)
from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint


# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# Record initial RNG state from script run.
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()


class ModelConfig:
    def __init__(self, hidden_size, eps, num_attention_heads, embed, num_layers, seq_len):
        self.hidden_size = hidden_size
        self.eps = eps
        self.num_attention_heads = num_attention_heads
        self.embed = embed
        self.num_layers = num_layers
        self.seq_len = seq_len


model_configs = {
    "126m": ModelConfig(768, 1e-5, 12, 64, 12, 2048),
}

model_configs_inference = {
    # hidden_size, eps, num_attention_heads, embed, num_layers, seq_len
    "126m": ModelConfig(768, 1e-5, 12, 64, 12, 16),
}
backends_inference = ["FlashAttention", "UnfusedAttention"]
module_inference = ["TransformerLayer", "MultiheadAttention"]
input_formats_inference = ["sbhd", "bshd"]

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

batch_sizes = [1, 2]

all_boolean = [True, False]

all_activations = ["gelu", "relu", "reglu", "geglu", "swiglu", "qgelu", "srelu"]

all_normalizations = ["LayerNorm", "RMSNorm"]

mask_types = ["causal", "no_mask"]


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
    raise ValueError(f"Unsuppored dtype ({dtype})")


def assert_allclose(
    l1: List[torch.Tensor],
    l2: List[torch.Tensor],
    atol: float,
) -> bool:
    """Ensures two lists are equal."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    for i, (t1, t2) in enumerate(zip(l1, l2)):
        result = torch.allclose(t1, t2, atol=atol)
        if not result:
            diff = torch.abs(t1 - t2).flatten()
            m = torch.argmax(diff)
            msg = (
                f"Outputs not close enough in tensor at idx={i}. "
                f"Location of the maximum difference: {m.item()} "
                f"with {t1.flatten()[m].item()} vs {t2.flatten()[m].item()} "
                f"(diff {diff[m].item()})."
            )
            raise AssertionError(msg)


def reset_rng_states() -> None:
    """revert back to initial RNG state."""
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state(_cuda_rng_state)


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
        bias: bool = True,
        normalization: str = "LayerNorm",
        zero_centered_gamma: bool = False,
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

        self.linear = nn.Linear(in_features, out_features)

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
    "geglu": nn.GELU(approximate="tanh"),
    "gelu": nn.GELU(approximate="tanh"),
    "reglu": nn.ReLU(),
    "relu": nn.ReLU(),
    "swiglu": nn.SiLU(),
    "qgelu": TorchQuickGELU(),
    "srelu": TorchSquaredRELU(),
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

        self.fc1 = nn.Linear(hidden_size, fc1_output_features)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(self.ln(x))))


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


def _test_e2e_selective_recompute(bs, dtype, config, fp8, fp8_model_params=False, recompute=False):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with fp8_model_init(enabled=fp8 and fp8_model_params):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.embed,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    with fp8_autocast(enabled=fp8):
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
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
def test_gpt_selective_activation_recompute(dtype, bs, model, fp8, fp8_model_params):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)

    config = model_configs[model]

    outputs = _test_e2e_selective_recompute(
        bs, dtype, config, fp8, fp8_model_params, recompute=False
    )
    outputs_recompute = _test_e2e_selective_recompute(
        bs, dtype, config, fp8, fp8_model_params, recompute=True
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
    bs, dtype, config, fp8, fp8_model_params=False, recompute=False, use_reentrant=True
):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with fp8_model_init(enabled=fp8 and fp8_model_params):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.embed,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=use_reentrant,
    )
    if use_reentrant:
        te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    with fp8_autocast(enabled=fp8):
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
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
@pytest.mark.parametrize("use_reentrant", all_boolean)
def test_gpt_full_activation_recompute(dtype, bs, model, fp8, fp8_model_params, use_reentrant):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)

    config = model_configs[model]

    if not use_reentrant:
        # Non-reentrant checkpoint becomes non-deterministic with bias+GELU fusion
        os.environ["NVTE_BIAS_GELU_NVFUSION"] = "0"

    outputs, names = _test_e2e_full_recompute(
        bs, dtype, config, fp8, fp8_model_params, recompute=False, use_reentrant=use_reentrant
    )
    outputs_recompute, _ = _test_e2e_full_recompute(
        bs, dtype, config, fp8, fp8_model_params, recompute=True, use_reentrant=use_reentrant
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
        config.num_attention_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.embed,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        params_dtype=dtype,
        device="cuda",
    )


def _test_e2e_checkpointing(bs, dtype, config, checkpoint=False, steps=10, path="checkpoint.pt"):
    reset_rng_states()

    te_inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
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

        global _cpu_rng_state, _cuda_rng_state
        _cpu_rng_state = torch.get_rng_state()
        _cuda_rng_state = torch.cuda.get_rng_state()

        del block
        block = _test_e2e_checkpointing_get_model(config, dtype)
        block.load_state_dict(torch.load(path))
        reset_rng_states()

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
@pytest.mark.parametrize("model", model_configs.keys())
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
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()
    inp_attn_mask = get_causal_attn_mask(config.seq_len)

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
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("parallel_attention_mlp", all_boolean)
def test_gpt_accuracy(dtype, bs, model, parallel_attention_mlp):
    config = model_configs[model]

    te_gpt = TransformerLayer(
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        num_attention_heads=config.num_attention_heads,
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
            config.num_attention_heads,
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

    # Check output.
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)


def _test_mha_accuracy(block, bs, dtype, config, mask_type, te=True):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()
    inp_attn_mask = get_causal_attn_mask(config.seq_len) if mask_type == "causal" else None

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
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("mask_type", mask_types)
def test_mha_accuracy(dtype, bs, model, mask_type):
    config = model_configs[model]

    te_mha = MultiheadAttention(
        config.hidden_size,
        config.num_attention_heads,
        fuse_qkv_params=True,
        params_dtype=dtype,
        qkv_weight_interleaved=False,
        input_layernorm=False,
        device="cuda",
    ).eval()

    torch_mha = (
        TorchMHA(
            config.hidden_size,
            config.num_attention_heads,
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


def _test_granular_accuracy(block, bs, dtype, config):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    out = block(inp_hidden_states)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


def _test_dpa_accuracy(block, bs, dtype, config):
    reset_rng_states()

    mask = torch.triu(
        torch.ones(config.seq_len, config.seq_len, dtype=torch.bool, device="cuda"), diagonal=1
    )
    query, key, value = [
        torch.randn(
            (config.seq_len, bs, config.num_attention_heads, config.embed),
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
@pytest.mark.parametrize("model", model_configs.keys())
def test_dpa_accuracy(dtype, bs, model):
    config = model_configs[model]

    te_dpa = (
        DotProductAttention(
            config.num_attention_heads,
            config.embed,
            attention_dropout=0.0,  # disable dropout, FU uses rng differently
        )
        .to(dtype=dtype)
        .cuda()
    )

    torch_dpa = (
        TorchDotProductAttention(
            config.embed,
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


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
def test_linear_accuracy(dtype, bs, model):
    config = model_configs[model]

    te_linear = Linear(
        config.hidden_size,
        4 * config.hidden_size,
        bias=True,
        params_dtype=dtype,
        device="cuda",
    ).eval()

    torch_linear = torch.nn.Linear(
        config.hidden_size,
        4 * config.hidden_size,
        bias=True,
        device="cuda",
        dtype=dtype,
    ).eval()

    # Share params
    with torch.no_grad():
        torch_linear.weight = Parameter(te_linear.weight.clone())
        torch_linear.bias = Parameter(te_linear.bias.clone())

    te_outputs = _test_granular_accuracy(te_linear, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_linear, bs, dtype, config)

    # Check output.
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
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

    # Check output.
    atol = {
        torch.float32: 1e-7,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
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

    # Check output.
    atol = {
        torch.float32: 1e-7,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("normalization", all_normalizations)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_linear_accuracy(dtype, bs, model, normalization, zero_centered_gamma):
    config = model_configs[model]

    te_ln_linear = LayerNormLinear(
        config.hidden_size,
        4 * config.hidden_size,
        config.eps,
        bias=True,
        normalization=normalization,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
    ).eval()

    torch_ln_linear = (
        TorchLayerNormLinear(
            config.hidden_size,
            4 * config.hidden_size,
            config.eps,
            bias=True,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_ln_linear.layernorm.weight = Parameter(te_ln_linear.layer_norm_weight.clone())
        if normalization != "RMSNorm":
            torch_ln_linear.layernorm.bias = Parameter(te_ln_linear.layer_norm_bias.clone())
        torch_ln_linear.linear.weight = Parameter(te_ln_linear.weight.clone())
        torch_ln_linear.linear.bias = Parameter(te_ln_linear.bias.clone())

    te_outputs = _test_granular_accuracy(te_ln_linear, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_ln_linear, bs, dtype, config)

    # Check output.
    atol = {
        torch.float32: 2.5e-4,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_layernorm_mlp_accuracy(dtype, bs, model, activation, normalization):
    config = model_configs[model]

    te_ln_mlp = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        activation=activation,
        normalization=normalization,
        params_dtype=dtype,
        device="cuda",
    ).eval()

    torch_ln_mlp = (
        TorchLayerNormMLP(
            config.hidden_size,
            4 * config.hidden_size,
            activation=activation,
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_ln_mlp.ln.weight = Parameter(te_ln_mlp.layer_norm_weight.clone())
        if normalization != "RMSNorm":
            torch_ln_mlp.ln.bias = Parameter(te_ln_mlp.layer_norm_bias.clone())
        torch_ln_mlp.fc1.weight = Parameter(te_ln_mlp.fc1_weight.clone())
        torch_ln_mlp.fc1.bias = Parameter(te_ln_mlp.fc1_bias.clone())
        torch_ln_mlp.fc2.weight = Parameter(te_ln_mlp.fc2_weight.clone())
        torch_ln_mlp.fc2.bias = Parameter(te_ln_mlp.fc2_bias.clone())

    te_outputs = _test_granular_accuracy(te_ln_mlp, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_ln_mlp, bs, dtype, config)

    # Check output.
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)


def _test_gpt_e2e_cuda_graph(block, bs, dtype, config, graph):
    reset_rng_states()

    # Initialize loss function and optimizer.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)

    # Placeholders used for graph capture.
    static_input = torch.randn(
        config.seq_len, bs, config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    static_target = torch.randn(config.seq_len, bs, config.hidden_size, device="cuda", dtype=dtype)

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
@pytest.mark.parametrize("model", model_configs.keys())
def test_gpt_cuda_graph(dtype, bs, model):
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.embed,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
    )
    graphed_block = copy.deepcopy(block)

    out, grads = _test_gpt_e2e_cuda_graph(block, bs, dtype, config, False)
    graphed_out, graphed_grads = _test_gpt_e2e_cuda_graph(graphed_block, bs, dtype, config, True)
    params = list(block.parameters())
    graphed_params = list(graphed_block.parameters())

    # Check that results match
    assert_allclose(out, graphed_out, 1e-3)
    assert_allclose(params, graphed_params, 1e-3)
    assert_allclose(grads, graphed_grads, 1e-3)


def _test_gpt_fp8_parameters(bs, dtype, config, fp8_model_params):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with fp8_model_init(enabled=fp8_model_params):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.embed,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    with fp8_autocast(enabled=True):
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
@pytest.mark.parametrize("model", model_configs.keys())
def test_gpt_fp8_parameters(dtype, bs, model):
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    config = model_configs[model]

    outputs = _test_gpt_fp8_parameters(bs, dtype, config, False)
    outputs_fp8_params = _test_gpt_fp8_parameters(bs, dtype, config, True)

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
@pytest.mark.parametrize("model", model_configs.keys())
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
        config.num_attention_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0,
        attention_dropout=0,
        kv_channels=config.embed,
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
        config.num_attention_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0,
        attention_dropout=0,
        kv_channels=config.embed,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
        attn_input_format="bshd",
    )

    for (n1, p1), (n2, p2) in zip(block_bshd.named_parameters(), block_sbhd.named_parameters()):
        assert torch.all(torch.eq(p1, p2)), f"{n1}, {n2} not identical"

    x_sbhd = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    x_bshd = x_sbhd.transpose(0, 1).contiguous()

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


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model_key", model_configs_inference.keys())
@pytest.mark.parametrize("use_RoPE", all_boolean)
@pytest.mark.parametrize("input_format", input_formats_inference)
@pytest.mark.parametrize("module", module_inference)
@pytest.mark.parametrize("backend", backends_inference)
def test_kv_cache_accuracy(dtype, bs, model_key, use_RoPE, input_format, module, backend):
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"

    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    elif backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    config = model_configs_inference[model_key]

    S = config.seq_len
    B = bs
    H = config.num_attention_heads
    D = config.hidden_size
    head_size = config.embed
    layer_number = 1

    # Limits the max size of KV-cache
    B_max = B
    S_max = S + 2

    if module == "TransformerLayer":
        model = TransformerLayer(
            hidden_size=D,
            ffn_hidden_size=4 * D,
            num_attention_heads=H,
            attn_input_format=input_format,
            layer_number=layer_number,
            attention_dropout=0.0,
            params_dtype=dtype,
            device="cuda",
        ).eval()
    else:
        model = (
            MultiheadAttention(
                hidden_size=D,
                num_attention_heads=H,
                qkv_format=input_format,
                layer_number=layer_number,
                attention_dropout=0.0,
                params_dtype=dtype,
            )
            .cuda()
            .eval()
        )

    inference_params = InferenceParams(max_batch_size=B_max, max_sequence_length=S_max)
    rotary_freqs = torch.randn((S_max, 1, 1, head_size), dtype=torch.float, device="cuda")

    input = torch.randn((S, B, D), dtype=dtype, device="cuda")
    if input_format == "bshd":
        input = input.transpose(0, 1).contiguous()

    incremental_output = torch.zeros_like(input)

    # Generate output for the entire sequence
    full_output = model(hidden_states=input, rotary_pos_emb=rotary_freqs if use_RoPE else None)

    # Incrementaly generate outputs using KV-cache
    for i in range(S):
        if input_format == "sbhd":
            incremental_input = input[i].view(1, B, D)
        else:
            incremental_input = input[:, i, :].view(B, 1, D)

        line_output = model(
            hidden_states=incremental_input,
            inference_params=inference_params,
            rotary_pos_emb=rotary_freqs if use_RoPE else None,
        )

        inference_params.sequence_len_offset += 1

        if input_format == "sbhd":
            incremental_output[i] = line_output.view(B, D)
        else:
            incremental_output[:, i, :] = line_output.view(B, D)

    if module == "TransformerLayer":
        atol = {
            torch.float32: 5e-3,
            torch.half: 5e-3,
            torch.bfloat16: 5e-2,
        }
    else:
        atol = {
            torch.float32: 1e-3,
            torch.half: 1e-3,
            torch.bfloat16: 1e-2,
        }

    # Check if the fully generated output matches the one generated incrementally
    assert_allclose(full_output, incremental_output, atol[dtype])
