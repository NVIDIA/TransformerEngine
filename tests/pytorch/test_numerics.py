# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
import os
import contextlib
from typing import List, Optional
import pytest
import copy

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager

from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
    attention_mask_func,
)
from transformer_engine.pytorch import (
    DotProductAttention, Linear, LayerNormLinear, LayerNormMLP, TransformerLayer, RMSNorm
)
from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint

seed = 1234
rng_str = "rng_state"
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

param_types = [torch.float32, torch.float16]
if torch.cuda.is_bf16_supported():
    param_types.append(torch.bfloat16)

batch_sizes = [1, 2]

all_boolean = [True, False]

all_activations = ["gelu", "relu", "reglu", "geglu", "swiglu"]

all_normalizations = ["LayerNorm", "RMSNorm"]

def get_causal_attn_mask(sq: int) -> torch.Tensor:
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()


def assert_all_equal(l1: List[torch.Tensor], l2: List[torch.Tensor]) -> bool:
    """Ensures two lists are equal."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    for t1, t2 in zip(l1, l2):
        assert torch.equal(t1, t2), "Output mismatch."


def assert_allclose(l1: List[torch.Tensor], l2: List[torch.Tensor], atol: float) -> bool:
    """Ensures two lists are equal."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    for t1, t2 in zip(l1, l2):
        result = torch.allclose(t1, t2, atol=atol)
        if not result:
            diff = torch.abs(t1 - t2).flatten()
            m = torch.argmax(diff)
            msg = (f"Outputs not close enough."
                   f"Location of the maximum difference: {m.item()} "
                   f"with {t1.flatten()[m].item()} vs {t2.flatten()[m].item()} "
                   f"(diff {diff[m].item()})."
            )
            raise AssertionError(msg)


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


def reset_rng_states() -> None:
    # revert back to initial RNG state.
    torch.set_rng_state(_cpu_rng_state)
    _set_cuda_rng_state(_cuda_rng_state)


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception("seed {} already exists".format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception("cuda rng state {} already exists".format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=rng_str):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception("cuda rng state {} is not added".format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


_DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
_DUMMY_CUDA_RNG_STATE_TRACKER.add(rng_str, seed)


def get_dummy_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _DUMMY_CUDA_RNG_STATE_TRACKER


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
        query_layer = query_layer.reshape(
            output_size[2], output_size[0] * output_size[1], -1
        )
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
        value_layer = value_layer.reshape(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context_layer = context_layer.view(seqlen, batch_size, -1)

        return context_layer

# Adapted from https://github.com/bzhangGo/rmsnorm/blob/c6691f20ec0af4128c8159c903071f7575404295/rmsnorm_torch.py
class TorchRMSNorm(nn.Module):
    def __init__(self, in_features, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.in_features = in_features

        self.weight = nn.Parameter(torch.ones(in_features))
        self.register_parameter("weight", self.weight)

    def forward(self, x):
        norm_x2 = torch.sum(x.float()**2, dim=-1, keepdim=True)
        d_x = self.in_features

        rms_x2 = norm_x2 / d_x + self.eps
        r_rms_x = rms_x2 ** (-1. / 2)
        x_normed = x * r_rms_x

        return (self.weight.float() * x_normed).to(x.dtype)

class TorchLayerNormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 eps: float, bias: bool = True,
                 normalization: str = "LayerNorm"):
        super().__init__()
        if normalization == "LayerNorm":
            self.layernorm = nn.LayerNorm(in_features, eps=eps)
        elif normalization == "RMSNorm":
            self.layernorm = TorchRMSNorm(in_features, eps=eps)
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

    def forward(self, x, attn_mask=None):
        return self.mhsa(x, x, x, attn_mask=attn_mask, need_weights=False)

_supported_act = {'geglu'  : nn.GELU(approximate="tanh"),
                  'gelu'  : nn.GELU(approximate="tanh"),
                  'reglu'  : nn.ReLU(),
                  'relu'  : nn.ReLU(),
                  'swiglu' : nn.SiLU()}

class TorchGLU(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        self.act = _supported_act[activation]

    def forward(self, x):
        shape = x.size(-1)
        a = x[..., :shape // 2]
        b = x[..., (shape // 2):]
        a = self.act(a)
        return a * b

class TorchLayerNormMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 eps: float = 1e-5, activation = 'gelu',
                 normalization: str = "LayerNorm"):
        super().__init__()
        if normalization == "LayerNorm":
            self.ln = nn.LayerNorm(hidden_size, eps=eps)
        elif normalization == "RMSNorm":
            self.ln = TorchRMSNorm(hidden_size, eps=eps)
        else:
            raise RuntimeError("Unsupported normalization")
        if 'glu' in activation:
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
    def __init__(self, hidden_size: int, eps: float, num_attention_heads: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps)
        self.causal_attn = TorchMHA(hidden_size, num_attention_heads)
        self.ln_mlp = TorchLayerNormMLP(hidden_size, 4 * hidden_size, eps)
        self.resid_attn_dropout = nn.Dropout(0.1)
        self.resid_mlp_dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = self.ln(x)
        b, _ = self.causal_attn(a, attn_mask)
        x = x + self.resid_attn_dropout(b)
        n = self.ln_mlp(x)
        x = x + self.resid_mlp_dropout(n)
        return x


def _test_e2e_selective_recompute(block, bs, dtype, config, recompute=False):
    reset_rng_states()

    te_inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    te_out = block(
        te_inp_hidden_states,
        te_inp_attn_mask,
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
def test_gpt_selective_activation_recompute(dtype, bs, model):
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
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
            get_rng_state_tracker=get_dummy_cuda_rng_tracker,
            params_dtype=dtype,
        )
        .cuda()
        .eval()
    )

    outputs = _test_e2e_selective_recompute(block, bs, dtype, config, recompute=False)
    outputs_recompute = _test_e2e_selective_recompute(block, bs, dtype, config, recompute=True)
    assert_all_equal(outputs, outputs_recompute)


def _test_e2e_full_recompute(block, bs, dtype, config, recompute=False):
    reset_rng_states()

    te_inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    if recompute:
        te_out = te_checkpoint(
            block,
            False,  # distribute_saved_activations
            get_dummy_cuda_rng_tracker,
            None,  # tp_group
            te_inp_hidden_states,
            te_inp_attn_mask,
            checkpoint_core_attention=False,
        )
    else:
        te_out = block(
            te_inp_hidden_states,
            te_inp_attn_mask,
            checkpoint_core_attention=False,
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
def test_gpt_full_activation_recompute(dtype, bs, model):
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
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
            get_rng_state_tracker=get_dummy_cuda_rng_tracker,
            params_dtype=dtype,
        )
        .cuda()
        .eval()
    )

    outputs = _test_e2e_full_recompute(block, bs, dtype, config, recompute=False)
    outputs_recompute = _test_e2e_full_recompute(block, bs, dtype, config, recompute=True)
    assert_all_equal(outputs, outputs_recompute)


def _test_e2e_checkpointing_get_model(config, dtype):
    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)
    return (
        TransformerLayer(
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
        )
        .cuda()
        .eval()
    )


def _test_e2e_checkpointing(bs, dtype, config, checkpoint=False, steps=10, path="checkpoint.pt"):
    reset_rng_states()

    te_inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    block = _test_e2e_checkpointing_get_model(config, dtype)

    for _ in range(steps // 2):
        te_out = block(
            te_inp_hidden_states,
            te_inp_attn_mask,
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

        del block
        block = _test_e2e_checkpointing_get_model(config, dtype)
        block.load_state_dict(torch.load(path))

        for p in block.parameters():
            if p.requires_grad:
                p.grad = param_grads.pop(0)

        assert not param_grads, "Oops!"

    for _ in range(steps // 2):
        te_out = block(
            te_inp_hidden_states,
            te_inp_attn_mask,
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
    outputs_recompute = _test_e2e_checkpointing(bs, dtype, config, checkpoint=True)
    assert_all_equal(outputs, outputs_recompute)


def _test_e2e_gpt_accuracy(block, bs, dtype, config):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    inp_hidden_states.retain_grad()
    inp_attn_mask = get_causal_attn_mask(config.seq_len)

    out = block(inp_hidden_states, inp_attn_mask)
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
def test_gpt_accuracy(dtype, bs, model):
    config = model_configs[model]

    te_gpt = (
        TransformerLayer(
            hidden_size=config.hidden_size,
            ffn_hidden_size=4 * config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            layernorm_epsilon=config.eps,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            fuse_qkv_params=True,
            qkv_weight_interleaved=False,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    torch_gpt = (
        TorchGPT(
            config.hidden_size,
            config.eps,
            config.num_attention_heads,
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


def _test_granular_accuracy(block, bs, dtype, config):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
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

    mask = torch.triu(torch.ones(config.seq_len, config.seq_len, device="cuda"), diagonal=1).bool()
    query, key, value = [
        torch.randn(config.seq_len, bs, config.num_attention_heads,
        config.embed, dtype=dtype, requires_grad=True).cuda() for _ in range(3)]

    query.retain_grad()
    key.retain_grad()
    value.retain_grad()

    out = block(query, key, value, mask)
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
            attention_dropout=0.1,  # dropout
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    torch_dpa = (
        TorchDotProductAttention(
            config.embed,
            0.1,  # dropout
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
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

    te_linear = (
        Linear(
            config.hidden_size,
            4 * config.hidden_size,
            bias=True,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    torch_linear = (
        torch.nn.Linear(
            config.hidden_size,
            4 * config.hidden_size,
            bias=True,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

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
def test_rmsnorm_accuracy(dtype, bs, model, eps):
    config = model_configs[model]

    te_rmsnorm = (
        RMSNorm(
            config.hidden_size,
            eps=eps,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    torch_rmsnorm = (
        TorchRMSNorm(
            config.hidden_size,
            eps=eps,
        )
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
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 1e-7)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 2e-2)

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("normalization", all_normalizations)
def test_layernorm_linear_accuracy(dtype, bs, model, normalization):
    config = model_configs[model]

    te_ln_linear = (
        LayerNormLinear(
            config.hidden_size,
            4 * config.hidden_size,
            config.eps,
            bias=True,
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    torch_ln_linear = (
        TorchLayerNormLinear(
            config.hidden_size,
            4 * config.hidden_size,
            config.eps,
            bias=True,
            normalization=normalization,
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
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_layernorm_mlp_accuracy(dtype, bs, model, activation, normalization):
    config = model_configs[model]

    te_ln_mlp = (
        LayerNormMLP(
            config.hidden_size,
            4 * config.hidden_size,
            activation=activation,
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

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
    static_input = torch.randn(config.seq_len, bs, config.hidden_size, device='cuda', dtype=dtype, requires_grad=True)
    static_target = torch.randn(config.seq_len, bs, config.hidden_size, device='cuda', dtype=dtype)

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

    block = (
        TransformerLayer(
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
        )
        .to(dtype=dtype)
        .cuda()
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
