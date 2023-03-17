# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import contextlib
from typing import List, Optional
import pytest

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager

from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
)
from transformer_engine.pytorch import Linear, LayerNormLinear, TransformerLayer
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

param_types = [torch.float32, torch.bfloat16, torch.float16]

batch_sizes = [1, 2]

all_boolean = [True, False]


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
        assert torch.allclose(t1, t2, atol=atol), "Outputs not close enough."


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


class TorchLayerNormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, eps: float, bias: bool = True):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features, eps=eps)
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


class TorchMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class TorchGPT(nn.Module):
    def __init__(self, hidden_size: int, eps: float, num_attention_heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=eps)
        self.causal_attn = TorchMHA(hidden_size, num_attention_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = TorchMLP(hidden_size)
        self.resid_attn_dropout = nn.Dropout(0.1)
        self.resid_mlp_dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = self.ln_1(x)
        b, _ = self.causal_attn(a, attn_mask)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_2(x)
        n = self.mlp(m)
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
        torch_gpt.ln_1.weight = Parameter(
            te_gpt.self_attention.layernorm_qkv.layer_norm_weight.clone()
        )
        torch_gpt.ln_1.bias = Parameter(te_gpt.self_attention.layernorm_qkv.layer_norm_bias.clone())
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
        torch_gpt.ln_2.weight = Parameter(te_gpt.layernorm_mlp.layer_norm_weight.clone())
        torch_gpt.ln_2.bias = Parameter(te_gpt.layernorm_mlp.layer_norm_bias.clone())
        torch_gpt.mlp.fc1.weight = Parameter(te_gpt.layernorm_mlp.fc1_weight.clone())
        torch_gpt.mlp.fc1.bias = Parameter(te_gpt.layernorm_mlp.fc1_bias.clone())
        torch_gpt.mlp.fc2.weight = Parameter(te_gpt.layernorm_mlp.fc2_weight.clone())
        torch_gpt.mlp.fc2.bias = Parameter(te_gpt.layernorm_mlp.fc2_bias.clone())

    te_outputs = _test_e2e_gpt_accuracy(te_gpt, bs, dtype, config)
    torch_outputs = _test_e2e_gpt_accuracy(torch_gpt, bs, dtype, config)
    # Check output.
    if dtype == torch.bfloat16:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)


def _test_e2e_accuracy(block, bs, dtype, config):
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

    te_outputs = _test_e2e_accuracy(te_linear, bs, dtype, config)
    torch_outputs = _test_e2e_accuracy(torch_linear, bs, dtype, config)
    assert_allclose(te_outputs, torch_outputs, 1e-5)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
def test_layernorm_linear_accuracy(dtype, bs, model):
    config = model_configs[model]

    te_ln_linear = (
        LayerNormLinear(
            config.hidden_size,
            4 * config.hidden_size,
            config.eps,
            bias=True,
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
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_ln_linear.layernorm.weight = Parameter(te_ln_linear.layer_norm_weight.clone())
        torch_ln_linear.layernorm.bias = Parameter(te_ln_linear.layer_norm_bias.clone())
        torch_ln_linear.linear.weight = Parameter(te_ln_linear.weight.clone())
        torch_ln_linear.linear.bias = Parameter(te_ln_linear.bias.clone())

    te_outputs = _test_e2e_accuracy(te_ln_linear, bs, dtype, config)
    torch_outputs = _test_e2e_accuracy(torch_ln_linear, bs, dtype, config)
    # Check output.
    if dtype == torch.bfloat16:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
