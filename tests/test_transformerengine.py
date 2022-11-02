# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import pytest

from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
)
from transformer_engine.pytorch import (
    LayerNormLinear,
    Linear,
    LayerNormMLP,
    TransformerLayer,
)

class ModelConfig:
    def __init__(
        self, hidden_size, eps, num_attention_heads, embed, num_layers, seq_len
    ):
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

skip_wgrad = [True, False]

def _disable_wgrads(block):
    for p in block.parameters():
            p.requires_grad = False

def _test_sanity_e2e_amp(block, bs, dtype, config, skip_wgrad):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return

    te_inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=torch.float32, requires_grad=True
    ).cuda()

    te_inp_attn_mask = (
        torch.rand(
            (
                1,
                1,
                config.seq_len,
                config.seq_len,
            )
        )
        .cuda()
        .bool()
    )

    if (skip_wgrad):
        _disable_wgrads(block)

    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        te_out = block(te_inp_hidden_states, te_inp_attn_mask)
        loss = te_out.sum()

    assert te_out.dtype == dtype
    loss.backward()
    torch.cuda.synchronize()


def _test_sanity_e2e(block, bs, dtype, config, skip_wgrad):
    te_inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    te_inp_attn_mask = (
        torch.rand(
            (
                1,
                1,
                config.seq_len,
                config.seq_len,
            )
        )
        .cuda()
        .bool()
    )

    if (skip_wgrad):
        _disable_wgrads(block)

    te_out = block(te_inp_hidden_states, te_inp_attn_mask)
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


def _test_sanity_e2e_T5(block, bs, dtype, config, skip_wgrad):
    te_inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    te_inp_attn_mask = (
        torch.rand(
            (
                1,
                1,
                config.seq_len,
                config.seq_len,
            )
        )
        .cuda()
        .bool()
    )

    if (skip_wgrad):
        _disable_wgrads(block)

    te_out = block(
        te_inp_hidden_states, te_inp_attn_mask, encoder_output=te_inp_hidden_states
    )
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


def _test_sanity_common(block, bs, dtype, config, skip_wgrad):
    te_inp = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()

 
    if (skip_wgrad):
        _disable_wgrads(block)

    te_out = block(te_inp)
    if isinstance(te_out, tuple):
        te_out = te_out[0]
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_layernorm_linear(dtype, bs, model, skip_wgrad):
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)

    block = (
        LayerNormLinear(
            config.hidden_size,
            config.hidden_size * 3,
            eps=config.eps,
            init_method=init_method,
        )
        .to(dtype=dtype)
        .cuda()
    )
    _test_sanity_common(block, bs, dtype, config, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_linear(dtype, bs, model, skip_wgrad):
    config = model_configs[model]

    sigma = 0.023
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        Linear(
            config.hidden_size, config.hidden_size, init_method=output_layer_init_method
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_common(block, bs, dtype, config, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_layernorm_mlp(dtype, bs, model, skip_wgrad):
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        LayerNormMLP(
            config.hidden_size,
            4 * config.hidden_size,
            eps=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )
        .to(dtype=dtype)
        .cuda()
    )
    _test_sanity_common(block, bs, dtype, config, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_gpt(dtype, bs, model, skip_wgrad):
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

    _test_sanity_e2e(block, bs, dtype, config, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_bert(dtype, bs, model, skip_wgrad):
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
            apply_residual_connection_post_layernorm=True,
            output_layernorm=True,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e(block, bs, dtype, config, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_T5(dtype, bs, model, skip_wgrad):
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
            layer_type="decoder",
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e_T5(block, bs, dtype, config, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_amp_and_nvfuser(dtype, bs, model, skip_wgrad):
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
        )
        .to(dtype=torch.float32)
        .cuda()
    )

    _test_sanity_e2e_amp(block, bs, dtype, config, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_drop_path(dtype, bs, model, skip_wgrad):
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
            drop_path_rate=1.0,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e(block, bs, dtype, config, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", skip_wgrad)
def test_sanity_fused_qkv_params(dtype, bs, model, skip_wgrad):
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
            fuse_qkv_params=True,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e(block, bs, dtype, config, skip_wgrad)
