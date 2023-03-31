# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import pytest

from transformer_engine.pytorch.fp8 import fp8_autocast
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
from transformer_engine.common import recipe

# Only run FP8 tests on H100.
fp8_available = torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9


def custom_amax_to_scale(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: torch.Tensor,
    recipe: recipe.DelayedScaling,
) -> torch.Tensor:
    """Custom func to test recipe."""
    sf = fp8_max / amax
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)

    return sf


def custom_amax_compute(amax_history: torch.Tensor) -> torch.Tensor:
    """Custom func to test recipe."""
    return torch.min(amax_history, dim=0).values


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

fp8_recipes = [
    None, # Handles non-FP8 case
    recipe.DelayedScaling(0, 1, recipe.Format.E4M3),
    recipe.DelayedScaling(0, 1, recipe.Format.HYBRID),
    recipe.DelayedScaling(
        0, 1, recipe.Format.E4M3, override_linear_precision=(False, False, True)
    ),
    recipe.DelayedScaling(
        0, 1, recipe.Format.E4M3, amax_history_len=16, amax_compute_algo="most_recent"
    ),
    recipe.DelayedScaling(
        0, 1, recipe.Format.E4M3, amax_history_len=16, amax_compute_algo="max"
    ),
    recipe.DelayedScaling(
        0,
        1,
        recipe.Format.E4M3,
        amax_history_len=16,
        amax_compute_algo=custom_amax_compute,
    ),
    recipe.DelayedScaling(
        0,
        1,
        recipe.Format.E4M3,
        amax_history_len=16,
        scaling_factor_compute_algo=custom_amax_to_scale,
    ),
]

param_types = [torch.float32, torch.float16]
if torch.cuda.is_bf16_supported():
    param_types.append(torch.bfloat16)

batch_sizes = [1, 2]

all_boolean = [True, False]


def _disable_wgrads(block):
    for p in block.parameters():
        p.requires_grad = False


def _test_sanity_e2e_amp(block, bs, dtype, config, fp8_recipe, skip_wgrad):
    te_inp_hidden_states = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=torch.float32, requires_grad=True
    ).cuda()
    te_inp_hidden_states.retain_grad()
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

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with torch.autocast(device_type="cuda", enabled=True, dtype=dtype):
        with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
            te_out = block(te_inp_hidden_states, te_inp_attn_mask)
        loss = te_out.sum()

    loss.backward()
    torch.cuda.synchronize()

    assert te_out.dtype == dtype, "AMP wrong output type."
    assert te_inp_hidden_states.grad.dtype == torch.float32, "AMP wrong dgrad type."
    for name, p in block.named_parameters():
        if p.requires_grad:
            assert p.grad.dtype == torch.float32, f"AMP wrong wgrad type for {name}."


def _test_sanity_e2e_gradient_accumulation_fusion(block, bs, dtype, config, fp8_recipe, skip_wgrad):
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

    if skip_wgrad:
        _disable_wgrads(block)

    for name, p in block.named_parameters():
        if "layer_norm_weight" in name:
            continue
        elif "weight" in name and p.requires_grad:
            p.main_grad = torch.zeros_like(p)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        te_out = block(te_inp_hidden_states, te_inp_attn_mask)
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

    for name, p in block.named_parameters():
        if "layer_norm_weight" in name:
            continue
        elif "weight" in name and p.requires_grad:
            assert (
                p.grad is None and torch.count_nonzero(p.main_grad) > 0
            ), "Gradient not accumulated."


def _test_sanity_e2e(block, bs, dtype, config, fp8_recipe, skip_wgrad):
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

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        te_out = block(te_inp_hidden_states, te_inp_attn_mask)
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


def _test_sanity_e2e_T5(block, bs, dtype, config, fp8_recipe, skip_wgrad):
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

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        te_out = block(
            te_inp_hidden_states, te_inp_attn_mask, encoder_output=te_inp_hidden_states
        )
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


def _test_sanity_common(block, bs, dtype, config, fp8_recipe, skip_wgrad):
    te_inp = torch.randn(
        config.seq_len, bs, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        te_out = block(te_inp)
    if isinstance(te_out, tuple):
        te_out = te_out[0]
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_sanity_layernorm_linear(dtype, bs, fp8_recipe, model, skip_wgrad, zero_centered_gamma):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)

    block = (
        LayerNormLinear(
            config.hidden_size,
            config.hidden_size * 3,
            eps=config.eps,
            init_method=init_method,
            zero_centered_gamma=zero_centered_gamma,
        )
        .to(dtype=dtype)
        .cuda()
    )
    _test_sanity_common(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
def test_sanity_linear(dtype, bs, fp8_recipe, model, skip_wgrad):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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
    _test_sanity_common(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_sanity_layernorm_mlp(dtype, bs, fp8_recipe, model, skip_wgrad, zero_centered_gamma):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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
            zero_centered_gamma=zero_centered_gamma,
        )
        .to(dtype=dtype)
        .cuda()
    )
    _test_sanity_common(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_sanity_gpt(dtype, bs, fp8_recipe, model, skip_wgrad, zero_centered_gamma):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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
            zero_centered_gamma=zero_centered_gamma,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_sanity_bert(dtype, bs, fp8_recipe, model, skip_wgrad, zero_centered_gamma):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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
            zero_centered_gamma=zero_centered_gamma,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_sanity_T5(dtype, bs, fp8_recipe, model, skip_wgrad, zero_centered_gamma):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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
            zero_centered_gamma=zero_centered_gamma,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e_T5(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
def test_sanity_amp_and_nvfuser(dtype, bs, fp8_recipe, model, skip_wgrad):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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

    _test_sanity_e2e_amp(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
def test_sanity_drop_path(dtype, bs, fp8_recipe, model, skip_wgrad):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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

    _test_sanity_e2e(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
def test_sanity_fused_qkv_params(dtype, bs, fp8_recipe, model, skip_wgrad):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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

    _test_sanity_e2e(block, bs, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_sanity_gradient_accumulation_fusion(dtype, bs, fp8_recipe, model, skip_wgrad, zero_centered_gamma):
    if fp8_recipe is not None and not fp8_available:
        pytest.skip("FP8 device not available.")

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
            zero_centered_gamma=zero_centered_gamma,
            fuse_qkv_params=True,
            fuse_wgrad_accumulation=True,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e_gradient_accumulation_fusion(block, bs, dtype, config, fp8_recipe, skip_wgrad)
