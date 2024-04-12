# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
from typing import Optional
from contextlib import nullcontext

import torch
import pytest

from transformer_engine.pytorch.fp8 import fp8_autocast, FP8GlobalStateManager
from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
    is_bf16_compatible,
)
from transformer_engine.pytorch import (
    LayerNormLinear,
    Linear,
    LayerNormMLP,
    TransformerLayer,
    RMSNorm,
    LayerNorm,
    get_cpu_offload_context,
)
from transformer_engine.common import recipe

# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


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


@dataclass
class ModelConfig:
    """Transformer model configuration"""

    num_layers: int
    seq_len: int
    batch_size: int
    hidden_size: int
    num_attention_heads: int
    kv_channels: Optional[int] = None

    def is_fp8_supported(self):
        if self.seq_len * self.batch_size % 16:
            return False
        if self.hidden_size % 16:
            return False
        return True

model_configs = {
    "126m": ModelConfig(12, 2048, 2, 768, 12),
    "small": ModelConfig(2, 32, 2, 64, 2),
    "weird": ModelConfig(2, 37, 3, 69, 3),
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
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

all_boolean = [True, False]

all_activations = ["gelu", "relu", "reglu", "geglu", "swiglu"]
all_normalizations = ["LayerNorm", "RMSNorm"]

def _disable_wgrads(block):
    for p in block.parameters():
        p.requires_grad = False


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def _test_sanity_e2e_cuda_graph(block, dtype, config, fp8_recipe, skip_wgrad):
    # Initialize loss function and optimizer.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)

    # Placeholders used for capture.
    static_input = torch.randn(config.seq_len, config.batch_size, config.hidden_size, device='cuda', dtype=dtype, requires_grad=True)
    static_target = torch.randn(config.seq_len, config.batch_size, config.hidden_size, device='cuda', dtype=dtype)

    real_input = torch.rand_like(static_input)
    real_target = torch.rand_like(static_target)

    use_fp8 = fp8_recipe is not None
    if skip_wgrad:
        _disable_wgrads(block)

    # Pre graph capture warmup in a separate stream.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe, _graph=True):
                out = block(static_input)
            loss = loss_fn(out, static_target)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    # Capture.
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe, _graph=True):
            static_output = block(static_input)
        static_loss = loss_fn(static_output, static_target)
        static_loss.backward()
        optimizer.step()

    # Fills the graph's input memory with new data to compute on
    with torch.no_grad():
        static_input.copy_(real_input)
        static_target.copy_(real_target)
    g.replay()

    torch.cuda.synchronize()


def _test_sanity_e2e_amp(block, dtype, config, fp8_recipe, skip_wgrad):
    te_inp_hidden_states = torch.randn(
        config.seq_len, config.batch_size, config.hidden_size, dtype=torch.float32, requires_grad=True
    ).cuda()
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = torch.randint(2, (1, 1, config.seq_len, config.seq_len)).cuda().bool()

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with torch.autocast(device_type="cuda", enabled=True, dtype=dtype):
        with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
            te_out = block(te_inp_hidden_states, attention_mask=te_inp_attn_mask)
        loss = te_out.sum()

    loss.backward()
    torch.cuda.synchronize()

    assert te_out.dtype == dtype, "AMP wrong output type."
    assert te_inp_hidden_states.grad.dtype == torch.float32, "AMP wrong dgrad type."
    for name, p in block.named_parameters():
        if p.requires_grad:
            assert p.grad.dtype == torch.float32, f"AMP wrong wgrad type for {name}."


def _test_sanity_e2e_gradient_accumulation_fusion(block, dtype, config, fp8_recipe, skip_wgrad):
    te_inp_hidden_states = torch.randn(
        config.seq_len, config.batch_size, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    te_inp_attn_mask = torch.randint(2, (1, 1, config.seq_len, config.seq_len)).cuda().bool()

    if skip_wgrad:
        _disable_wgrads(block)

    for name, p in block.named_parameters():
        if "layer_norm_weight" in name:
            continue
        elif "weight" in name and p.requires_grad:
            p.main_grad = torch.zeros_like(p)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        te_out = block(te_inp_hidden_states, attention_mask=te_inp_attn_mask)
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

    for name, p in block.named_parameters():
        if "layer_norm_weight" in name:
            continue
        elif "weight" in name and p.requires_grad:
            assert torch.count_nonzero(p.main_grad) > 0, "Gradient not accumulated."


def _test_sanity_e2e(block, dtype, config, fp8_recipe, skip_wgrad, cpu_offload):
    te_inp_hidden_states = torch.randn(
        config.seq_len, config.batch_size, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()

    if skip_wgrad:
        _disable_wgrads(block)

    if cpu_offload:
        offload_context, sync_function = get_cpu_offload_context(enabled=True)
    else:
        offload_context = nullcontext()
        sync_function = lambda x: x

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe), offload_context:
        te_out = block(te_inp_hidden_states)
    te_out = sync_function(te_out)
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


def _test_sanity_e2e_bert(block, dtype, config, fp8_recipe, skip_wgrad):
    te_inp_hidden_states = torch.randn(
        config.seq_len, config.batch_size, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()

    te_inp_attn_mask = torch.rand(torch.Size([config.batch_size, 1, 1, config.seq_len])).cuda() > 0.5

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        te_out = block(te_inp_hidden_states, attention_mask=te_inp_attn_mask)
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


def _test_sanity_e2e_T5(block, dtype, config, fp8_recipe, skip_wgrad):
    te_inp_hidden_states = torch.randn(
        config.seq_len, config.batch_size, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    te_inp_attn_mask = torch.randint(2, (1, 1, config.seq_len, config.seq_len)).cuda().bool()
    enc_dec_attn_mask = torch.rand(torch.Size([config.batch_size, 1, 1, config.seq_len])).cuda() > 0.5

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        te_out = block(
            te_inp_hidden_states,
            attention_mask=te_inp_attn_mask,
            encoder_output=te_inp_hidden_states,
            enc_dec_attn_mask=enc_dec_attn_mask,
        )
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


def _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad):
    if skip_dgrad and skip_wgrad:
        pytest.skip("No gradient computation; Skipping to avoid PyTorch RuntimeError.")

    te_inp = torch.randn(
        config.seq_len, config.batch_size, config.hidden_size, dtype=dtype, requires_grad=not skip_dgrad
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


def _test_sanity_normalization_amp(block, dtype, config, skip_wgrad, skip_dgrad):
    if skip_dgrad and skip_wgrad:
        pytest.skip("No gradient computation; Skipping to avoid PyTorch RuntimeError.")

    te_inp = torch.randn(
        config.seq_len, config.batch_size, config.hidden_size, requires_grad=True
    ).cuda()
    te_inp.retain_grad()

    with torch.autocast(device_type="cuda", enabled=True, dtype=dtype):
        te_out = block(te_inp)
        loss = te_out.sum()
    loss.backward()

    torch.cuda.synchronize()

    assert te_out.dtype == dtype, "AMP wrong output type."
    assert te_inp.grad.dtype == torch.float32, "AMP wrong dgrad type."
    for name, p in block.named_parameters():
        if p.requires_grad:
            assert p.grad.dtype == torch.float32, f"AMP wrong wgrad type for {name}."


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("skip_dgrad", all_boolean)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_sanity_normalization_amp(dtype, model, skip_wgrad, skip_dgrad, normalization):
    config = model_configs[model]
    module = RMSNorm if normalization == "RMSNorm" else LayerNorm

    block = (
        module(config.hidden_size)
        .to(dtype=torch.float32)
        .cuda()
    )
    _test_sanity_normalization_amp(block, dtype, config, skip_wgrad, skip_dgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("skip_dgrad", all_boolean)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_sanity_layernorm_linear(dtype, fp8_recipe, model, skip_wgrad,
                                 zero_centered_gamma, skip_dgrad,
                                 normalization):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)

    block = (
        LayerNormLinear(
            config.hidden_size,
            config.hidden_size * 3,
            init_method=init_method,
            zero_centered_gamma=zero_centered_gamma,
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
    )
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("skip_dgrad", all_boolean)
def test_sanity_linear(dtype, fp8_recipe, model, skip_wgrad, skip_dgrad):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        Linear(
            config.hidden_size, config.hidden_size, init_method=output_layer_init_method
        )
        .to(dtype=dtype)
        .cuda()
    )
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("skip_dgrad", all_boolean)
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_sanity_layernorm_mlp(dtype, fp8_recipe, model, skip_wgrad,
                              zero_centered_gamma, skip_dgrad, activation,
                              normalization):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        LayerNormMLP(
            config.hidden_size,
            4 * config.hidden_size,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            zero_centered_gamma=zero_centered_gamma,
            activation=activation,
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
    )
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("bias", all_boolean)
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
@pytest.mark.parametrize("parallel_attention_mlp", all_boolean)
@pytest.mark.parametrize("cpu_offload", all_boolean)
def test_sanity_gpt(dtype, fp8_recipe, model, skip_wgrad,
                    zero_centered_gamma, bias, activation,
                    normalization, parallel_attention_mlp,
                    cpu_offload):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            zero_centered_gamma=zero_centered_gamma,
            bias=bias,
            activation=activation,
            normalization=normalization,
            parallel_attention_mlp=parallel_attention_mlp,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e(block, dtype, config, fp8_recipe, skip_wgrad, cpu_offload)


def test_sanity_gpt_126m():
    fp8_recipe = None
    if fp8_available:
        fp8_recipe = recipe.DelayedScaling(
            0,
            1,
            recipe.Format.E4M3,
            amax_history_len=16,
            amax_compute_algo="most_recent",
        )
    test_sanity_gpt(
        dtype=param_types[-1],
        fp8_recipe=fp8_recipe,
        model="126m",
        skip_wgrad=False,
        zero_centered_gamma=True,
        bias=True,
        activation="gelu",
        normalization="LayerNorm",
        parallel_attention_mlp=False,
        cpu_offload=False,
    )


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_sanity_bert(dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma,
                     normalization):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=True,
            output_layernorm=True,
            zero_centered_gamma=zero_centered_gamma,
            self_attn_mask_type="padding",
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e_bert(block, dtype, config, fp8_recipe, skip_wgrad)


def test_sanity_bert_126m():
    fp8_recipe = recipe.DelayedScaling(
        0,
        1,
        recipe.Format.E4M3,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )
    test_sanity_bert(
        dtype=param_types[-1],
        fp8_recipe=fp8_recipe,
        model="126m",
        skip_wgrad=False,
        zero_centered_gamma=False,
        normalization="LayerNorm",
    )


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_sanity_T5(dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma,
                   normalization):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            layer_type="decoder",
            zero_centered_gamma=zero_centered_gamma,
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e_T5(block, dtype, config, fp8_recipe, skip_wgrad)


def test_sanity_T5_126m():
    fp8_recipe = recipe.DelayedScaling(
        0,
        1,
        recipe.Format.E4M3,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )
    test_sanity_T5(
        dtype=param_types[-1],
        fp8_recipe=fp8_recipe,
        model="126m",
        skip_wgrad=False,
        zero_centered_gamma=False,
        normalization="LayerNorm",
    )


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
def test_sanity_amp_and_nvfuser(dtype, fp8_recipe, model, skip_wgrad):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
        )
        .to(dtype=torch.float32)
        .cuda()
    )

    _test_sanity_e2e_amp(block, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
def test_sanity_drop_path(dtype, fp8_recipe, model, skip_wgrad):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            drop_path_rate=1.0,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e(block, dtype, config, fp8_recipe, skip_wgrad, False)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
def test_sanity_fused_qkv_params(dtype, fp8_recipe, model, skip_wgrad):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            fuse_qkv_params=True,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e(block, dtype, config, fp8_recipe, skip_wgrad, False)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_sanity_gradient_accumulation_fusion(dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            zero_centered_gamma=zero_centered_gamma,
            fuse_qkv_params=True,
            fuse_wgrad_accumulation=True,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e_gradient_accumulation_fusion(block, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_gpt_cuda_graph(dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma,
                        normalization):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = (
        TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.kv_channels,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            zero_centered_gamma=zero_centered_gamma,
            fuse_qkv_params=True,
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
    )

    _test_sanity_e2e_cuda_graph(block, dtype, config, fp8_recipe, skip_wgrad)

def test_model_multiple_cast():
    a = torch.zeros((16,16)).cuda()
    m = Linear(16,32)

    y = m(a)
    assert y.dtype == torch.float32

    m.half()
    a = a.half()

    y2 = m(a)
    assert y2.dtype == torch.float16
