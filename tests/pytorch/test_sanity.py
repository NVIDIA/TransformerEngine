# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
from typing import Optional
from contextlib import nullcontext

import torch
import pytest
import io

from transformer_engine.pytorch.fp8 import (
    fp8_autocast,
    FP8GlobalStateManager,
    fp8_model_init,
)
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
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
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import (
    gemm,
    fp8_gemm,
    gelu,
    cast_to_fp8,
    cast_from_fp8,
)
from transformer_engine.pytorch.module.base import get_workspace
from test_onnx_export import create_meta

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
    "large": ModelConfig(1, 128, 2, 512, 4, 128),
}

fp8_recipes = [
    None,  # Handles non-FP8 case
    recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3),
    recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.HYBRID),
    recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,
        override_linear_precision=(False, False, True),
    ),
    recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,
        amax_history_len=16,
        amax_compute_algo="most_recent",
    ),
    recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,
        amax_history_len=16,
        amax_compute_algo="max",
    ),
    recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,
        amax_history_len=16,
        amax_compute_algo=custom_amax_compute,
    ),
    recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,
        amax_history_len=16,
        scaling_factor_compute_algo=custom_amax_to_scale,
    ),
]

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

all_boolean = [True, False]
batch_sizes_with_zero = [0, 1, 2]

all_activations = ["gelu", "relu", "reglu", "geglu", "swiglu", "srelu"]
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
    static_input = torch.randn(
        config.seq_len,
        config.batch_size,
        config.hidden_size,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    static_target = torch.randn(
        config.seq_len, config.batch_size, config.hidden_size, device="cuda", dtype=dtype
    )

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
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = torch.randint(
        2,
        (1, 1, config.seq_len, config.seq_len),
        dtype=torch.bool,
        device="cuda",
    )

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
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_attn_mask = torch.randint(
        2,
        (1, 1, config.seq_len, config.seq_len),
        dtype=torch.bool,
        device="cuda",
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
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

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
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    te_inp_attn_mask = torch.randint(
        2,
        (config.batch_size, 1, 1, config.seq_len),
        dtype=torch.bool,
        device="cuda",
    )

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
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_attn_mask = torch.randint(
        2,
        (1, 1, config.seq_len, config.seq_len),
        dtype=torch.bool,
        device="cuda",
    )

    enc_dec_attn_mask = torch.randint(
        2,
        (config.batch_size, 1, 1, config.seq_len),
        dtype=torch.bool,
        device="cuda",
    )

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
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=not skip_dgrad,
    )

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
        (config.seq_len, config.batch_size, config.hidden_size),
        device="cuda",
        requires_grad=True,
    )
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

    block = module(config.hidden_size).to(dtype=torch.float32).cuda()
    _test_sanity_normalization_amp(block, dtype, config, skip_wgrad, skip_dgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("skip_dgrad", all_boolean)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_sanity_layernorm_linear(
    dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma, skip_dgrad, normalization
):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)

    block = LayerNormLinear(
        config.hidden_size,
        config.hidden_size * 3,
        init_method=init_method,
        zero_centered_gamma=zero_centered_gamma,
        normalization=normalization,
        params_dtype=dtype,
        device="cuda",
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

    block = Linear(
        config.hidden_size,
        config.hidden_size,
        init_method=output_layer_init_method,
        params_dtype=dtype,
        device="cuda",
    )
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes_with_zero)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
@pytest.mark.parametrize("use_bias", all_boolean)
def test_sanity_linear_with_zero_tokens(dtype, bs, model, fp8_recipe, fp8_model_params, use_bias):
    config = model_configs[model]
    ffn_hidden_size = 4 * config.hidden_size
    num_tokens = bs * config.seq_len

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    use_fp8 = fp8_recipe is not None
    with fp8_model_init(enabled=use_fp8 and fp8_model_params):
        te_linear = Linear(
            config.hidden_size, ffn_hidden_size, bias=use_bias, params_dtype=dtype
        ).cuda()

    inp_hidden_states = torch.randn(
        num_tokens, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        out = te_linear(inp_hidden_states)
    loss = out.sum()
    loss.backward()
    assert out.shape == (num_tokens, ffn_hidden_size)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("skip_dgrad", all_boolean)
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_sanity_layernorm_mlp(
    dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma, skip_dgrad, activation, normalization
):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        zero_centered_gamma=zero_centered_gamma,
        activation=activation,
        normalization=normalization,
        params_dtype=dtype,
        device="cuda",
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
def test_sanity_gpt(
    dtype,
    fp8_recipe,
    model,
    skip_wgrad,
    zero_centered_gamma,
    bias,
    activation,
    normalization,
    parallel_attention_mlp,
    cpu_offload,
):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        zero_centered_gamma=zero_centered_gamma,
        bias=bias,
        activation=activation,
        normalization=normalization,
        device="cuda",
        parallel_attention_mlp=parallel_attention_mlp,
    )

    _test_sanity_e2e(block, dtype, config, fp8_recipe, skip_wgrad, cpu_offload)


def test_sanity_gpt_126m():
    fp8_recipe = None
    if fp8_available:
        fp8_recipe = recipe.DelayedScaling(
            margin=0,
            fp8_format=recipe.Format.E4M3,
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
def test_sanity_bert(dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma, normalization):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=True,
        output_layernorm=True,
        zero_centered_gamma=zero_centered_gamma,
        self_attn_mask_type="padding",
        normalization=normalization,
        device="cuda",
    )

    _test_sanity_e2e_bert(block, dtype, config, fp8_recipe, skip_wgrad)


def test_sanity_bert_126m():
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,
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
def test_sanity_T5(dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma, normalization):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        layer_type="decoder",
        zero_centered_gamma=zero_centered_gamma,
        normalization=normalization,
        device="cuda",
    )

    _test_sanity_e2e_T5(block, dtype, config, fp8_recipe, skip_wgrad)


def test_sanity_T5_126m():
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,
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

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=torch.float32,
        device="cuda",
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

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        drop_path_rate=1.0,
        device="cuda",
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

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        fuse_qkv_params=True,
        device="cuda",
    )

    _test_sanity_e2e(block, dtype, config, fp8_recipe, skip_wgrad, False)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_sanity_gradient_accumulation_fusion(
    dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma
):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        zero_centered_gamma=zero_centered_gamma,
        fuse_qkv_params=True,
        fuse_wgrad_accumulation=True,
        device="cuda",
    )

    _test_sanity_e2e_gradient_accumulation_fusion(block, dtype, config, fp8_recipe, skip_wgrad)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_gpt_cuda_graph(dtype, fp8_recipe, model, skip_wgrad, zero_centered_gamma, normalization):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.kv_channels,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        zero_centered_gamma=zero_centered_gamma,
        fuse_qkv_params=True,
        normalization=normalization,
        device="cuda",
    )

    _test_sanity_e2e_cuda_graph(block, dtype, config, fp8_recipe, skip_wgrad)


def test_model_multiple_cast():
    a = torch.zeros((16, 16), device="cuda")
    m = Linear(16, 32)

    y = m(a)
    assert y.dtype == torch.float32

    m.half()
    a = a.half()

    y2 = m(a)
    assert y2.dtype == torch.float16


@pytest.mark.parametrize("N", [32])
@pytest.mark.parametrize("offset", [1, 3, 5])
@pytest.mark.parametrize("datatype", param_types)
def test_sanity_gemm_with_unalignment(N, offset, datatype):
    scratchpad = torch.randn(N * N + 2 * offset, device="cuda", dtype=datatype)
    inp = torch.reshape(scratchpad[offset:-offset], (N, N))
    weight = torch.reshape(scratchpad[offset * 2 :], (N, N))

    _, _, _ = gemm(A=weight, B=inp, dtype=datatype, workspace=get_workspace())
    torch.cuda.synchronize()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("N", [32])
@pytest.mark.parametrize("datatype", [torch.float16, torch.bfloat16])
def test_sanity_fp8_gemm_with_unalignment(N, datatype):
    offset = 16
    scratchpad = torch.randn(N * N + offset, device="cuda", dtype=datatype)

    fp8_tensor_inp = tex.FP8FwdTensors.GEMM1_INPUT
    fp8_tensor_weight = tex.FP8FwdTensors.GEMM1_WEIGHT

    nb_inp_scales, nb_weight_scales = 1, N
    scale_factor = 1.0
    meta_inp = create_meta(scale_factor, nb_inp_scales)
    meta_weight = create_meta(scale_factor, nb_weight_scales)
    inp_type = tex.DType.kFloat8E4M3
    weights_type = tex.DType.kFloat8E4M3
    outp_type = datatype

    scratchpad_fp8 = cast_to_fp8(scratchpad, meta_weight, fp8_tensor_inp, inp_type)
    inp_fp8 = torch.reshape(scratchpad_fp8[:-offset], (N, N))
    weight_fp8 = torch.reshape(scratchpad_fp8[offset:], (N, N))
    _, _ = fp8_gemm(
        weight_fp8,
        meta_weight.scale_inv,
        fp8_tensor_weight,
        inp_type,
        inp_fp8,
        meta_inp.scale_inv,
        fp8_tensor_inp,
        weights_type,
        outp_type,
        get_workspace(),
        bias=None,
        use_bias=False,
        use_split_accumulator=False,
    )
    torch.cuda.synchronize()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.skipif(get_device_compute_capability() != (9, 0), reason="FP8 tests require Hopper.")
@pytest.mark.parametrize("model", ["large"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sanity_attention_extra_state(model, dtype):
    config = model_configs[model]
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_dpa=True,
        fp8_mha=False,
    )
    hidden_states = torch.randn(
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    with fp8_model_init(enabled=True):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            fuse_qkv_params=True,
            params_dtype=dtype,
            device="cuda",
        )
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        output = block(hidden_states, is_first_microbatch=True)
        loss = output.sum()
        loss.backward()

    # call state_dict()
    sd = block.state_dict()

    # check core_attention._extra_state
    attn_extra_state = sd["self_attention.core_attention._extra_state"]
    attn_extra_state.seek(0)
    attn_extra_state = torch.load(attn_extra_state, map_location="cuda")

    # add random core_attention.fused_attention._extra_state
    # it should not be loaded or cause any 'unexpected key' errors
    random_state = {"a": 1, "b": 2}
    fused_attn_extra_state = io.BytesIO()
    torch.save(random_state, fused_attn_extra_state)
    sd["self_attention.core_attention.fused_attention._extra_state"] = fused_attn_extra_state

    # save checkpoint
    path = "./checkpoint.pt"
    torch.save(sd, path)

    # reinit the model
    del block
    with fp8_model_init(enabled=True):
        block_new = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            fuse_qkv_params=True,
            params_dtype=dtype,
            device="cuda",
        )
    FP8GlobalStateManager.reset()

    # load from checkpoint
    block_new.load_state_dict(torch.load(path))

    # check state_dict
    sd_new = block_new.state_dict()
    attn_extra_state_new = sd_new["self_attention.core_attention._extra_state"]
    attn_extra_state_new.seek(0)
    attn_extra_state_new = torch.load(attn_extra_state_new, map_location="cuda")
    for k, v in attn_extra_state_new.items():
        if k != "extra_fp8_variables":
            assert torch.equal(v, attn_extra_state[k]), f"{k} is not equal"
        else:
            for ek, ev in attn_extra_state_new["extra_fp8_variables"].items():
                assert ev == attn_extra_state["extra_fp8_variables"][ek], f"{ek} is not equal"
