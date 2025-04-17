# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
from typing import Optional
from contextlib import nullcontext

import torch
import pytest
import os

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
    get_cudnn_version,
)
from transformer_engine.pytorch import (
    LayerNormLinear,
    Linear,
    GroupedLinear,
    LayerNormMLP,
    TransformerLayer,
    RMSNorm,
    LayerNorm,
    get_cpu_offload_context,
)
from transformer_engine.common import recipe
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
)
from transformer_engine.pytorch.tensor.utils import replace_raw_data
from transformer_engine.pytorch.distributed import checkpoint
from test_numerics import reset_rng_states, dtype_tols

# Only run FP8 tests on supported devices.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()


def create_meta(scale_factor: float, size: int = 1):
    meta = tex.FP8TensorMeta()
    meta.amax_history = torch.zeros(1, size, dtype=torch.float32, device="cuda")
    meta.scale_inv = torch.ones(size, dtype=torch.float32, device="cuda") / scale_factor
    meta.scale = torch.ones(size, dtype=torch.float32, device="cuda") * scale_factor
    return meta


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
    None,  # Test non-FP8
    recipe.MXFP8BlockScaling(),  # Test default
    recipe.Float8CurrentScaling(),  # Test default
    recipe.Float8BlockScaling(),  # Test default
    recipe.DelayedScaling(),  # Test default
    recipe.DelayedScaling(  # Test most_recent algo
        amax_history_len=16,
        amax_compute_algo="most_recent",
    ),
    recipe.DelayedScaling(  # Test custom amax and scale compute algo
        fp8_format=recipe.Format.E4M3,
        amax_compute_algo=custom_amax_compute,
        scaling_factor_compute_algo=custom_amax_to_scale,
    ),
]

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

all_boolean = [True, False]
batch_sizes_with_zero = [0, 1, 2]

all_activations = ["gelu", "relu", "reglu", "geglu", "swiglu", "srelu", "qgelu", "qgeglu"]
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
    assert te_inp_hidden_states.grad is not None, "Gradient should not be empty"
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

    failed_grads = []
    for name, p in block.named_parameters():
        if "layer_norm_weight" in name:
            continue
        elif "weight" in name and p.requires_grad:
            if not torch.count_nonzero(p.main_grad) > 0:
                failed_grads.append(name)
    assert len(failed_grads) == 0, f"Gradient not accumulated for {failed_grads}."


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


def _test_sanity_common(
    block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad, microbatching=True
):
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
        if not microbatching:
            te_out = block(te_inp)
        else:
            _ = block(te_inp, is_first_microbatch=True)
            te_out = block(te_inp, is_first_microbatch=False)
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
    assert te_inp.grad is not None, "Gradient should not be empty"
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
@pytest.mark.parametrize("microbatching", all_boolean)
def test_sanity_layernorm_linear(
    dtype,
    fp8_recipe,
    model,
    skip_wgrad,
    zero_centered_gamma,
    skip_dgrad,
    normalization,
    microbatching,
):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad, microbatching)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("skip_wgrad", all_boolean)
@pytest.mark.parametrize("skip_dgrad", all_boolean)
@pytest.mark.parametrize("microbatching", all_boolean)
def test_sanity_linear(dtype, fp8_recipe, model, skip_wgrad, skip_dgrad, microbatching):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad, microbatching)


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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    use_fp8 = fp8_recipe is not None
    with fp8_model_init(enabled=use_fp8 and fp8_model_params, recipe=fp8_recipe):
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
@pytest.mark.parametrize("bs", batch_sizes_with_zero)
@pytest.mark.parametrize("model", ["small", "weird"])
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
@pytest.mark.parametrize("use_bias", all_boolean)
@pytest.mark.parametrize("empty_split", ["first", "last", "middle"])
@pytest.mark.parametrize("num_gemms", [4])
def test_sanity_grouped_linear(
    dtype, bs, model, fp8_recipe, fp8_model_params, use_bias, num_gemms, empty_split
):
    config = model_configs[model]
    ffn_hidden_size = 4 * config.hidden_size
    # Small batch size used to catch bug from https://github.com/NVIDIA/TransformerEngine/pull/1527.
    bs = bs * 16
    num_tokens = bs * config.seq_len * (num_gemms - 1)

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    use_fp8 = fp8_recipe is not None
    with fp8_model_init(enabled=use_fp8 and fp8_model_params, recipe=fp8_recipe):
        te_grouped_linear = GroupedLinear(
            num_gemms, config.hidden_size, ffn_hidden_size, bias=use_bias, params_dtype=dtype
        ).cuda()

    inp_hidden_states = torch.randn(
        num_tokens, config.hidden_size, dtype=dtype, requires_grad=True
    ).cuda()
    m_splits = [bs * config.seq_len] * num_gemms
    if empty_split == "first":
        m_splits[0] = 0
    elif empty_split == "last":
        m_splits[-1] = 0
    elif empty_split == "middle":
        m_splits[num_gemms // 2] = 0

    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        out = te_grouped_linear(inp_hidden_states, m_splits)
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
@pytest.mark.parametrize("microbatching", all_boolean)
def test_sanity_layernorm_mlp(
    dtype,
    fp8_recipe,
    model,
    skip_wgrad,
    zero_centered_gamma,
    skip_dgrad,
    activation,
    normalization,
    microbatching,
):
    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad, microbatching)


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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
        self_attn_mask_type="causal",
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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
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
        if fp8_recipe.float8_block_scaling() and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        if fp8_recipe.mxfp8() and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
        if fp8_recipe.float8_block_scaling():
            pytest.skip("cuda graph not supported for float8_block_scaling recipe")
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

    _ = general_gemm(A=weight, B=inp, workspace=get_workspace())
    torch.cuda.synchronize()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("N", [32])
@pytest.mark.parametrize("datatype", [torch.float16, torch.bfloat16])
def test_sanity_fp8_gemm_with_unalignment(N, datatype):
    offset = 16
    scratchpad = torch.randn(N, N * N + offset, device="cuda", dtype=datatype)

    scales = torch.ones(1).cuda().squeeze()
    amaxes = torch.ones(1).cuda().squeeze()
    dtype = tex.DType.kFloat8E4M3
    fp8_quantizer = Float8Quantizer(scales, amaxes, dtype)

    outp_type = datatype

    scratchpad_fp8 = fp8_quantizer(scratchpad)
    inp_fp8 = torch.reshape(scratchpad_fp8[0][:-offset], (N, N))
    weight_fp8 = torch.reshape(scratchpad_fp8[0][offset:], (N, N))
    general_gemm(
        weight_fp8,
        inp_fp8,
        get_workspace(),
        outp_type,
        bias=None,
        use_split_accumulator=False,
    )
    torch.cuda.synchronize()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="FP8 tests require Hopper.")
@pytest.mark.skipif(get_cudnn_version() < (9, 3, 0), reason="cuDNN 9.3.0+ is required.")
@pytest.mark.parametrize("model", ["large"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sanity_attention_extra_state(model, dtype):
    config = model_configs[model]
    outputs = _run_attention_extra_state(dtype, config, checkpoint=False)
    outputs_checkpoint = _run_attention_extra_state(dtype, config, checkpoint=True)
    outputs_checkpoint_v1_6 = _run_attention_extra_state(
        dtype, config, mimic_v1_6=True, checkpoint=True
    )

    # Check that results match
    tols = dtype_tols(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        tols.update(dict(rtol=2e-2, atol=2e-3))
    for i, (ref, test) in enumerate(zip(outputs, outputs_checkpoint)):
        torch.testing.assert_close(
            test,
            ref,
            **tols,
        )
    for i, (ref, test) in enumerate(zip(outputs, outputs_checkpoint_v1_6)):
        torch.testing.assert_close(
            test,
            ref,
            **tols,
        )


def _run_attention_extra_state(dtype, config, checkpoint=False, mimic_v1_6=False):
    steps = 10
    path = "checkpoint.pt"
    fp8_enabled = True
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_dpa=fp8_enabled,
        fp8_mha=False,
    )

    reset_rng_states()
    hidden_states = torch.randn(
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    def get_model(dtype, config):
        sigma = 0.023
        init_method = init_method_normal(sigma)
        output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

        with fp8_model_init(enabled=fp8_enabled, recipe=fp8_recipe):
            block = TransformerLayer(
                config.hidden_size,
                4 * config.hidden_size,
                config.num_attention_heads,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                fuse_qkv_params=True,
                params_dtype=dtype,
                device="cuda",
            )
        return block

    block = get_model(dtype, config)
    for i in range(steps // 2):
        with fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe):
            output = block(hidden_states, None)
            loss = output.sum()
            loss.backward()

    if checkpoint:
        sd = block.state_dict()
        if mimic_v1_6:
            sd["self_attention.core_attention.fused_attention._extra_state"] = sd[
                "self_attention.core_attention._extra_state"
            ]
            del sd["self_attention.core_attention._extra_state"]
        torch.save(sd, path)

        param_grads = []
        for p in block.parameters():
            if p.requires_grad:
                param_grads.append(p.grad.clone())

        _cpu_rng_state_new = torch.get_rng_state()
        _cuda_rng_state_new = torch.cuda.get_rng_state()

        del block
        block = get_model(dtype, config)
        block.load_state_dict(torch.load(path, weights_only=False))
        torch.set_rng_state(_cpu_rng_state_new)
        torch.cuda.set_rng_state(_cuda_rng_state_new)

        for p in block.parameters():
            if p.requires_grad:
                p.grad = param_grads.pop(0)

        assert not param_grads, "Oops!"

    for i in range((steps + 1) // 2):
        with fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe):
            output = block(hidden_states, None)
            loss = output.sum()
            loss.backward()

    torch.cuda.synchronize()

    if os.path.exists(path):
        os.remove(path)

    outputs = [output, hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)

    return outputs


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_replace_raw_data_for_float8tensor():
    """Test the functionality of replace_raw_data"""
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)

    fp8_quantizer = Float8CurrentScalingQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, device="cuda")
    fp8_tensor = fp8_quantizer.make_empty([128, 128], dtype=torch.bfloat16, device="cuda")
    random_bf16_data = torch.randn(fp8_tensor.shape, dtype=torch.bfloat16, device="cuda")
    fp8_quantizer.update_quantized(random_bf16_data, fp8_tensor)

    attrs_to_check = ["_quantizer", "_fp8_dtype", "_scale_inv", "_transpose", "_transpose_invalid"]
    attrs = {}
    for attr in attrs_to_check:
        attrs[attr] = getattr(fp8_tensor, attr)

    old_data = fp8_tensor._data
    new_data = torch.empty_like(old_data)
    replace_raw_data(fp8_tensor, new_data)

    # Make sure the new_data is properly assigned.
    assert fp8_tensor._data.data_ptr() != old_data.data_ptr()
    assert fp8_tensor._data.data_ptr() == new_data.data_ptr()
    # Make sure the values are not changed.
    torch.testing.assert_close(old_data, fp8_tensor._data, atol=0, rtol=0)
    # Make sure other attributes are not changed (totally identical)
    for attr in attrs_to_check:
        assert id(getattr(fp8_tensor, attr)) == id(attrs[attr])


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_fp8_model_init_high_precision_init_val():
    """Test fp8_model_init with preserve_high_precision_init_val=True"""
    with fp8_model_init(preserve_high_precision_init_val=True):
        model = Linear(768, 768)

    weight = model.weight

    assert isinstance(weight, QuantizedTensor), "Weight should be QuantizedTensor"
    assert hasattr(weight, "_high_precision_init_val"), "_high_precision_init_val not found"
    assert hasattr(weight, "get_high_precision_init_val"), "get_high_precision_init_val() not found"
    assert hasattr(
        weight, "clear_high_precision_init_val"
    ), "clear_high_precision_init_val() not found"

    high_precision = weight.get_high_precision_init_val()
    assert high_precision.device.type == "cpu", "high_precision_init_val is not on the CPU"

    new_weight = weight._get_quantizer().make_empty(
        shape=weight.shape, dtype=weight.dtype, device=weight.device
    )
    weight._get_quantizer().update_quantized(high_precision.to(weight.device), new_weight)

    torch.testing.assert_close(
        new_weight.dequantize(dtype=weight.dtype),
        weight.dequantize(dtype=weight.dtype),
        rtol=0,
        atol=0,
    )

    weight.clear_high_precision_init_val()
    assert weight.get_high_precision_init_val() is None, "clear_high_precision_init_val() not work"
    assert not hasattr(
        weight, "._high_precision_init_val"
    ), "clear_high_precision_init_val() not work"


def test_sanity_checkpointing_on_callables():
    """Test that TE checkpointing works correctly on callable modules."""

    # torch.autograf.function
    class MyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            return inp

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    module = MyFunction.apply
    inp = torch.randn(10, 10, device="cuda", requires_grad=True)

    out_checkpoint = checkpoint(module, inp)
    out_checkpoint.sum().backward()
    grad_checkpoint = inp.grad

    out_standard = module(inp)
    out_standard.sum().backward()
    grad_standard = inp.grad

    # Assert that gradients are the same
    torch.testing.assert_close(grad_checkpoint, grad_standard)
