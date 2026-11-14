# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import nvdlfw_inspect.api as debug_api
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.debug.pytorch.debug_quantization import (
    DebugQuantizedTensor,
    DebugQuantizer,
    HIGH_PRECISION,
    STANDARD_QUANTIZE,
)
from transformer_engine.pytorch.module._common import (
    set_quantizer_usage_for_wgrad_all_gather,
)
from transformer_engine.pytorch.quantized_tensor import Quantizer
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer

from test_numerics import create_config_file

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)

B, S, H, D = 64, 64, 64, 64

model_keys = ["linear", "layernorm_linear", "layernorm_mlp", "mha_attention", "transformer_layer"]

configs = {
    "": "",
    "log": """log:
  layers:
    layer_types: [linear]
  enabled:
    True
  transformer_engine:
    LogTensorStats:
      enabled: True
      tensors: [activation, gradient, weight, output, wgrad, dgrad]
      stats: [min, max, mean, std, l1_norm, l2_norm, cur_amax, dynamic_range]
      start_step : 0
      end_step: 1
""",
    "log_fp8": """log_fp8:
  layers:
    layer_types: [linear]
  enabled:
    True
  transformer_engine:
    LogFp8TensorStats:
      enabled: True
      tensors: [activation, gradient, weight]
      stats: [underflows%]
      start_step : 0
      end_step: 1
""",
    "fake_quant": """
fake_quant_config:
  enabled: True
  layers:
    layer_types: [linear]
  transformer_engine:
    FakeQuant:
      enabled: True
      gemms: [fprop, dgrad, wgrad]
      tensors: [activation, weight, gradient]
      quant_format: FP8E5M2
""",
}


def _make_debug_quantizer_for_usage_test(parent_quantizer):
    """Construct a DebugQuantizer without initializing the debug API."""
    quantizer = object.__new__(DebugQuantizer)
    Quantizer.__init__(quantizer, rowwise=True, columnwise=True)
    quantizer.parent_quantizer = parent_quantizer
    quantizer.output_tensor = False
    if parent_quantizer is None:
        quantizer.rowwise_tensor_plan = HIGH_PRECISION
        quantizer.columnwise_tensor_plan = HIGH_PRECISION
    else:
        quantizer.rowwise_tensor_plan = STANDARD_QUANTIZE
        quantizer.columnwise_tensor_plan = STANDARD_QUANTIZE
    return quantizer


def test_wgrad_all_gather_usage_handles_debug_quantizer_without_parent():
    """High-precision debug mode has no parent quantizer to unwrap."""
    quantizer = _make_debug_quantizer_for_usage_test(None)

    set_quantizer_usage_for_wgrad_all_gather(quantizer)

    assert quantizer.rowwise_usage is False
    assert quantizer.columnwise_usage is True


def test_wgrad_all_gather_usage_updates_debug_wrapper_and_blockwise_parent():
    """Usage changes must remain synchronized across the debug wrapper and parent."""
    parent_quantizer = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
    )
    quantizer = _make_debug_quantizer_for_usage_test(parent_quantizer)

    set_quantizer_usage_for_wgrad_all_gather(quantizer)

    assert quantizer.rowwise_usage is False
    assert quantizer.columnwise_usage is True
    assert parent_quantizer.rowwise_usage is False
    assert parent_quantizer.columnwise_usage is True


def test_wgrad_all_gather_usage_detects_hybrid_through_debug_wrapper():
    """Hybrid classification may inspect the parent, but usage mutates the wrapper."""
    parent_quantizer = HybridQuantizer(
        rowwise_quantizer=IdentityQuantizer(),
        columnwise_quantizer=IdentityQuantizer(),
    )
    quantizer = _make_debug_quantizer_for_usage_test(parent_quantizer)

    set_quantizer_usage_for_wgrad_all_gather(quantizer)

    assert quantizer.rowwise_usage is False
    assert quantizer.columnwise_usage is True
    assert parent_quantizer.rowwise_usage is False
    assert parent_quantizer.columnwise_usage is True


def test_debug_quantized_tensor_routes_usage_to_distinct_representations():
    """A usage request should only reach the child for that GEMM direction."""
    rowwise = IdentityQuantizer()(torch.ones(2, 2))
    columnwise = IdentityQuantizer()(torch.ones(2, 2))
    rowwise_calls = []
    columnwise_calls = []

    def record_rowwise(rowwise_usage=None, columnwise_usage=None):
        rowwise_calls.append((rowwise_usage, columnwise_usage))

    def record_columnwise(rowwise_usage=None, columnwise_usage=None):
        columnwise_calls.append((rowwise_usage, columnwise_usage))

    rowwise.update_usage = record_rowwise
    columnwise.update_usage = record_columnwise
    tensor = DebugQuantizedTensor(rowwise, columnwise, quantizer=None)

    tensor.update_usage(columnwise_usage=True)
    assert rowwise_calls == []
    assert columnwise_calls == [(None, True)]

    tensor.update_usage(rowwise_usage=True)
    assert rowwise_calls == [(True, None)]
    assert columnwise_calls == [(None, True)]

    tensor.update_usage(rowwise_usage=False)
    assert tensor.rowwise_gemm_tensor is None
    with pytest.raises(RuntimeError, match="Cannot recreate rowwise tensor"):
        tensor.update_usage(rowwise_usage=True, columnwise_usage=False)
    assert tensor.columnwise_gemm_tensor is columnwise


def test_debug_quantized_tensor_updates_shared_representation_once():
    """A shared child receives one combined usage update."""
    shared = IdentityQuantizer()(torch.ones(2, 2))
    calls = []

    def record_usage(rowwise_usage=None, columnwise_usage=None):
        calls.append((rowwise_usage, columnwise_usage))

    shared.update_usage = record_usage
    tensor = DebugQuantizedTensor(shared, shared, quantizer=None)

    tensor.update_usage(rowwise_usage=True, columnwise_usage=False)
    assert calls == [(True, False)]


# Configs that require FP8 to be enabled
fp8_required_configs = {"log_fp8"}


def _get_model(model_key):
    if model_key == "linear":
        return te.Linear(D, D, name="layer")
    if model_key == "layernorm_linear":
        return te.LayerNormLinear(D, D, name="layer")
    if model_key == "layernorm_mlp":
        return te.LayerNormMLP(D, D, D, name="layer")
    if model_key == "mha_attention":
        return te.MultiheadAttention(D, H, name="layer")
    if model_key == "transformer_layer":
        return te.TransformerLayer(D, D, H, name="layer")


def _run_forward_backward(model, fp8):
    for _ in range(3):
        inp = torch.randn((S, B, H)).cuda()
        with te.autocast(enabled=fp8):
            out = model(inp)
        out.sum().backward()
        debug_api.step()


@create_config_file
def _run_test(model_key, fp8, config, feature_dirs, config_file, log_dir):
    try:
        if config != "":
            config_file.write(config)
            config_file.flush()
        config_file_name = config_file.name if config != "" else ""
        debug_api.initialize(feature_dirs=feature_dirs, config_file=config_file_name)
        model = _get_model(model_key)
        _run_forward_backward(model, fp8)
    except Exception as error:
        raise error
    finally:
        debug_api.end_debug()


@pytest.mark.parametrize("model_key", model_keys)
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("config_key", configs.keys())
def test_sanity_debug(model_key, fp8, config_key, feature_dirs):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if not fp8 and config_key in fp8_required_configs:
        pytest.skip(f"Config '{config_key}' requires FP8")
    _run_test(model_key, fp8, configs[config_key], feature_dirs)


@pytest.mark.skipif(not te.is_nvfp4_available(), reason="Requires NVFP4")
def test_batch_rowwise_debug_dispatch(monkeypatch, feature_dirs):
    """An enabled debug quantizer must retain inspection through grouped forward/backward."""
    from pathlib import Path
    from transformer_engine.common import recipe
    from transformer_engine.pytorch.module import _split_quantization
    from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    FP8GlobalStateManager.reset()
    torch.manual_seed(319)
    config = Path(__file__).parent / "test_configs/dummy_feature.yaml"
    debug_api.initialize(config_file=str(config), feature_dirs=feature_dirs)
    try:
        fp4_recipe = recipe.NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
            row_scaled_activation=True,
            backward_override="high_precision",
        )
        fp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
        fp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
        fp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
        models = [
            te.GroupedLinear(2, 128, 128, bias=True, params_dtype=torch.bfloat16) for _ in range(2)
        ]
        models[1].load_state_dict(models[0].state_dict())
        x = torch.randn(384, 128, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        seen_debug = []
        forward = _GroupedLinear.forward

        def record_forward(ctx, inp, args, *parameters):
            seen_debug.append(args.debug)
            return forward(ctx, inp, args, *parameters)

        monkeypatch.setattr(_GroupedLinear, "forward", record_forward)
        monkeypatch.setattr(
            _split_quantization,
            "_batch_nvfp4_rowwise",
            lambda *_: pytest.fail("Debug quantization must not be bypassed"),
            raising=False,
        )
        results, inputs = [], []
        for model in models:
            inp = x.clone().requires_grad_()
            inputs.append(inp)
            with te.autocast(recipe=fp4_recipe):
                out = model(inp, [128, 256])
            out.backward(dy)
            results.append(out)
        assert seen_debug == [True, True]
        torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
        for actual, expected in [
            (inputs[0], inputs[1]),
            *zip(models[0].parameters(), models[1].parameters()),
        ]:
            assert torch.isfinite(actual.grad).all()
            torch.testing.assert_close(actual.grad, expected.grad, atol=0, rtol=0)
    finally:
        debug_api.end_debug()
        FP8GlobalStateManager.reset()
