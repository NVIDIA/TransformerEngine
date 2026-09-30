# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import dataclasses
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    NVFP4BlockScaling,
)
from transformer_engine.pytorch import is_fp8_available, is_nvfp4_available
from transformer_engine.pytorch import distributed as te_distributed
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.graph import make_graphed_callables
from transformer_engine.pytorch.module import GroupedLinear, LayerNormLinear, LayerNormMLP, Linear
from transformer_engine.pytorch.module import _common
from transformer_engine.pytorch.module import grouped_linear
from transformer_engine.pytorch.module import layernorm_mlp
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    FP8GlobalState,
    TEAutocastState,
    QuantizationCalibrationConfig,
    autocast,
    fp8_autocast,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

nvfp4_available, reason_for_no_nvfp4 = is_nvfp4_available(return_reason=True)
fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
DEFAULT_CALIBRATION_CONFIG = QuantizationCalibrationConfig()


@pytest.fixture(autouse=True)
def reset_quantization_state():
    yield
    FP8GlobalStateManager.reset()


def _make_test_quantizer(quantizer_cls):
    """Construct a quantizer without allocating recipe-specific CUDA state."""
    quantizer = object.__new__(quantizer_cls)
    Quantizer.__init__(quantizer, rowwise=True, columnwise=False)
    return quantizer


def _make_test_quantized_storage(**metadata):
    """Construct bare quantized storage carrying only the requested metadata."""
    storage = QuantizedTensorStorage()
    for name, value in metadata.items():
        setattr(storage, name, value)
    return storage


def test_calibration_api_additions_preserve_existing_parameter_order():
    state_fields = [field.name for field in dataclasses.fields(FP8GlobalState)]
    assert state_fields[-1] == "calibration_config"
    assert state_fields[:2] == ["fp8_enabled", "fp8_calibration"]

    assert list(inspect.signature(FP8GlobalStateManager.autocast_enter).parameters) == [
        "enabled",
        "calibrating",
        "fp8_recipe",
        "fp8_group",
        "_graph",
        "calibration_config",
    ]
    assert list(inspect.signature(autocast).parameters) == [
        "enabled",
        "calibrating",
        "recipe",
        "amax_reduction_group",
        "_graph",
        "calibration_config",
    ]
    assert list(inspect.signature(fp8_autocast).parameters) == [
        "enabled",
        "calibrating",
        "fp8_recipe",
        "fp8_group",
        "_graph",
        "calibration_config",
    ]
    assert list(inspect.signature(make_graphed_callables).parameters)[-2:] == [
        "capture_time_hooks",
        "calibration_config",
    ]
    assert not inspect.signature(FP8GlobalStateManager.get_autocast_state).parameters
    assert list(inspect.signature(FP8GlobalStateManager.set_autocast_state).parameters) == ["state"]
    autocast_state = FP8GlobalStateManager.get_autocast_state()
    assert isinstance(autocast_state, TEAutocastState)
    assert [field.name for field in dataclasses.fields(autocast_state)] == [
        "fp8_enabled",
        "fp8_calibration",
        "calibration_config",
        "fp8_recipe",
        "fp8_distributed_group",
        "is_first_fp8_module",
        "fp8_graph_capturing",
    ]


def test_activation_recompute_detection_uses_te_marker(monkeypatch):
    monkeypatch.setattr(
        te_distributed,
        "in_fp8_activation_recompute_phase",
        lambda: True,
    )

    assert _common._is_in_activation_recompute_phase()


def test_calibrating_argument_enables_default_calibration_config():
    assert FP8GlobalStateManager.get_calibration_config() is None
    with autocast(enabled=False, calibrating=True):
        assert FP8GlobalStateManager.quantization_state.fp8_calibration
        assert FP8GlobalStateManager.get_calibration_config() == QuantizationCalibrationConfig()
    assert FP8GlobalStateManager.get_calibration_config() is None


def test_explicit_calibration_config_is_active_in_autocast():
    config = QuantizationCalibrationConfig(transformer_engine_calibration_decay=0.5)
    with autocast(enabled=False, calibration_config=config):
        assert FP8GlobalStateManager.quantization_state.fp8_calibration
        assert FP8GlobalStateManager.get_calibration_config() is config
        assert FP8GlobalStateManager.get_autocast_state().calibration_config is config


def test_nested_autocast_restores_custom_calibration_config():
    config = QuantizationCalibrationConfig(transformer_engine_calibration_decay=0.5)
    with autocast(enabled=False, calibration_config=config):
        with autocast(enabled=False):
            assert FP8GlobalStateManager.get_calibration_config() is None
        assert FP8GlobalStateManager.get_calibration_config() is config


def test_global_calibration_boolean_enables_default_config():
    qstate = FP8GlobalStateManager.quantization_state
    qstate.fp8_calibration = True
    try:
        assert FP8GlobalStateManager.get_calibration_config() == QuantizationCalibrationConfig()
    finally:
        qstate.fp8_calibration = False


def test_autocast_enter_preserves_calibrating_boolean_api():
    saved_state = FP8GlobalStateManager.get_autocast_state()
    try:
        FP8GlobalStateManager.autocast_enter(False, True)
        assert FP8GlobalStateManager.is_fp8_calibration()
        assert FP8GlobalStateManager.get_calibration_config() == QuantizationCalibrationConfig()
    finally:
        FP8GlobalStateManager.autocast_exit(False, False)
        FP8GlobalStateManager.set_autocast_state(saved_state)


def test_calibrating_argument_accepts_explicit_calibration_config():
    config = QuantizationCalibrationConfig(transformer_engine_calibration_decay=0.5)
    with autocast(
        enabled=False,
        calibrating=True,
        calibration_config=config,
    ):
        assert FP8GlobalStateManager.get_calibration_config() is config


def test_row_scaled_nvfp4_calibration_fails_at_autocast_entry():
    recipe = NVFP4BlockScaling(row_scaled_activation=True)

    with pytest.raises(NotImplementedError, match="Row-wise NVFP4 calibration"):
        with autocast(enabled=False, calibrating=True, recipe=recipe):
            pass


def test_calibration_config_rejects_negative_decay():
    with pytest.raises(
        ValueError,
        match="transformer_engine_calibration_decay must be non-negative",
    ):
        QuantizationCalibrationConfig(transformer_engine_calibration_decay=-0.1)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize(
    ("module_name", "expected_buffer_count"),
    (
        ("linear", 2),
        ("layernorm_linear", 2),
        ("layernorm_mlp", 4),
        ("grouped_linear", 4),
    ),
)
@pytest.mark.parametrize("enabled", (False, True))
def test_calibration_config_registers_module_scaling_factor_buffers(
    module_name, expected_buffer_count, enabled
):
    module_kwargs = {
        "params_dtype": torch.bfloat16,
        "device": "cuda",
        "bias": False,
    }
    if module_name == "linear":
        module = Linear(32, 32, **module_kwargs)
    elif module_name == "layernorm_linear":
        module = LayerNormLinear(32, 32, **module_kwargs)
    elif module_name == "layernorm_mlp":
        module = LayerNormMLP(32, 32, **module_kwargs)
    else:
        module = GroupedLinear(2, 32, 32, use_grouped_tensor=False, **module_kwargs)

    inp = torch.randn((16, 32), dtype=torch.bfloat16, device="cuda")
    with autocast(
        enabled=enabled,
        recipe=Float8CurrentScaling(),
        calibration_config=QuantizationCalibrationConfig(),
    ):
        if module_name == "grouped_linear":
            module(inp, [8, 8])
        else:
            module(inp)

    buffers = {
        name: value for name, value in module.named_buffers() if name.endswith("_te_ptq_calibrated")
    }
    assert len(buffers) == expected_buffer_count
    assert all(torch.isfinite(value).all() for value in buffers.values())


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_linear_calibration_config_applies_activation_decay_only():
    module = Linear(32, 32, params_dtype=torch.bfloat16, device="cuda", bias=False)
    calibration_config = QuantizationCalibrationConfig(transformer_engine_calibration_decay=0.5)
    buffer_suffix = "_scale_inv_float8currentscaling_te_ptq_calibrated"

    with autocast(
        enabled=False,
        recipe=Float8CurrentScaling(),
        calibration_config=calibration_config,
    ):
        module(torch.full((16, 32), 448.0, dtype=torch.bfloat16, device="cuda"))

    torch.testing.assert_close(
        module.get_buffer(f"input{buffer_suffix}"), torch.ones(1, device="cuda")
    )
    weight_scale = module.get_buffer(f"weight{buffer_suffix}").clone()

    with autocast(
        enabled=False,
        recipe=Float8CurrentScaling(),
        calibration_config=calibration_config,
    ):
        module(torch.full((16, 32), 112.0, dtype=torch.bfloat16, device="cuda"))

    torch.testing.assert_close(
        module.get_buffer(f"input{buffer_suffix}"), torch.full((1,), 0.5, device="cuda")
    )
    torch.testing.assert_close(module.get_buffer(f"weight{buffer_suffix}"), weight_scale)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("use_reentrant", (False, True))
def test_external_activation_recomputation_does_not_update_calibration(use_reentrant):
    module = Linear(32, 32, params_dtype=torch.bfloat16, device="cuda", bias=False)
    calibration_config = QuantizationCalibrationConfig(transformer_engine_calibration_decay=0.5)
    recipe = Float8CurrentScaling()
    buffer_name = "input_scale_inv_float8currentscaling_te_ptq_calibrated"

    with (
        torch.no_grad(),
        autocast(enabled=False, recipe=recipe, calibration_config=calibration_config),
    ):
        module(torch.full((16, 32), 448.0, dtype=torch.bfloat16, device="cuda"))

    def checkpointed_forward(inp):
        with autocast(enabled=False, recipe=recipe, calibration_config=calibration_config):
            return module(inp)

    inp = torch.full(
        (16, 32),
        56.0,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    out = checkpoint(checkpointed_forward, inp, use_reentrant=use_reentrant)
    scale_after_forward = module.get_buffer(buffer_name).clone()
    torch.testing.assert_close(scale_after_forward, torch.full((1,), 0.5, device="cuda"))

    out.sum().backward()

    torch.testing.assert_close(module.get_buffer(buffer_name), scale_after_forward)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_layernorm_mlp_recomputation_does_not_update_calibration(monkeypatch):
    recomputation_states = []
    is_in_recompute = _common._is_in_activation_recompute_phase

    def record_recomputation_state():
        state = is_in_recompute()
        recomputation_states.append(state)
        return state

    monkeypatch.setattr(
        layernorm_mlp,
        "_is_in_activation_recompute_phase",
        record_recomputation_state,
    )
    module = LayerNormMLP(
        32,
        32,
        params_dtype=torch.bfloat16,
        device="cuda",
        bias=False,
        activation="relu",
        checkpoint=True,
    )
    calibration_config = QuantizationCalibrationConfig(transformer_engine_calibration_decay=0.5)
    torch.manual_seed(123)
    calibration_input = torch.randn((16, 32), dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        module.layer_norm_weight.fill_(448.0)
    with (
        torch.no_grad(),
        autocast(
            enabled=False,
            recipe=Float8CurrentScaling(),
            calibration_config=calibration_config,
        ),
    ):
        module(calibration_input)

    buffer_name = "fc1_input_scale_inv_float8currentscaling_te_ptq_calibrated"
    previous_scale = module.get_buffer(buffer_name).clone()
    with torch.no_grad():
        module.layer_norm_weight.fill_(56.0)
    inp = calibration_input.detach().clone().requires_grad_()

    with autocast(
        enabled=False,
        recipe=Float8CurrentScaling(),
        calibration_config=calibration_config,
    ):
        out = module(inp)

    calibration_buffers = {
        name: value.clone()
        for name, value in module.named_buffers()
        if name.endswith("_te_ptq_calibrated")
    }
    assert calibration_buffers
    torch.testing.assert_close(
        calibration_buffers[buffer_name],
        previous_scale * 0.5,
    )

    out.sum().backward()

    assert any(recomputation_states)
    for name, value in calibration_buffers.items():
        torch.testing.assert_close(module.get_buffer(name), value)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_linear_calibration_config_buffers_delayed_scaling_amax():
    module = Linear(32, 32, params_dtype=torch.bfloat16, device="cuda", bias=False)

    with autocast(
        enabled=False,
        recipe=DelayedScaling(),
        calibration_config=QuantizationCalibrationConfig(),
    ):
        module(torch.full((16, 32), 2.0, dtype=torch.bfloat16, device="cuda"))

    calibration_buffers = {
        name: value for name, value in module.named_buffers() if name.endswith("_te_ptq_calibrated")
    }
    assert set(calibration_buffers) == {
        "input_amax_delayedscaling_te_ptq_calibrated",
        "weight_amax_delayedscaling_te_ptq_calibrated",
    }
    torch.testing.assert_close(
        calibration_buffers["input_amax_delayedscaling_te_ptq_calibrated"],
        torch.full((1,), 2.0, device="cuda"),
    )


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_calibrating_boolean_registers_scaling_factor_buffers():
    module = Linear(32, 32, params_dtype=torch.bfloat16, device="cuda", bias=False)
    inp = torch.randn((16, 32), dtype=torch.bfloat16, device="cuda")

    with autocast(
        enabled=False,
        calibrating=True,
        recipe=Float8CurrentScaling(),
    ):
        module(inp)

    assert (
        len([name for name, _ in module.named_buffers() if name.endswith("_te_ptq_calibrated")])
        == 2
    )


@pytest.mark.parametrize(
    ("recipe", "recipe_name", "metadata_name", "expected_value"),
    (
        ("fp8_current_scaling", "float8currentscaling", "scale_inv", 0.25),
        ("fp8_delayed_scaling", "delayedscaling", "amax", 448.0),
        ("nvfp4", "nvfp4blockscaling", "amax", 2688.0),
    ),
)
def test_scale_buffer_info_selects_recipe_metadata(
    recipe, recipe_name, metadata_name, expected_value
):
    tensor = _make_test_quantized_storage(
        _scale_inv=torch.tensor([0.25], dtype=torch.float32),
        _amax_rowwise=torch.tensor([2688.0 if recipe == "nvfp4" else 1344.0], dtype=torch.float32),
    )
    if recipe == "fp8_delayed_scaling":
        quantizer = _make_test_quantizer(Float8Quantizer)
        quantizer.amax = torch.tensor([0.0], dtype=torch.float32)
        tensor = torch.tensor([expected_value])
    elif recipe == "fp8_current_scaling":
        quantizer = _make_test_quantizer(Float8CurrentScalingQuantizer)
    else:
        quantizer = _make_test_quantizer(NVFP4Quantizer)
        quantizer.row_scaled_nvfp4 = False

    quantizer.calibrate(tensor, calibration_config=DEFAULT_CALIBRATION_CONFIG)
    buffers = quantizer._get_calibration_metadata_buffers("input")
    buffer_name = f"input_{metadata_name}_{recipe_name}_te_ptq_calibrated"
    value = buffers[buffer_name]

    torch.testing.assert_close(value, torch.tensor([expected_value]))
    assert value is quantizer._calibration_state[metadata_name]


def test_row_scaled_nvfp4_calibration_and_metadata_buffering_are_unsupported():
    quantizer = _make_test_quantizer(NVFP4Quantizer)
    quantizer.row_scaled_nvfp4 = True
    tensor = _make_test_quantized_storage(
        _amax_rowwise=torch.tensor([1344.0], dtype=torch.float32)
    )

    with pytest.raises(NotImplementedError, match="Row-wise NVFP4 calibration"):
        quantizer.calibrate(tensor, calibration_config=DEFAULT_CALIBRATION_CONFIG)

    quantizer._calibration_state["amax_rowwise"] = tensor._amax_rowwise
    with pytest.raises(
        NotImplementedError,
        match="Row-wise NVFP4 calibration metadata buffering",
    ):
        quantizer._get_calibration_metadata_buffers("input")


def test_mxfp8_calibration_skips_non_global_scaling_metadata():
    tensor = SimpleNamespace(_rowwise_scale_inv=torch.ones(2, 2))
    quantizer = _make_test_quantizer(MXFP8Quantizer)

    quantizer.calibrate(tensor, calibration_config=DEFAULT_CALIBRATION_CONFIG)
    assert not quantizer._get_calibration_metadata_buffers("input")


def test_float8_blockwise_calibration_is_unsupported():
    quantizer = _make_test_quantizer(Float8BlockQuantizer)

    with pytest.raises(NotImplementedError, match="does not support calibration"):
        quantizer.calibrate(
            torch.ones(1),
            calibration_config=DEFAULT_CALIBRATION_CONFIG,
        )


def test_custom_quantizer_defaults_to_no_calibration_metadata():
    quantizer = Quantizer(rowwise=True, columnwise=False)

    assert not quantizer._get_calibration_metadata_buffers("input")


def test_hybrid_quantizer_calibration_is_unsupported():
    quantizer = HybridQuantizer(
        rowwise_quantizer=IdentityQuantizer(),
        columnwise_quantizer=IdentityQuantizer(),
    )

    with pytest.raises(NotImplementedError, match="calibrate\\(\\) is not supported"):
        quantizer.calibrate(
            torch.ones(1),
            calibration_config=DEFAULT_CALIBRATION_CONFIG,
        )


def test_quantizer_calibration_state_is_keyed_by_quantized_metadata():
    quantizer = Quantizer(rowwise=True, columnwise=False)

    quantizer._update_calibration_value(
        "amax", torch.tensor([2.0]), calibration_config=DEFAULT_CALIBRATION_CONFIG
    )
    quantizer._update_calibration_value(
        "scale_inv", torch.tensor([0.5]), calibration_config=DEFAULT_CALIBRATION_CONFIG
    )

    assert set(quantizer._calibration_state) == {"amax", "scale_inv"}
    torch.testing.assert_close(quantizer._calibration_state["amax"], torch.tensor([2.0]))
    torch.testing.assert_close(quantizer._calibration_state["scale_inv"], torch.tensor([0.5]))


def test_grouped_calibration_metadata_buffers_are_per_gemm():
    inputs = [
        _make_test_quantized_storage(_scale_inv=torch.tensor([0.25])),
        _make_test_quantized_storage(_scale_inv=torch.tensor([0.5])),
    ]
    weights = [
        _make_test_quantized_storage(_scale_inv=torch.tensor([0.75])),
        _make_test_quantized_storage(_scale_inv=torch.tensor([1.0])),
    ]
    input_quantizers = [
        _make_test_quantizer(Float8CurrentScalingQuantizer),
        _make_test_quantizer(Float8CurrentScalingQuantizer),
    ]
    weight_quantizers = [
        _make_test_quantizer(Float8CurrentScalingQuantizer),
        _make_test_quantizer(Float8CurrentScalingQuantizer),
    ]
    calibration_buffers = {}

    grouped_linear._calibrate_grouped_tensors(
        inputs,
        weights,
        input_quantizers,
        weight_quantizers,
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )
    grouped_linear._update_grouped_calibration_metadata_buffers(
        calibration_buffers,
        input_quantizers,
        weight_quantizers,
    )

    assert set(calibration_buffers) == {
        "input_gemm0_scale_inv_float8currentscaling_te_ptq_calibrated",
        "input_gemm1_scale_inv_float8currentscaling_te_ptq_calibrated",
        "weight_gemm0_scale_inv_float8currentscaling_te_ptq_calibrated",
        "weight_gemm1_scale_inv_float8currentscaling_te_ptq_calibrated",
    }
    torch.testing.assert_close(
        calibration_buffers["input_gemm1_scale_inv_float8currentscaling_te_ptq_calibrated"],
        torch.tensor([0.5]),
    )
    assert (
        calibration_buffers["input_gemm1_scale_inv_float8currentscaling_te_ptq_calibrated"]
        is input_quantizers[1]._calibration_state["scale_inv"]
    )


def test_grouped_calibration_applies_decay_only_to_activations():
    input_quantizer = _make_test_quantizer(Float8CurrentScalingQuantizer)
    input_quantizer._calibration_state = {"scale_inv": torch.tensor([4.0])}
    weight_quantizer = _make_test_quantizer(Float8CurrentScalingQuantizer)
    weight_quantizer._calibration_state = {"scale_inv": torch.tensor([4.0])}

    grouped_linear._calibrate_grouped_tensors(
        [_make_test_quantized_storage(_scale_inv=torch.tensor([1.0]))],
        [_make_test_quantized_storage(_scale_inv=torch.tensor([1.0]))],
        [input_quantizer],
        [weight_quantizer],
        calibration_config=QuantizationCalibrationConfig(
            transformer_engine_calibration_decay=0.5
        ),
    )

    torch.testing.assert_close(input_quantizer._calibration_state["scale_inv"], torch.tensor([2.0]))
    torch.testing.assert_close(
        weight_quantizer._calibration_state["scale_inv"], torch.tensor([1.0])
    )


def test_grouped_calibration_metadata_uses_per_gemm_delayed_scaling_amax():
    quantizers = []
    for amax in (1.0, 2.0):
        quantizer = _make_test_quantizer(Float8Quantizer)
        quantizer.amax = torch.tensor([amax])
        quantizers.append(quantizer)
    calibration_buffers = {}

    grouped_linear._calibrate_grouped_tensors(
        [torch.tensor([1.0]), torch.tensor([2.0])],
        [torch.tensor([1.0]), torch.tensor([2.0])],
        quantizers,
        quantizers,
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )
    grouped_linear._update_grouped_calibration_metadata_buffers(
        calibration_buffers,
        quantizers,
        quantizers,
    )

    torch.testing.assert_close(
        calibration_buffers["input_gemm0_amax_delayedscaling_te_ptq_calibrated"],
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(
        calibration_buffers["input_gemm1_amax_delayedscaling_te_ptq_calibrated"],
        torch.tensor([2.0]),
    )


@pytest.mark.parametrize(
    ("observed_scale", "expected_scale"),
    (
        # Decayed max is greater than the observed.
        (1.0, 2.0),
        # Decayed max is less than the observed.
        (3.0, 3.0),
    ),
)
def test_activation_scale_buffer_uses_decaying_maximum(observed_scale, expected_scale):
    name = "fc1_input_scale_inv_float8currentscaling_te_ptq_calibrated"
    quantizer = _make_test_quantizer(Float8CurrentScalingQuantizer)
    initial_buffer = torch.tensor([4.0])
    quantizer._calibration_state = {"scale_inv": initial_buffer}
    quantizer.calibrate(
        _make_test_quantized_storage(_scale_inv=torch.tensor([observed_scale])),
        calibration_config=QuantizationCalibrationConfig(
            transformer_engine_calibration_decay=0.5
        ),
    )
    buffers = quantizer._get_calibration_metadata_buffers("fc1_input")
    value = buffers[name]

    torch.testing.assert_close(value, torch.tensor([expected_scale]))
    assert value is initial_buffer
    assert value is quantizer._calibration_state["scale_inv"]


def test_zero_decay_stores_sanitized_metadata_copy():
    observed_scale = torch.tensor([2.0])
    quantizer = _make_test_quantizer(Float8CurrentScalingQuantizer)

    quantizer.calibrate(
        _make_test_quantized_storage(_scale_inv=observed_scale),
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )

    calibrated_scale = quantizer._calibration_state["scale_inv"]
    torch.testing.assert_close(calibrated_scale, observed_scale)
    assert calibrated_scale.data_ptr() != observed_scale.data_ptr()


def test_decaying_calibration_rejects_metadata_shape_change():
    quantizer = _make_test_quantizer(Float8CurrentScalingQuantizer)
    quantizer._calibration_state = {"scale_inv": torch.ones(1)}

    with pytest.raises(RuntimeError, match="calibration value shape changed"):
        quantizer.calibrate(
            _make_test_quantized_storage(_scale_inv=torch.ones(2)),
            calibration_config=QuantizationCalibrationConfig(
                transformer_engine_calibration_decay=0.5
            ),
        )


@pytest.mark.parametrize(
    ("transformer_engine_calibration_decay", "initial_scale", "expected_scale"),
    (
        (0.0, None, 0.0),
        (0.0, 4.0, 0.0),
        (0.5, None, 0.0),
        (0.5, 4.0, 4.0),
    ),
)
def test_nan_activation_scale_is_sanitized_before_export(
    transformer_engine_calibration_decay, initial_scale, expected_scale
):
    quantizer = _make_test_quantizer(Float8CurrentScalingQuantizer)
    if initial_scale is not None:
        quantizer._calibration_state = {"scale_inv": torch.tensor([initial_scale])}

    quantizer.calibrate(
        _make_test_quantized_storage(_scale_inv=torch.tensor([float("nan")])),
        calibration_config=QuantizationCalibrationConfig(
            transformer_engine_calibration_decay=transformer_engine_calibration_decay
        ),
    )
    result = quantizer._get_calibration_metadata_buffers("fc1_input")

    value = result["fc1_input_scale_inv_float8currentscaling_te_ptq_calibrated"]
    torch.testing.assert_close(value, torch.tensor([expected_scale]))


def test_current_scaling_calibrates_from_high_precision_tensor():
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=DType.kFloat8E4M3,
        device=torch.device("cpu"),
    )

    quantizer.calibrate(
        torch.tensor([-112.0, 224.0]),
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )
    value = quantizer._calibration_state["scale_inv"]

    torch.testing.assert_close(value, torch.tensor([0.5]))


@pytest.mark.parametrize("input_value", (0.0, float("inf")))
@pytest.mark.parametrize("force_pow_2_scales", (False, True))
def test_current_scaling_calibration_handles_non_finite_scale(input_value, force_pow_2_scales):
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=DType.kFloat8E4M3,
        device=torch.device("cpu"),
        force_pow_2_scales=force_pow_2_scales,
    )

    quantizer.calibrate(
        torch.tensor([input_value]),
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )

    torch.testing.assert_close(quantizer._calibration_state["scale_inv"], torch.ones(1))


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("fp8_dtype", (DType.kFloat8E4M3, DType.kFloat8E5M2))
def test_delayed_scaling_high_precision_calibration_matches_quantization(fp8_dtype):
    torch.manual_seed(123)
    tensor = torch.randn((32, 32), dtype=torch.bfloat16, device="cuda")

    active_quantizer = Float8Quantizer(
        scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=False,
    )
    quantized_tensor = active_quantizer(tensor)
    active_quantizer.calibrate(
        quantized_tensor,
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )

    calibration_quantizer = Float8Quantizer(
        scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=False,
    )
    calibration_quantizer.calibrate(
        tensor,
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )
    assert (
        calibration_quantizer.copy()._calibration_state is calibration_quantizer._calibration_state
    )

    torch.testing.assert_close(
        active_quantizer._calibration_state["amax"],
        active_quantizer.amax,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        calibration_quantizer._calibration_state["amax"],
        active_quantizer.amax,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("fp8_dtype", (DType.kFloat8E4M3, DType.kFloat8E5M2))
@pytest.mark.parametrize("force_pow_2_scales", (False, True))
@pytest.mark.parametrize("amax_epsilon", (0.0, 4.0))
def test_current_scaling_high_precision_calibration_matches_quantization(
    fp8_dtype, force_pow_2_scales, amax_epsilon
):
    torch.manual_seed(123)
    tensor = torch.randn((32, 32), dtype=torch.bfloat16, device="cuda")
    quantizer_kwargs = {
        "fp8_dtype": fp8_dtype,
        "device": torch.device("cuda"),
        "rowwise": True,
        "columnwise": False,
        "force_pow_2_scales": force_pow_2_scales,
        "amax_epsilon": amax_epsilon,
    }

    active_quantizer = Float8CurrentScalingQuantizer(**quantizer_kwargs)
    quantized_tensor = active_quantizer(tensor)
    active_quantizer.calibrate(
        quantized_tensor,
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )

    calibration_quantizer = Float8CurrentScalingQuantizer(**quantizer_kwargs)
    calibration_quantizer.calibrate(
        tensor,
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )

    torch.testing.assert_close(
        active_quantizer._calibration_state["scale_inv"],
        quantized_tensor._scale_inv,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        calibration_quantizer._calibration_state["scale_inv"],
        quantized_tensor._scale_inv,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
@pytest.mark.parametrize(
    ("with_rht", "with_post_rht_amax", "with_random_sign_mask"),
    (
        (False, False, False),
        (True, True, False),
        (True, True, True),
    ),
)
def test_nvfp4_high_precision_calibration_matches_quantization(
    with_rht, with_post_rht_amax, with_random_sign_mask
):
    torch.manual_seed(123)
    tensor = torch.randn((32, 32), dtype=torch.bfloat16, device="cuda")
    quantizer_kwargs = {
        "rowwise": True,
        "columnwise": True,
        "with_rht": with_rht,
        "with_post_rht_amax": with_post_rht_amax,
        "row_scaled_nvfp4": False,
        "with_random_sign_mask": with_random_sign_mask,
    }

    active_quantizer = NVFP4Quantizer(**quantizer_kwargs)
    quantized_tensor = active_quantizer(tensor)
    active_quantizer.calibrate(
        quantized_tensor,
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )
    calibration_quantizer = NVFP4Quantizer(**quantizer_kwargs)
    calibration_quantizer.calibrate(
        tensor,
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )
    assert (
        calibration_quantizer.copy()._calibration_state is calibration_quantizer._calibration_state
    )

    active_amax = active_quantizer._calibration_state["amax"]
    calibrated_amax = calibration_quantizer._calibration_state["amax"]
    torch.testing.assert_close(
        active_amax,
        quantized_tensor._amax_rowwise,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        calibrated_amax,
        quantized_tensor._amax_rowwise,
        atol=0.0,
        rtol=0.0,
    )


def test_shallow_quantizer_copy_shares_calibration_state():
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=DType.kFloat8E4M3,
        device=torch.device("cpu"),
    )
    copied_quantizer = quantizer.copy()
    copied_quantizer.calibrate(
        torch.tensor([-112.0, 224.0]),
        calibration_config=DEFAULT_CALIBRATION_CONFIG,
    )
    value = copied_quantizer._calibration_state["scale_inv"]

    assert quantizer._calibration_state["scale_inv"] is value
