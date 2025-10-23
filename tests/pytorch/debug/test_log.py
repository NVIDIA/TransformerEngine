# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import nvdlfw_inspect.api as debug_api
import transformer_engine.debug
import transformer_engine.pytorch as te
import torch
import tempfile
from transformer_engine.common import recipe
import pytest
import contextlib
import os
from transformer_engine.pytorch import (
    is_fp8_available,
    is_mxfp8_available,
    is_fp8_block_scaling_available,
    is_nvfp4_available,
)
from transformer_engine.pytorch.quantization import RecipeState
from transformer_engine.debug.pytorch.debug_state import TEDebugState


fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = is_fp8_block_scaling_available(
    return_reason=True
)
nvfp4_available, reason_for_no_nvfp4 = is_nvfp4_available(return_reason=True)

LOG_QUANTIZED_CONFIG_BASE = """
log:
  layers:
    layer_name_regex_pattern: .*
  enabled:
    True
  transformer_engine:
    LogFp8TensorStats:
      enabled: True
      stats: [
        {stats}
      ]
      tensors: [activation, gradient, weight]
      freq: 2
      start_step: 0
      end_step: 10
"""
recipes = [
    "fp8_delayed_scaling",
    "fp8_current_scaling",
    "fp8_block_scaling",
    "mxfp8",
]

bare_stats = [
    "underflows%",
    "scale_inv_min",
    "scale_inv_max",
    "mse",
]

all_stats = []

for r in recipes:
    for stat in bare_stats:
        for columnwise_postfix in ["", "_columnwise"]:
            if (
                r in ["fp8_current_scaling", "fp8_block_scaling"]
                and torch.cuda.get_device_capability()[0] < 9
            ):
                # hopper is needed for current-scaling, block-scaling
                continue
            if r == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
                # blackwell is needed for mxfp8
                continue
            if (
                r in ["fp8_delayed_scaling", "fp8_current_scaling"]
                and columnwise_postfix == "_columnwise"
            ):
                # columnwise stats are not supported for fp8_delayed_scaling and fp8_current_scaling
                continue

            all_stats.append(f"{r}_{stat}{columnwise_postfix}")

all_stats.append("fp8_delayed_scaling_overflows%")  # only delayed-scaling supports overflows%


@contextlib.contextmanager
def debug_session(config_str: str, feature_dirs):
    """
    Helper context manager that
    1. writes the YAML `config_str` to a temporary file,
    2. starts a debug session, and
    3. yields the directory that contains the statistics log.

    The session is closed automatically – even on exceptions – so every test
    stays concise and leak-free.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False
    ) as cfg_file, tempfile.TemporaryDirectory() as log_dir:
        cfg_file.write(config_str)
        cfg_file.flush()

        debug_api.initialize(
            config_file=cfg_file.name,
            feature_dirs=feature_dirs,
            log_dir=log_dir,
        )
        try:
            yield log_dir
        finally:
            debug_api.end_debug()


def read_log(log_dir: str) -> str:
    """Return the content of the statistics log produced by `debug_session`."""
    stat_path = os.path.join(
        log_dir,
        "nvdlfw_inspect_statistics_logs",
        "nvdlfw_inspect_globalrank-0.log",
    )
    with open(stat_path, "r") as f:
        return f.read()


def test_sanity(feature_dirs):
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    log_all_stats_config = LOG_QUANTIZED_CONFIG_BASE.format(stats=", ".join(all_stats))
    with debug_session(log_all_stats_config, feature_dirs) as log_dir:
        model = te.Linear(128, 128, params_dtype=torch.bfloat16)
        inp = torch.zeros(128, 128, dtype=torch.bfloat16).cuda()

        for _ in range(10):
            with te.autocast(recipe=recipe.DelayedScaling()):
                output = model(inp)
            loss = output.sum()
            loss.backward()
            debug_api.step()

        output = read_log(log_dir)

    assert output, "Output is empty"
    for stat in all_stats:
        assert stat in output, f"Stat {stat} not found in output"


fp8_recipes = [
    recipe.MXFP8BlockScaling(),
    recipe.DelayedScaling(),
    recipe.Float8CurrentScaling(),
    recipe.Float8BlockScaling(),
]


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
def test_numerics(fp8_recipe, feature_dirs):
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if not mxfp8_available and fp8_recipe == recipe.MXFP8BlockScaling():
        pytest.skip(reason_for_no_mxfp8)
    if not fp8_block_scaling_available and fp8_recipe == recipe.Float8BlockScaling():
        pytest.skip(reason_for_no_fp8_block_scaling)

    log_only_bare_stats_config = LOG_QUANTIZED_CONFIG_BASE.format(stats=", ".join(bare_stats))

    with debug_session(log_only_bare_stats_config, feature_dirs) as log_dir:
        recipe_state = RecipeState.create(
            fp8_recipe,
            mode="forward",
            num_quantizers=3,
        )

        tensor = torch.randn(1024, 1024).cuda()
        tensor[0, 100:200] = -0.0
        quantizer = recipe_state.make_quantizers()[0]
        quantized_tensor = quantizer(tensor)

        debug_api.transformer_engine.inspect_tensor(
            layer_name="layer_name",
            tensor_name="activation",
            iteration=0,
            tp_group=None,
            tensor=tensor,
            quantizer=quantizer,
            rowwise_quantized_tensor=quantized_tensor,
            columnwise_quantized_tensor=quantized_tensor,
        )
        debug_api.step()

        dequantized_tensor = quantized_tensor.dequantize()
        output = read_log(log_dir)

    for line in output.splitlines():
        if "underflows%" in line:
            underflows = float(line.split("value=")[1])
            expected = (
                ((dequantized_tensor == 0).sum() - (tensor == 0).sum()) / tensor.numel() * 100
            )
            assert underflows == pytest.approx(expected.cpu(), abs=1e-4)
        if "mse" in line:
            mse = float(line.split("value=")[1])
            expected = torch.nn.functional.mse_loss(dequantized_tensor, tensor, reduction="mean")
            assert mse == pytest.approx(expected.cpu(), abs=1e-4)
        if "overflows%" in line:
            overflows = float(line.split("value=")[1])
            expected = (
                (abs(dequantized_tensor) > abs(tensor)).sum() / dequantized_tensor.numel() * 100
            )
            assert overflows == pytest.approx(expected.cpu(), abs=1e-4)


@pytest.mark.parametrize("layer", ["linear", "transformer"])
def test_log_every_3_or_5_layers(layer, configs_dir, feature_dirs):
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    # If layer does not invoke any feature in current iteration,
    # then it changed into non-debug mode.
    # This test checks whether this works correctly -
    # non-quantized statistics should be logged every 3 iterations,
    # and quantized statistics should be logged every 5 iterations.
    with tempfile.TemporaryDirectory() as temp_dir:
        debug_api.initialize(
            config_file=configs_dir + "/log_config.yaml",
            feature_dirs=feature_dirs,
            log_dir=temp_dir,
        )

        if layer == "linear":
            model = te.Linear(128, 128, name="linear1")
        elif layer == "transformer":
            model = te.TransformerLayer(128, 128, 4, name="transformer1")
        else:
            raise ValueError(f"Invalid layer: {layer}")

        for i in range(20):
            x = torch.randn(4, 128, 128).cuda()
            with te.autocast(enabled=True):
                y = model(x)
            y.sum().backward()
            debug_api.step()

        with open(
            os.path.join(
                temp_dir, "nvdlfw_inspect_statistics_logs/nvdlfw_inspect_globalrank-0.log"
            ),
            "r",
        ) as f:
            file_content = f.read()
            for i in range(1, 20):
                if i % 3 == 0 or i % 5 == 0:
                    assert f"iteration={i:06d}" in file_content
                else:
                    assert f"iteration={i:06d}" not in file_content

    debug_api.end_debug()
    TEDebugState._reset()


# NVFP4 tests
LOG_NVFP4_CONFIG_BASE = """
log:
  layers:
    layer_name_regex_pattern: .*
  enabled:
    True
  transformer_engine:
    LogNvfp4TensorStats:
      enabled: True
      stats: [
        {stats}
      ]
      tensors: [activation, gradient, weight]
      freq: 2
      start_step: 0
      end_step: 10
"""


def test_nvfp4_numeric(feature_dirs):
    """Test that NVFP4 underflows% and MSE stats are computed correctly with known values."""
    if not nvfp4_available:
        pytest.skip(reason_for_no_nvfp4)

    log_nvfp4_config = LOG_NVFP4_CONFIG_BASE.format(stats="underflows%, mse")
    
    with debug_session(log_nvfp4_config, feature_dirs) as log_dir:
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
        from transformer_engine.pytorch.quantization import RecipeState
        
        recipe_state = RecipeState.create(
            recipe.NVFP4BlockScaling(),
            mode="forward",
            num_quantizers=3,
        )

        # Create test tensor with known distribution
        torch.manual_seed(42)
        tensor = torch.randn(128, 128, dtype=torch.bfloat16).cuda()
        # Add some small values that should underflow to zero in FP4
        tensor[0, :16] = 0.0001
        
        quantizer = recipe_state.make_quantizers()[0]
        quantized_tensor = quantizer(tensor)

        debug_api.transformer_engine.inspect_tensor(
            layer_name="test_layer",
            tensor_name="activation",
            iteration=0,
            tp_group=None,
            tensor=tensor,
            quantizer=quantizer,
            rowwise_quantized_tensor=quantized_tensor,
            columnwise_quantized_tensor=quantized_tensor,
        )
        debug_api.step()

        dequantized_tensor = quantized_tensor.dequantize()
        output = read_log(log_dir)

    # Validate both stats are present
    assert "nvfp4_underflows%" in output, "underflows% stat missing"
    assert "nvfp4_mse" in output, "mse stat missing"
    
    # Extract values and validate numerics
    underflows_value = None
    mse_value = None
    
    for line in output.splitlines():
        if "nvfp4_underflows%" in line and "value=" in line:
            underflows_value = float(line.split("value=")[1].split()[0])
        if "nvfp4_mse" in line and "value=" in line:
            mse_value = float(line.split("value=")[1].split()[0])
    
    # Validate underflows%
    assert underflows_value is not None, "Could not extract underflows% value"
    assert underflows_value >= 0, f"Underflows should be non-negative, got {underflows_value}"
    assert underflows_value <= 100, f"Underflows% should be <= 100, got {underflows_value}"
    
    # Compute expected underflows: non-zero elements that became zero after quantization
    orig_nonzero_mask = (tensor != 0)
    dequant_zero_mask = (dequantized_tensor == 0)
    expected_underflows = (orig_nonzero_mask & dequant_zero_mask).sum().float() / tensor.numel() * 100
    
    # Allow some tolerance
    assert abs(underflows_value - expected_underflows.item()) < 1.0, \
        f"Underflows mismatch: got {underflows_value}, expected {expected_underflows.item()}"
    
    # Validate MSE
    assert mse_value is not None, "Could not extract MSE value"
    assert mse_value >= 0, f"MSE should be non-negative, got {mse_value}"
    
    # Compute expected MSE
    expected_mse = torch.nn.functional.mse_loss(
        dequantized_tensor.float(), 
        tensor.float(), 
        reduction="mean"
    )
    
    assert mse_value == pytest.approx(expected_mse.cpu().item(), abs=1e-4), \
        f"MSE mismatch: got {mse_value}, expected {expected_mse.cpu().item()}"


def test_fp8_stats_allows_nvfp4_with_recipe_prefix(feature_dirs):
    """Test that LogFp8TensorStats allows recipe-prefixed stats with NVFP4 for what-if analysis."""
    if not nvfp4_available:
        pytest.skip(reason_for_no_nvfp4)
    
    # Use recipe-prefixed stat with NVFP4 - should work (computes MXFP8 separately)
    log_fp8_config = LOG_QUANTIZED_CONFIG_BASE.format(stats="mxfp8_mse")
    
    with debug_session(log_fp8_config, feature_dirs) as log_dir:
        model = te.Linear(128, 128, params_dtype=torch.bfloat16)
        inp = torch.randn(128, 128, dtype=torch.bfloat16).cuda()

        # Should work - recipe-prefixed stats compute MXFP8 separately for comparison
        for _ in range(2):
            with te.autocast(recipe=recipe.NVFP4BlockScaling()):
                output = model(inp)
            loss = output.sum()
            loss.backward()
            debug_api.step()
        
        output = read_log(log_dir)
        # Should have logged MXFP8 MSE stat (what-if scenario)
        assert "mxfp8_mse" in output
