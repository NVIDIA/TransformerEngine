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
)
from transformer_engine.pytorch.quantization import RecipeState
from transformer_engine.debug.pytorch.debug_state import TEDebugState
import math

fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = is_fp8_block_scaling_available(
    return_reason=True
)

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
def test_log_quantized_stats_numerics(fp8_recipe, feature_dirs):
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


LOG_HIGH_PRECISION_CONFIG_BASE = """
log:
  layers:
    layer_name_regex_pattern: .*
  enabled:
    True
  transformer_engine:
    LogTensorStats:
      enabled: True
      stats:
        - dynamic_range
        - max_blockwise_dynamic_range:
            block_size: 4
            dims: 1
        - max_blockwise_dynamic_range:
            block_size: 4
            dims: 2
      tensors: [activation, gradient, weight]
      freq: 2
      start_step: 0
      end_step: 10
"""


def test_log_stats_numerics(feature_dirs):
    """Check corectness of dynamic range and max blockwise dynamic range stats"""
    stats = ["dynamic_range", "max_blockwise_4_dynamic_range"]
    log_only_bare_stats_config = LOG_HIGH_PRECISION_CONFIG_BASE.format(stats=", ".join(stats))

    with debug_session(log_only_bare_stats_config, feature_dirs) as log_dir:
        # There is 1024 x 1024 tensor with very small epsilon values in almost all elements,
        # one row of large value A and three rows of large value B.
        epsilon = 1e-10
        A = 1000
        B = 50
        tensor = torch.zeros(1024, 1024).cuda() + epsilon
        tensor[0, :] = A
        tensor[1:4, :] = B

        debug_api.transformer_engine.inspect_tensor(
            layer_name="layer_name",
            tensor_name="activation",
            iteration=0,
            tp_group=None,
            tensor=tensor,
            quantizer=None,
            rowwise_quantized_tensor=None,
            columnwise_quantized_tensor=None,
        )
        debug_api.step()

        output = read_log(log_dir)

    for line in output.splitlines():
        if "max_blockwise_dynamic_range_block_size_4_dims_1" in line:
            max_blockwise_dynamic_range_block_size_4_dims_1 = float(line.split("value=")[1])
            expected = 0
            assert max_blockwise_dynamic_range_block_size_4_dims_1 == pytest.approx(
                expected, abs=1e-4
            )
        elif "max_blockwise_dynamic_range_block_size_4_dims_2" in line:
            max_blockwise_dynamic_range_block_size_4_dims_2 = float(line.split("value=")[1])
            expected = math.log2(A) - math.log2(B)
            assert max_blockwise_dynamic_range_block_size_4_dims_2 == pytest.approx(
                expected, abs=1e-4
            )
        elif "dynamic_range" in line:
            dynamic_range = float(line.split("value=")[1])
            expected = math.log2(A) - math.log2(epsilon)
            assert dynamic_range == pytest.approx(expected, abs=1e-4)


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
