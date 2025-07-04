# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import nvdlfw_inspect.api as debug_api
import transformer_engine.debug
import transformer_engine.pytorch as te
import torch
import tempfile
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import RecipeState
import pytest
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager


fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
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
      tensors: [activation, gradient, weight, output]
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

stats = []

for r in recipes:
    for stat in bare_stats:
        for columnwise_postfix in ["", "_columnwise"]:
            if (
                r in ["fp8_current_scaling", "fp8_block_scaling", "mxfp8"]
                and torch.cuda.get_device_capability()[0] < 9
            ):
                # hopper in needed for current-scaling, block-scaling and mxfp8
                continue
            stats.append(f"{r}_{stat}{columnwise_postfix}")

stats.append("fp8_delayed_scaling_overflows%")  # only delayed-scaling supports overflows%

LOG_QUANTIZED_CONFIG = LOG_QUANTIZED_CONFIG_BASE.format(stats=", ".join(stats))


def test_log_quantized(feature_dirs):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file.write(LOG_QUANTIZED_CONFIG)
            temp_file.flush()
            temp_file_path = temp_file.name
            debug_api.initialize(
                config_file=temp_file_path, feature_dirs=feature_dirs, log_dir=temp_dir
            )

            model = te.Linear(128, 128, params_dtype=torch.bfloat16)

            inp = torch.zeros(128, 128, dtype=torch.bfloat16).cuda()
            inp[0, 0] = 1000

            for i in range(10):
                with te.fp8_autocast(fp8_recipe=recipe.DelayedScaling()):
                    output = model(inp)
                loss = output.sum()
                loss.backward()
                debug_api.step()

            debug_api.end_debug()

            stat_file_path = (
                temp_dir + "/nvdlfw_inspect_statistics_logs/nvdlfw_inspect_globalrank-0.log"
            )

            output = None
            with open(stat_file_path, "r") as f:
                output = f.read()

            assert len(output) > 0, "Output is empty"

            for stat in stats:
                assert stat in output, f"Stat {stat} not found in output"

fp8_recipes = [
    recipe.MXFP8BlockScaling(),
    recipe.DelayedScaling(),
    recipe.Float8CurrentScaling(),
    recipe.Float8BlockScaling(),
]

LOG_QUANTIZED_CONFIG_2 = LOG_QUANTIZED_CONFIG_BASE.format(stats=", ".join(bare_stats))

@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
def test_api_log_quantized(fp8_recipe, feature_dirs):
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if not mxfp8_available and fp8_recipe == recipe.MXFP8BlockScaling():
        pytest.skip(reason_for_no_mxfp8)
    if not fp8_block_scaling_available and fp8_recipe == recipe.Float8BlockScaling():
        pytest.skip(reason_for_no_fp8_block_scaling)

    # Test output of each stat with API, not layers.

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(LOG_QUANTIZED_CONFIG_2)
        temp_file.flush()
        temp_file_path = temp_file.name
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_api.initialize(
                config_file=temp_file_path, feature_dirs=feature_dirs, log_dir=temp_dir
            )

            recipe_state = RecipeState.create(
                fp8_recipe,
                mode="forward",
                num_quantizers=3,
            )

            tensor = torch.zeros(1024, 1024).cuda()
            tensor[0, :] = 1000
            quantizer = recipe_state.make_quantizers()[0]
            quantized_tensor = quantizer(tensor)

            debug_api.transformer_engine.inspect_tensor_all(
                layer_name="layer_name",
                tensor_name="activation",
                iteration=0,
                tp_group=None,
                original_tensor=tensor,
                quantizer=quantizer,
                quantized_tensor_rowwise=quantized_tensor,
                quantized_tensor_columnwise=quantized_tensor,
            )
            debug_api.step()

            # read stats
            stat_file_path = (
                temp_dir + "/nvdlfw_inspect_statistics_logs/nvdlfw_inspect_globalrank-0.log"
            )

            dequantized_tensor = quantized_tensor.dequantize()

            output = None
            with open(stat_file_path, "r") as f:
                output = f.read()
            
            for line in output.split("\n"):
                if "undeflows%" in line:
                    underflows = float(line.split("value=")[1])
                    expected_underflows = (dequantized_tensor == 0).sum()
                    assert underflows == pytest.approx(expected_underflows.cpu())
                if "mse" in line:
                    mse = float(line.split("value=")[1])
                    expected_mse = torch.nn.functional.mse_loss(dequantized_tensor, tensor, reduction="mean")
                    assert mse == pytest.approx(expected_mse.cpu(), abs=1e-6)
                if "scale_inv_min" in line:
                    scale_inv_min = float(line.split("value=")[1])
                if "scale_inv_max" in line:
                    scale_inv_max = float(line.split("value=")[1])
                if "overflows%" in line:
                    overflows = float(line.split("value=")[1])
                    expected_overflows = (abs(dequantized_tensor) > abs(tensor)).sum()
                    assert overflows == pytest.approx(expected_overflows.cpu())
            

            debug_api.end_debug()