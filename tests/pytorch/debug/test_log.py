# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import nvdlfw_inspect.api as debug_api
import transformer_engine.debug
import transformer_engine.pytorch as te
import torch
import tempfile


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
    # "mxfp8",
]

bare_stats = [
    "underflows%",
    "scale_inv_min",
    "scale_inv_max",
    "mse",
]

stats = []

for recipe in recipes:
    for stat in bare_stats:
        for columnwise_postfix in ["", "_columnwise"]:
            if (
                recipe in ["fp8_current_scaling", "fp8_block_scaling", "mxfp8"]
                and torch.cuda.get_device_capability()[0] < 9
            ):
                # hopper in needed for current-scaling, block-scaling and mxfp8
                continue
            stats.append(f"{recipe}_{stat}{columnwise_postfix}")

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

            for i in range(10):
                with te.fp8_autocast():
                    output = model(torch.randn(128, 128, dtype=torch.bfloat16).cuda())
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
