# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
import itertools
import os
import random
import tempfile
from string import Template

import pytest
import torch

import nvdlfw_inspect.api as debug_api
import transformer_engine.debug
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.fp8 import _default_sf_compute
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from test_numerics import create_config_file

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
    LogFp8TensorStats:
      enabled: True
      tensors: [activation, gradient, weight]
      stats: [underflows, overflows]
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
      quant_format: FP8E5M2
""",
}


def _get_model(model_key):
    if model_key == "linear":
        return te.Linear(D, D)
    if model_key == "layernorm_linear":
        return te.LayerNormLinear(D, D)
    if model_key == "layernorm_mlp":
        return te.LayerNormMLP(D, D, D)
    if model_key == "mha_attention":
        return te.MultiheadAttention(D, H)
    if model_key == "transformer_layer":
        return te.TransformerLayer(D, D, H)


def _run_forward_backward(model, fp8):
    for _ in range(3):
        inp = torch.randn((S, B, H)).cuda()
        with te.fp8_autocast(enabled=fp8):
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
    _run_test(model_key, fp8, configs[config_key], feature_dirs)
