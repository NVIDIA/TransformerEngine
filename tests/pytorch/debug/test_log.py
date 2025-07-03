# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import pytest
import torch
import transformer_engine.pytorch as te 
import time
import tempfile
import os

import nvdlfw_inspect.api as debug_api

from transformer_engine.debug.pytorch.debug_state import TEDebugState



@pytest.mark.parametrize("layer", ["linear", "transformer"])
def test_cpu_overhead(layer, configs_dir, feature_dirs):
    with tempfile.TemporaryDirectory() as temp_dir:
        debug_api.initialize(
            config_file=configs_dir + "/log_config.yaml",
            feature_dirs=feature_dirs,
            log_dir=temp_dir
        )

        model = te.TransformerLayer(128, 128, 4, name="transformer1")
        for i in range(10):
            x = torch.randn(4, 4, 128).cuda()
            y = model(x)
            y.sum().backward()
            debug_api.step()
        os.system(f"ls -l {temp_dir}")

        with open(os.path.join(temp_dir, "nvdlfw_inspect_statistics_logs/nvdlfw_inspect_globalrank-0.log"), "r") as f:
            file_content = f.read()
            for i in range(1, 11):
                if i % 3 == 0:
                    assert f"iteration={i:06d}" in file_content
                else:
                    assert f"iteration={i:06d}" not in file_content
    
    debug_api.end_debug()
    TEDebugState._reset()
        






