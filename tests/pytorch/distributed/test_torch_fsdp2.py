# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch

import transformer_engine.pytorch as te

fp8_available, reason_for_no_fp8 = te.FP8GlobalStateManager.is_fp8_available()

NUM_PROCS: int = torch.cuda.device_count()


def _run_test(fp_init, sharding_dims, layer_type):
    test_path = Path(__file__).parent.resolve() / "run_fsdp2_model.py"
    test_cmd = ["torchrun", f"--nproc_per_node={NUM_PROCS}", str(test_path)]

    test_cmd = ["--layer-type", layer_type.__name__]
    if fp_init:
        test_cmd += ["--fp8-init"]
    if len(sharding_dims) == 1:
        test_cmd += ["--sharding-dims", str(sharding_dims[0])]
    elif len(sharding_dims) == 2:
        test_cmd += ["--sharding-dims", str(sharding_dims[0]), str(sharding_dims[1])]
    else:
        assert False
    return subprocess.run(test_cmd, env=os.environ, check=True)


@pytest.mark.skipif(NUM_PROCS < 4, reason="Requires 4+ GPUs")
@pytest.mark.skipif(NUM_PROCS % 2 != 0, reason="Requires even number of GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
@pytest.mark.parametrize("sharding_dims", ([NUM_PROCS], [2, NUM_PROCS // 2]))
@pytest.mark.parametrize("fp8_init", (False, True))
@pytest.mark.parametrize(
    "layer_type",
    (te.Linear, te.LayerNormLinear, te.LayerNormMLP, te.MultiheadAttention, te.TransformerLayer),
)
def test_torch_fsdp2(fp8_init, sharding_dims, layer_type):
    """Test a Transformer Engine Linear layer with PyTorch native FSDP2."""
    # Skip invalid configurations
    if torch.cuda.device_count() < 4:
        pytest.skip("FSDP2 test requires at least 4 GPUs")

    if fp8_init and not fp8_available:
        pytest.skip(reason_for_no_fp8)

    _run_test(fp8_init, sharding_dims, layer_type)
