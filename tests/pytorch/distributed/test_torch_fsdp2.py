# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import subprocess
from pathlib import Path
import transformer_engine.pytorch as te

import torch


fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
NUM_PROCS: int = torch.cuda.device_count()


def _run_test(fp_init, sharding_dims, recipe, layer_type):
    test_path = Path(__file__).parent.resolve() / "run_fsdp2_model.py"
    test_cmd = ["torchrun", f"--nproc_per_node={NUM_PROCS}", str(test_path)]

    if fp_init:
        test_cmd += ["--fp8-init"]

    if len(sharding_dims) == 1:
        test_cmd += ["--sharding-dims", str(sharding_dims[0])]
    elif len(sharding_dims) == 2:
        test_cmd += ["--sharding-dims", str(sharding_dims[0]), str(sharding_dims[1])]
    else:
        assert False
    test_cmd += ["--recipe", recipe]
    test_cmd += ["--layer-type", layer_type]

    result = subprocess.run(test_cmd, env=os.environ, check=True)


@pytest.mark.skipif(NUM_PROCS < 4, reason="Requires 4+ GPUs")
@pytest.mark.skipif(NUM_PROCS % 2 != 0, reason="Requires even number of GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
@pytest.mark.parametrize("sharding_dims", ([NUM_PROCS], [2, NUM_PROCS // 2]))
@pytest.mark.parametrize("fp8_init", (False, True))
@pytest.mark.parametrize("recipe", ("delayed_scaling", "current_scaling", "mx_fp8_block_scaling"))
@pytest.mark.parametrize("layer_type", ("LayerNormLinear", "TransformerLayer"))
def test_distributed(fp8_init, sharding_dims, recipe, layer_type):

    # Skip invalid configurations
    if torch.cuda.device_count() < 4:
        pytest.skip("FSDP2 test requires at least 4 GPUs")

    if recipe == "mx_fp8_block_scaling" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    elif not fp8_available:
        pytest.skip(reason_for_no_fp8)

    _run_test(fp8_init, sharding_dims, recipe, layer_type)


def test_dummy() -> None:
    """Dummy test

    pytest returns exit code 5 if all tests are skipped.

    """
    pass
