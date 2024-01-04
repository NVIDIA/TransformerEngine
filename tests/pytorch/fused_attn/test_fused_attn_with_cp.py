# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import subprocess
from test_fused_attn import ModelConfig

model_configs = {
    #   test:             b,  h, hg,   d,    sq,   skv,   p,      mask,      bias
    "cp_1_0": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0,  "causal", "no_bias"), # MHA
    "cp_1_1": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0, "no_mask", "no_bias"), # MHA
    "cp_2_0": ModelConfig(1, 12,  1, 128, 16384, 16384, 0.0,  "causal", "no_bias"), # GQA
    "cp_2_1": ModelConfig(1, 12,  1, 128, 16384, 16384, 0.0, "no_mask", "no_bias"), # GQA
}

def get_bash_arguments(**kwargs):
    args = ["python", "-m", "torch.distributed.launch", "--nproc-per-node=2"]
    te_path = os.getenv("TE_PATH", "/opt/transformerengine")
    script_path = os.path.join(te_path, "tests/pytorch/fused_attn/run_fused_attn_with_cp.py")
    args.append(script_path)
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args

@pytest.mark.parametrize("dtype", ['bf16', 'fp16'])
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("qkv_format", ['bshd', 'sbhd'])
@pytest.mark.parametrize("kernel_backend", ['FlashAttention', 'FusedAttention'])
def test_dpa_with_cp(dtype, model, qkv_format, kernel_backend):
    subprocess.run(
        get_bash_arguments(
            dtype=dtype,
            model=model,
            qkv_format=qkv_format,
            kernel_backend=kernel_backend
        ),
        check=True
    )
