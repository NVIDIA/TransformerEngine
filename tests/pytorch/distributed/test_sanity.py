# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import List
import pytest
import subprocess
import os
from dataclasses import dataclass, asdict


@dataclass()
class ModelConfigGPT:
    NUM_LAYERS: int = 12
    HIDDEN_SIZE: int = 768
    NHEADS: int = 12
    SEQLEN: int = 2048
    MAX_POSITION_EMBEDDINGS: int = 2048
    LR: float = 6.0e-4
    MIN_LR: float = 6.0e-5
    SPLIT: str = "98,2,0"
    CLIP_GRAD: float = 1.0
    WEIGHT_DECAY: float = 0.1
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.95
    INIT_METHOD_STD: float = 0.023


model_configs = {
    "126m": ModelConfigGPT(),
}

dtypes = ["bf16"]

# (TP_SIZE, PP_SIZE)
# DP_SIZE = 4 / (TP_SIZE * PP_SIZE)
parallel_configs = [
    (1, 1), # DP only
    (1, 4), # PP only
    (4, 1), # TP only
    (2, 2), # TP + PP
    (2, 1), # TP + DP
    (1, 2), # DP + PP
]

fp8_recipes = [False, "hybrid"]

all_boolean = [True, False]


def get_bash_arguments(**kwargs) -> List[str]:
    args = []
    script_path = os.path.join(
        os.getenv("TE_PATH", "/opt/transformerengine"),
        "tests/pytorch/distributed/run_megatron_lm_gpt.sh")
    args.append(script_path)

    for k, v in kwargs.items():
        args.append(f"{k}={str(v)}")
    return args


@pytest.mark.parametrize("sp", all_boolean)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("tp, pp", parallel_configs)
@pytest.mark.parametrize("model", model_configs.keys())
def test_distributed(dtype, fp8_recipe, tp, pp, sp, model):
    if sp and tp == 1:
        pytest.skip("No tensor parallel.")
    subprocess.run(
        get_bash_arguments(
            DTYPE=dtype,
            FP8=fp8_recipe,
            SP=sp,
            TP_SIZE=tp,
            PP_SIZE=pp,
            **asdict(model_configs[model]),
        ),
        check=True)
