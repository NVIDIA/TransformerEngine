# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Tuple, Union
import pytest
import subprocess
import os
from dataclasses import dataclass, asdict
from functools import lru_cache

import torch


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


fp8_recipes = [False, "hybrid"]


all_boolean = [True, False]


te_path = os.getenv("TE_PATH", "/opt/transformerengine")
mlm_log_dir = os.path.join(te_path, "ci_logs")


@lru_cache(maxsize=1)
def get_parallel_configs() -> List[Tuple[int, int]]:
    """Returns valid combinations of (tp, pp)."""
    sizes = [1, 2, 4]
    num_devices = torch.cuda.device_count()
    parallel_configs = []
    if num_devices > 1:
        for dp in sizes:
            for tp in sizes:
                for pp in sizes:
                    if dp * tp * pp == num_devices:
                        parallel_configs.append((dp, tp, pp))
    return parallel_configs


def get_filename(
    model: str, dp: int, tp: int, pp: int, sp: bool, use_te: bool, fp8_recipe: Union[bool, str]
) -> str:
    sp = tp if sp else 1
    config = f"gpt3_{model}_dp{dp}_tp{tp}_pp{pp}_sp{sp}"
    config_dir = os.path.join(mlm_log_dir, config)
    os.makedirs(config_dir, exist_ok=True)
    fname = (
        f"{'te' if use_te else 'megatron'}" + (f"_fp8_{fp8_recipe}" if fp8_recipe else "") + ".txt"
    )
    return os.path.join(config_dir, fname)


def get_bash_arguments(filename: str, **kwargs) -> List[str]:
    args = []
    script_path = os.path.join(te_path, "tests/pytorch/distributed/run_megatron_lm_gpt.sh")
    args.append(script_path)

    for k, v in kwargs.items():
        args.append(f"{k}={str(v)}")
    args.append(f"FILENAME={filename}")
    return args


@pytest.mark.parametrize("sp", all_boolean)
@pytest.mark.parametrize("use_te", all_boolean)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("dp, tp, pp", get_parallel_configs())
@pytest.mark.parametrize("model", model_configs.keys())
def test_distributed(dtype, fp8_recipe, dp, tp, pp, sp, use_te, model):
    if sp and tp == 1:
        pytest.skip("No tensor parallel.")
    if fp8_recipe and not use_te:
        pytest.skip("TransformerEngine needed for FP8.")
    subprocess.run(
        get_bash_arguments(
            get_filename(model, dp, tp, pp, sp, use_te, fp8_recipe),
            DTYPE=dtype,
            FP8=fp8_recipe,
            SP=sp,
            DP_SIZE=dp,
            TP_SIZE=tp,
            PP_SIZE=pp,
            TRANSFORMER_IMPL="transformer_engine" if use_te else "local",
            **asdict(model_configs[model]),
        ),
        check=True,
    )
