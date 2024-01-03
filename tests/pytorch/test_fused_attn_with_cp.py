import pytest
import subprocess
from test_fused_attn import ModelConfig

model_configs = {
    #   test:             b,  h, hg,   d,    sq,   skv,   p,     mask,      bias
    "cp_1_0": ModelConfig(1, 12, 12, 128, 16384, 16384, 0.0, "causal", "no_bias"),
}

def get_bash_arguments(**kwargs):
    args = ["python", "-m", "torch.distributed.launch", "--nproc-per-node=2", "run_fused_attn_with_cp.py"]
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args

@pytest.mark.parametrize("dtype", ['bf16'])
@pytest.mark.parametrize("model", model_configs.keys())
@pytest.mark.parametrize("qkv_format", ['bshd'])
def test_dpa_with_cp(dtype, model, qkv_format):
    subprocess.run(
        get_bash_arguments(
            dtype=dtype,
            model=model,
            qkv_format=qkv_format
        ),
        check=True
    )
