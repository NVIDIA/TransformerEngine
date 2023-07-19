from pprint import pprint
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.sequential_new import Sequential, Residual
from transformer_engine.pytorch.sequential_new.common_back.compile_env import CompileEnv
from transformer_engine.pytorch.sequential_new.pytorch_back.pytorch_interface import (
    PytorchInterface,
)

token_size = 512
ffn_size = 2048
heads = 8
blocks = 6

transformer = blocks * Sequential(
    Residual(
        te.LayerNorm(token_size),
        te.Linear(token_size, 3 * token_size),
        te.DotProductAttention(heads, token_size // heads),
        te.Linear(3 * token_size, token_size),
        nn.Dropout(0.3),
    ),
    Residual(
        te.LayerNorm(token_size),
        nn.Linear(token_size, ffn_size),
        nn.ReLU(),
        nn.Linear(ffn_size, token_size),
        nn.Dropout(0.3),
    ),
    model_parallel=True,
)

fp8_recipe = recipe.DelayedScaling()

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    transformer._compile_checked(CompileEnv.current())

pprint(
    [
        f"{type(op).__name__}[{op.input_type.value},{op.output_type.value}]".ljust(40)
        + op.name
        for op in transformer._pipeline._fwd
    ],
    width=200,
)

assert isinstance(transformer._pipeline._framework_interface, PytorchInterface)

pprint(transformer._pipeline._framework_interface.__dict__)
