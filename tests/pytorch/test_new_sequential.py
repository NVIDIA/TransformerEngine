from pprint import pprint
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.pytorch.sequential_new import Sequential
from transformer_engine.pytorch.sequential_new.compile_env import CompileEnv
from transformer_engine.pytorch.sequential_new.pytorch_interface import PytorchInterface

seq = Sequential(
    te.LayerNorm(10),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    model_parallel=True,
)

seq._compile_checked(CompileEnv.current())

pprint([type(op).__name__ for op in seq._pipeline._ops])

assert isinstance(seq._pipeline._framework_interface, PytorchInterface)

pprint(seq._pipeline._framework_interface._buffers)
