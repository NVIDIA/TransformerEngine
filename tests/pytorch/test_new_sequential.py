from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.pytorch.sequential_new import Sequential
from transformer_engine.pytorch.sequential_new.compile_env import CompileEnv

seq = Sequential(
    te.LayerNorm(10),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
)

seq._compile_checked(CompileEnv.current())

print(seq._pipeline._ops)
