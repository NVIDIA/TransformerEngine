from typing import Callable
from torch import nn
from .compile_env import CompileEnv
from .ops import Op

CUSTOM_EXPAND_FOR_SEQUENTIAL: dict[
    type, Callable[[nn.Module, CompileEnv], list[Op]]
] = {}

__all__ = ["CUSTOM_EXPAND_FOR_SEQUENTIAL"]
