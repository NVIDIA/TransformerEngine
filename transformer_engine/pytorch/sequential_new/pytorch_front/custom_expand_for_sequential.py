from typing import Callable
from torch import nn
from ..common_back.compile_env import CompileEnv
from ..common_back.ops import OpBase

CUSTOM_EXPAND_FOR_SEQUENTIAL: dict[
    type, Callable[[nn.Module, CompileEnv], list[OpBase]]
] = {}


def expand(m: nn.Module, compile_env: CompileEnv) -> list[OpBase]:
    if hasattr(m, "expand_for_sequential"):
        return m.expand_for_sequential(compile_env)  # type: ignore
    elif type(m) in CUSTOM_EXPAND_FOR_SEQUENTIAL:
        return CUSTOM_EXPAND_FOR_SEQUENTIAL[type(m)](m, compile_env)
    else:
        raise NotImplementedError


__all__ = ["CUSTOM_EXPAND_FOR_SEQUENTIAL", "expand"]
