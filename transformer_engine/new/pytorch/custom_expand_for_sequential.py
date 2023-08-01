from typing import Callable
from torch import nn
from ..common_back.generic_environment import ExecutionEnv
from ..common_back.ops import Op

CUSTOM_EXPAND_FOR_SEQUENTIAL: dict[
    type, Callable[[nn.Module, ExecutionEnv], list[Op]]
] = {}


def expand(m: nn.Module, compile_env: ExecutionEnv) -> list[Op]:
    if hasattr(m, "expand_for_sequential"):
        return m.expand_for_sequential(compile_env)  # type: ignore
    elif type(m) in CUSTOM_EXPAND_FOR_SEQUENTIAL:
        return CUSTOM_EXPAND_FOR_SEQUENTIAL[type(m)](m, compile_env)
    else:
        raise NotImplementedError


__all__ = ["CUSTOM_EXPAND_FOR_SEQUENTIAL", "expand"]
