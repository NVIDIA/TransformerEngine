from torch import nn
from .compile_env import CompileEnv
from .ops import Op
from .custom_expand_for_sequential import CUSTOM_EXPAND_FOR_SEQUENTIAL
from . import custom_expand  # loads custom expanders


def expand(m: nn.Module, compile_env: CompileEnv) -> list[Op]:
    if hasattr(m, "expand_for_sequential"):
        return m.expand_for_sequential(compile_env)  # type: ignore
    elif type(m) in CUSTOM_EXPAND_FOR_SEQUENTIAL:
        return CUSTOM_EXPAND_FOR_SEQUENTIAL[type(m)](m, compile_env)
    else:
        raise NotImplementedError


__all__ = ["expand"]
