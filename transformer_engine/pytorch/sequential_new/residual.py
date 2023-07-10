from torch import nn

from .compile_env import CompileEnv
from .ops import Op
from .expand_for_sequential import expand


class Residual(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()  # type: ignore
        self.module_list = [*modules]

    def expand_for_sequential(self, compile_env: CompileEnv):
        return [
            Op.RESIDUAL_BEGIN,
            *[op for m in self.module_list for op in expand(m, compile_env)],
            Op.RESIDUAL_END,
        ]


__all__ = ["Residual"]
