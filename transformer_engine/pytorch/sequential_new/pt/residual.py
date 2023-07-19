from torch import nn

from ..common.compile_env import CompileEnv
from .expand_for_sequential import expand
from ..common.ops import ResidualBegin, ResidualEnd


class Residual(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()  # type: ignore
        self.module_list = [*modules]

    def expand_for_sequential(self, compile_env: CompileEnv):
        begin = ResidualBegin("residual_begin")
        end = ResidualEnd("residual_end", begin)
        begin.end = end

        return [
            begin,
            *[op for m in self.module_list for op in expand(m, compile_env)],
            end,
        ]


__all__ = ["Residual"]
