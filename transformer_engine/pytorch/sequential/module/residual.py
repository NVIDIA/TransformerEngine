from ..compute_pipeline import ops
from .sequential import Sequential


class Residual(Sequential):
    def _ops(self):
        begin, end = ops.ResidualBegin(), ops.ResidualEnd()
        begin.end = end
        end.begin = begin
        return [begin] + super()._ops() + [end]
