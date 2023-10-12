from abc import ABC
from .base import BaseModule
from ..compute_pipeline import ops


class Activation(BaseModule, ABC):
    def __init__(self):
        super().__init__()

    def _ops(self) -> list[ops.Op | None]:
        return [type(self)._op_type()]

    _op_type: type[ops.Activation]


class ReLU(Activation):
    _op_type = ops.ReLU


class GELU(Activation):
    _op_type = ops.GELU


class ReGLU(Activation):
    _op_type = ops.ReGLU


class GeGLU(Activation):
    _op_type = ops.GeGLU


class SwiGLU(Activation):
    _op_type = ops.SwiGLU
