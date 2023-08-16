from abc import ABC
from torch import nn
from .base import BaseModule
from .. import ops


class Activation(BaseModule, ABC):
    def __init__(self):
        nn.Module.__init__(self)  # type: ignore
        super().__init__(type(self)._op_type())

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
