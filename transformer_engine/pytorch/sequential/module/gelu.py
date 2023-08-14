from torch import nn
from .base import BaseModule
from .. import ops


class GELU(BaseModule):
    def __init__(self):
        nn.Module.__init__(self)  # type: ignore
        super().__init__(ops.GELU())
