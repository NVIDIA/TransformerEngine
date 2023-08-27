from __future__ import annotations
from .dynamic_load import inject_real

inject_real(globals())

from .tensor import Tensor
