from typing import Literal
from . import _pass


def set_current_pass(pass__: Literal["forward", "backward", "inference"]):
    _pass.pass_ = pass__
