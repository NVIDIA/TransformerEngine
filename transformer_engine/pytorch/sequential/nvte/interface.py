from typing import Literal
from . import _common



def set_current_pass(pass__: Literal["forward", "backward", "inference"]):
    _common.pass_ = pass__
