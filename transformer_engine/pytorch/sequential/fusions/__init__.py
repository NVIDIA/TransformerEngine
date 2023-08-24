from .interface import FusedOp, get_fused_op_list
from ..utils import import_file_as_module

import_file_as_module("mmt", only_for_side_effects=True)

__all__ = ["FusedOp", "get_fused_op_list"]
