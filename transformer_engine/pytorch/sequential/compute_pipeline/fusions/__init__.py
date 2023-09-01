from ...utils import prevent_import

prevent_import("torch")
from .interface import FusedOp, get_fused_op_list
from . import mmt  # only for side effects

__all__ = ["FusedOp", "get_fused_op_list"]
