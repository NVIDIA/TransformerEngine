from __future__ import annotations
from typing import Callable
from typing_extensions import TypeVarTuple, Unpack
from ..ops import Context, Grads
from .. import nvte
from ._storage import FUSIONS_FWD, FUSIONS_BWD, FUSIONS_INF
from ..utils import get_arg_types

_Ops = TypeVarTuple("_Ops")
_OpsAndCtxs = TypeVarTuple("_OpsAndCtxs")


def register_fusion_inference(f: Callable[[Unpack[_Ops], nvte.Tensor], nvte.Tensor]):  # type: ignore[invalid-typevar-use]
    fused_modules = get_arg_types(f)[:-1]
    FUSIONS_INF[tuple(fused_modules)] = f
    return f


def register_fusion_forward(
    f: Callable[
        [Unpack[_Ops], nvte.Tensor],  # type: ignore[invalid-typevar-use]
        tuple[nvte.Tensor, tuple[Context, ...]],
    ]
):
    fused_modules = get_arg_types(f)[:-1]
    FUSIONS_FWD[tuple(fused_modules)] = f
    return f


def register_fusion_backward(
    f: Callable[
        [Unpack[_OpsAndCtxs], nvte.Tensor],  # type: ignore[invalid-typevar-use]
        tuple[nvte.Tensor, tuple[Grads, ...]],
    ]
):
    arg_types = get_arg_types(f)
    module_count = (len(arg_types) - 1) // 2
    fused_modules = arg_types[:module_count]
    FUSIONS_BWD[tuple(fused_modules)] = f
    return f
