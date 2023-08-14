from __future__ import annotations
import ast
import typing
from typing import Callable, Any
from typing_extensions import TypeVarTuple, Unpack
from ..ops import Context, Grads
from .. import nvte
from ._storage import FUSIONS_FWD, FUSIONS_BWD, FUSIONS_INF

_Ops = TypeVarTuple("_Ops")
_OpsAndCtxs = TypeVarTuple("_OpsAndCtxs")


def _get_arg_types(f: Callable[..., Any]):
    annotations = typing.get_type_hints(f)
    annotations.pop("return", None)
    arg_type_annotations: tuple[str | type] = tuple(annotations.values())
    assert all(isinstance(val, (str, type)) for val in arg_type_annotations)
    arg_types: tuple[type] = tuple(
        ast.literal_eval(val) if isinstance(val, str) else val
        for val in arg_type_annotations
    )
    return arg_types


def register_fusion_inference(f: Callable[[Unpack[_Ops], nvte.Tensor], nvte.Tensor]):  # type: ignore[invalid-typevar-use]
    fused_modules = _get_arg_types(f)[:-1]
    FUSIONS_INF[fused_modules] = f
    return f


def register_fusion_forward(
    f: Callable[
        [Unpack[_Ops], nvte.Tensor],  # type: ignore[invalid-typevar-use]
        tuple[nvte.Tensor, tuple[Context, ...]],
    ]
):
    fused_modules = _get_arg_types(f)[:-1]
    FUSIONS_FWD[fused_modules] = f
    return f


def register_fusion_backward(
    f: Callable[
        [Unpack[_OpsAndCtxs], nvte.Tensor],  # type: ignore[invalid-typevar-use]
        tuple[nvte.Tensor, tuple[Grads, ...]],
    ]
):
    arg_types = _get_arg_types(f)
    module_count = (len(arg_types) - 1) // 2
    fused_modules = arg_types[:module_count]
    FUSIONS_BWD[fused_modules] = f
    return f
