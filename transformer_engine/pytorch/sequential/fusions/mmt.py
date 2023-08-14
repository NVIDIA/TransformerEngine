from __future__ import annotations
from .. import nvte
from ..ops import Context, MMT, Add
from .. import nvte
from ._common import (
    register_fusion_inference,
    register_fusion_backward,
    register_fusion_forward,
)


@register_fusion_inference
def mmt_add_inf_fused(mmt: MMT, add: Add, x: nvte.Tensor):
    x = nvte.cast_checked(x, mmt.x_dtype)
    weight = nvte.cast_checked(mmt.weight, mmt.weight_dtype)
    bias = nvte.cast_checked(add.bias, add.bias_dtype)

    y = nvte.matmul_transpose_add(x, weight, bias, add.y_dtype)

    return y


@register_fusion_forward
def mmt_add_fwd_fused(mmt: MMT, add: Add, x: nvte.Tensor):
    (x, x_t), (weight, weight_t) = nvte.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (mmt.weight, mmt.weight_dtype)
    )
    bias = nvte.cast_checked(add.bias, add.bias_dtype)

    y = nvte.matmul_transpose_add(x, weight, bias, add.y_dtype)

    return y, ({"x_t": x_t, "weight_t": weight_t}, Context())


@register_fusion_backward
def mmt_add_bwd_fused(
    mmt: MMT,
    add: Add,
    mmt_ctx: Context,
    add_ctx: Context,
    dy: nvte.Tensor,
):
    del add_ctx
    x_t, weight_t = mmt_ctx["x_t"], mmt_ctx["weight_t"]
    dy, dy_t, dbias = nvte.cast_transpose_dbias_checked(
        dy, mmt.dy_dtype, add.dbias_dtype
    )

    dx = nvte.matmul_transpose(dy, weight_t, mmt.dx_dtype)
    dweight = nvte.matmul_transpose(x_t, dy_t, mmt.dweight_dtype)

    return dx, ([dweight], [dbias])


# fusion function names (ex. mmt_add_bwd_fused) are for debugging only, as they are called from a dictionary like FUSIONS_FWD
__all__ = []
