from __future__ import annotations

from ... import nvte
from ..ops import Context, Grads, MMT, Add, GELU, GeGLU
from ... import nvte
from ._common import (
    register_fusion_inference,
    register_fusion_backward,
    register_fusion_forward,
)


# MMT, Add
@register_fusion_inference
def mmt_add_inf_fused(mmt: MMT, add: Add, x: nvte.Tensor):
    x = nvte.cast_checked(x, mmt.x_dtype)
    weight = nvte.cast_checked(mmt.weight, mmt.weight_dtype)
    bias = nvte.cast_checked(add.bias, add.bias_dtype)

    y = nvte.matmul_transpose_add(
        x, weight, bias, add.y_dtype or mmt.y_dtype or x.dtype
    )

    return y


@register_fusion_forward
def mmt_add_fwd_fused(
    mmt: MMT, add: Add, x: nvte.Tensor
) -> tuple[nvte.Tensor, tuple[Context, Context]]:
    (x, x_t), (weight, weight_t) = nvte.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (mmt.weight, mmt.weight_dtype)
    )
    bias = nvte.cast_checked(add.bias, add.bias_dtype)

    y = nvte.matmul_transpose_add(
        x, weight, bias, add.y_dtype or mmt.y_dtype or x.dtype
    )

    return y, ({"x_t": x_t, "weight_t": weight_t}, {})


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
        dy, mmt.dy_dtype, add.dbias_dtype or add.bias.dtype
    )

    dx = nvte.matmul_transpose(dy, weight_t, mmt.dx_dtype or add.dx_dtype or dy.dtype)
    dweight = nvte.matmul_transpose(x_t, dy_t, mmt.dweight_dtype or mmt.weight.dtype)

    return dx, ([dweight], [dbias])


# MMT, Add, GELU
@register_fusion_inference
def mmt_add_gelu_inf_fused(mmt: MMT, add: Add, gelu: GELU, x: nvte.Tensor):
    x = nvte.cast_checked(x, mmt.x_dtype)
    weight = nvte.cast_checked(mmt.weight, mmt.weight_dtype)
    bias = nvte.cast_checked(add.bias, add.bias_dtype)

    _, y = nvte.matmul_transpose_add_gelu(
        x, weight, bias, gelu.y_dtype or add.y_dtype or mmt.y_dtype or x.dtype
    )

    return y


@register_fusion_forward
def mmt_add_gelu_fwd_fused(
    mmt: MMT, add: Add, gelu: GELU, x: nvte.Tensor
) -> tuple[nvte.Tensor, tuple[Context, Context, Context]]:
    (x, x_t), (weight, weight_t) = nvte.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (mmt.weight, mmt.weight_dtype)
    )
    bias = nvte.cast_checked(add.bias, add.bias_dtype)

    pre_gelu, y = nvte.matmul_transpose_add_gelu(
        x, weight, bias, gelu.y_dtype or add.y_dtype or mmt.y_dtype or x.dtype
    )

    return y, ({"x_t": x_t, "weight_t": weight_t}, {}, {"x": pre_gelu})


@register_fusion_backward
def mmt_add_gelu_bwd_fused(
    mmt: MMT,
    add: Add,
    gelu: GELU,
    mmt_ctx: Context,
    add_ctx: Context,
    gelu_ctx: Context,
    dy: nvte.Tensor,
) -> tuple[nvte.Tensor, tuple[Grads, Grads, Grads]]:
    del add_ctx
    x_t, weight_t, pre_gelu = mmt_ctx["x_t"], mmt_ctx["weight_t"], gelu_ctx["x"]
    dy, dy_t, dbias = nvte.cast_transpose_dbias_dgelu_checked(
        dy, pre_gelu, mmt.dy_dtype, add.dbias_dtype or add.bias.dtype
    )

    dx = nvte.matmul_transpose(
        dy, weight_t, mmt.dx_dtype or add.dx_dtype or gelu.dx_dtype or dy.dtype
    )
    dweight = nvte.matmul_transpose(x_t, dy_t, mmt.dweight_dtype or mmt.weight.dtype)

    return dx, ([dweight], [dbias], [])


# MMT, GELU
@register_fusion_inference
def mmt_gelu_inf_fused(mmt: MMT, gelu: GELU, x: nvte.Tensor):
    x = nvte.cast_checked(x, mmt.x_dtype)
    weight = nvte.cast_checked(mmt.weight, mmt.weight_dtype)

    _, y = nvte.matmul_transpose_gelu(x, weight, gelu.y_dtype or mmt.y_dtype or x.dtype)

    return y


@register_fusion_forward
def mmt_gelu_fwd_fused(mmt: MMT, gelu: GELU, x: nvte.Tensor):
    (x, x_t), (weight, weight_t) = nvte.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (mmt.weight, mmt.weight_dtype)
    )

    pre_gelu, y = nvte.matmul_transpose_gelu(
        x, weight, gelu.y_dtype or mmt.y_dtype or x.dtype
    )

    return y, ({"x_t": x_t, "weight_t": weight_t}, {"x": pre_gelu})


# MMT, GELU, Add
@register_fusion_inference
def mmt_gelu_add_inf_fused(mmt: MMT, gelu: GELU, add: Add, x: nvte.Tensor):
    x = nvte.cast_checked(x, mmt.x_dtype)
    weight = nvte.cast_checked(mmt.weight, mmt.weight_dtype)
    bias = nvte.cast_checked(add.bias, add.bias_dtype)

    _, y = nvte.matmul_transpose_gelu_add(x, weight, bias)

    return y


@register_fusion_forward
def mmt_gelu_add_fwd_fused(mmt: MMT, gelu: GELU, add: Add, x: nvte.Tensor):
    (x, x_t), (weight, weight_t) = nvte.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (mmt.weight, mmt.weight_dtype)
    )
    bias = nvte.cast_checked(add.bias, add.bias_dtype)

    pre_gelu, y = nvte.matmul_transpose_gelu_add(x, weight, bias)

    return y, ({"x_t": x_t, "weight_t": weight_t}, {"x": pre_gelu})


# MMT, Add, Add
@register_fusion_inference
def mmt_add_add_inf_fused(mmt: MMT, add1: Add, add2: Add, x: nvte.Tensor):
    x = nvte.cast_checked(x, mmt.x_dtype)
    weight = nvte.cast_checked(mmt.weight, mmt.weight_dtype)
    bias1 = nvte.cast_checked(add1.bias, add1.bias_dtype)
    bias2 = nvte.cast_checked(add2.bias, add2.bias_dtype)

    y = nvte.matmul_transpose_add_add(x, weight, bias1, bias2)

    return y


@register_fusion_forward
def mmt_add_add_fwd_fused(
    mmt: MMT, add1: Add, add2: Add, x: nvte.Tensor
) -> tuple[nvte.Tensor, tuple[Context, Context, Context]]:
    (x, x_t), (weight, weight_t) = nvte.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (mmt.weight, mmt.weight_dtype)
    )
    bias1 = nvte.cast_checked(add1.bias, add1.bias_dtype)
    bias2 = nvte.cast_checked(add2.bias, add2.bias_dtype)

    y = nvte.matmul_transpose_add_add(x, weight, bias1, bias2)

    return y, ({"x_t": x_t, "weight_t": weight_t}, {}, {})


# MMT, Add, GELU, Add
@register_fusion_inference
def mmt_add_gelu_add_inf_fused(
    mmt: MMT, add1: Add, gelu: GELU, add2: Add, x: nvte.Tensor
):
    x = nvte.cast_checked(x, mmt.x_dtype)
    weight = nvte.cast_checked(mmt.weight, mmt.weight_dtype)
    bias1 = nvte.cast_checked(add1.bias, add1.bias_dtype)
    bias2 = nvte.cast_checked(add2.bias, add2.bias_dtype)

    _, y = nvte.matmul_transpose_add_gelu_add(x, weight, bias1, bias2)

    return y


@register_fusion_forward
def mmt_add_gelu_add_fwd_fused(
    mmt: MMT, add1: Add, gelu: GELU, add2: Add, x: nvte.Tensor
) -> tuple[nvte.Tensor, tuple[Context, Context, Context, Context]]:
    (x, x_t), (weight, weight_t) = nvte.multi_cast_transpose_checked(
        (x, mmt.x_dtype), (mmt.weight, mmt.weight_dtype)
    )
    bias1 = nvte.cast_checked(add1.bias, add1.bias_dtype)
    bias2 = nvte.cast_checked(add2.bias, add2.bias_dtype)

    pre_gelu, y = nvte.matmul_transpose_add_gelu_add(x, weight, bias1, bias2)

    return y, (
        {"x_t": x_t, "weight_t": weight_t},
        {},
        {"x": pre_gelu},
        {},
    )


# MMT, GEGLU
@register_fusion_backward
def mmt_geglu_bwd_fused(
    mmt: MMT, geglu: GeGLU, mmt_ctx: Context, geglu_ctx: Context, grad: nvte.Tensor
) -> tuple[nvte.Tensor, tuple[Grads, Grads]]:
    x_t, weight_t, pre_geglu = mmt_ctx["x_t"], mmt_ctx["weight_t"], geglu_ctx["x"]
    dy, dy_t = nvte.cast_transpose_dgeglu_checked(grad, pre_geglu, mmt.dy_dtype)

    dx = nvte.matmul_transpose(dy, weight_t, mmt.dx_dtype or geglu.dx_dtype or dy.dtype)
    dweight = nvte.matmul_transpose(x_t, dy_t, mmt.dweight_dtype or mmt.weight.dtype)

    return dx, ([dweight], [])


# fusion function names (ex. mmt_add_bwd_fused) are for debugging only, as they are called from a dictionary like FUSIONS_FWD
__all__ = []
