# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX te custom call"""

from typing import Tuple
from functools import partial, reduce
import operator
import numpy as np
from jaxlib.mhlo_helpers import custom_call
import jax.numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, mlir
from jax.interpreters.mlir import ir, dtype_to_ir_type

import transformer_engine_jax
from transformer_engine_jax import DType as TEDType

for _name, _value in transformer_engine_jax.te_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")


def _te_dtype_to_jax_dtype(te_dtype):
    assert isinstance(te_dtype, TEDType)
    if te_dtype == TEDType.kFloat32:
        return np.float32
    if te_dtype == TEDType.kFloat16:
        return np.float16
    if te_dtype == TEDType.kBFloat16:
        return jnp.bfloat16
    if te_dtype == TEDType.kInt32:
        return np.int32
    return np.int8


def _jax_dtype_to_te_dtype(jax_dtype):
    if jax_dtype == np.float32:
        return TEDType.kFloat32
    if jax_dtype == np.float16:
        return TEDType.kFloat16
    if jax_dtype == jnp.bfloat16:
        return TEDType.kBFloat16
    raise ValueError(f"Not support the jax dtype: {jax_dtype}")


def te_transpose(inputs: jnp.ndarray, dtype: TEDType) -> jnp.ndarray:
    """
    transpose wrapper
    """
    return te_transpose_p.bind(inputs, dtype=dtype)


def te_transpose_abstract(inputs, *, dtype):
    """
    te_transpose abstract
    """
    input_dtype = dtypes.canonicalize_dtype(inputs.dtype)
    output_dtype = _te_dtype_to_jax_dtype(dtype)

    assert len(inputs.shape) == 2
    assert isinstance(dtype, TEDType)
    assert input_dtype == output_dtype

    return ShapedArray((inputs.shape[1], inputs.shape[0]),
                       input_dtype,
                       named_shape=inputs.named_shape)


def te_transpose_cuda_lowering(ctx, inputs, *, dtype):
    """
    te_transpose cuda lowering
    """

    i_aval = ctx.avals_in[0]
    assert i_aval.dtype in [np.float32, np.float16, jnp.bfloat16, jnp.int8]

    i_type = ir.RankedTensorType(inputs.type)
    i_shape = i_type.shape

    inter_dtype = dtype_to_ir_type(np.dtype(_te_dtype_to_jax_dtype(dtype)))

    opaque = transformer_engine_jax.build_te_mat_descriptor(
        i_shape[0], i_shape[1], dtype, dtype)

    out = custom_call(b"te_transpose", [
        ir.RankedTensorType.get([i_shape[1], i_shape[0]], inter_dtype),
    ], [inputs],
                      backend_config=opaque,
                      operand_layouts=[[1, 0]],
                      result_layouts=[[1, 0]],
                      has_side_effect=False)
    return [out]


te_transpose_p = core.Primitive("te_transpose")
te_transpose_p.multiple_results = False
te_transpose_p.def_impl(partial(xla.apply_primitive, te_transpose_p))
te_transpose_p.def_abstract_eval(te_transpose_abstract)
mlir.register_lowering(te_transpose_p,
                       te_transpose_cuda_lowering,
                       platform='cuda')


def te_cast_transpose(
        inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
        scale_inv: jnp.ndarray,
        out_ctype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose wrapper
    """
    return te_cast_transpose_p.bind(inputs,
                                    amax,
                                    scale,
                                    scale_inv,
                                    out_ctype=out_ctype)


def te_cast_transpose_abstract(inputs, amax, scale, scale_inv, *, out_ctype):
    """
    te_cast_transpose_p abstract
    """
    dtype = dtypes.canonicalize_dtype(inputs.dtype)
    assert dtype in [np.float32, np.float16, jnp.bfloat16]
    assert amax.dtype == np.float32
    assert scale.dtype == np.float32
    assert scale_inv.dtype == np.float32
    out_dtype = _te_dtype_to_jax_dtype(out_ctype)
    # input_cast, input_cast_trans, amax
    return (ShapedArray((inputs.shape[0], inputs.shape[1]),
                        out_dtype,
                        named_shape=inputs.named_shape),
            ShapedArray((inputs.shape[1], inputs.shape[0]),
                        out_dtype,
                        named_shape=inputs.named_shape),
            ShapedArray((1, ), amax.dtype, named_shape=amax.named_shape))


def te_cast_transpose_cuda_lowering(ctx, inputs, amax, scale, scale_inv, *,
                                    out_ctype):
    """
    te_cast_transpose_p lowering rules
    """
    i_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
    assert i_aval.dtype in [np.float32, np.float16, jnp.bfloat16]
    assert amax_aval.dtype == np.float32
    assert scale_aval.dtype == np.float32
    assert scale_inv_aval.dtype == np.float32
    i_type = ir.RankedTensorType(inputs.type)
    i_shape = i_type.shape
    out_dtype = dtype_to_ir_type(np.dtype(_te_dtype_to_jax_dtype(out_ctype)))
    amax_type = ir.RankedTensorType(amax.type).element_type

    opaque = transformer_engine_jax.build_te_mat_descriptor(
        i_shape[0], i_shape[1], _jax_dtype_to_te_dtype(i_aval.dtype),
        out_ctype)
    out = custom_call(b"te_cast_transpose", [
        ir.RankedTensorType.get([i_shape[0], i_shape[1]], out_dtype),
        ir.RankedTensorType.get([i_shape[1], i_shape[0]], out_dtype),
        ir.RankedTensorType.get((1, ), amax_type),
    ], [inputs, amax, scale, scale_inv],
                      backend_config=opaque,
                      operand_layouts=[[1, 0], [0], [0], [0]],
                      result_layouts=[[1, 0], [1, 0], [0]],
                      has_side_effect=False)
    return out


te_cast_transpose_p = core.Primitive("te_cast_transpose")
te_cast_transpose_p.multiple_results = True
te_cast_transpose_p.def_impl(partial(xla.apply_primitive, te_cast_transpose_p))
te_cast_transpose_p.def_abstract_eval(te_cast_transpose_abstract)
mlir.register_lowering(te_cast_transpose_p,
                       te_cast_transpose_cuda_lowering,
                       platform='cuda')


def te_gated_gelu(
        inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
        scale_inv: jnp.ndarray,
        out_ctype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast gated gelu wrapper
    """
    return te_gated_gelu_p.bind(inputs,
                                amax,
                                scale,
                                scale_inv,
                                out_ctype=out_ctype)


def te_gated_gelu_abstract(inputs, amax, scale, scale_inv, *, out_ctype):
    """
    te_gated_gelu_p abstract
    """
    dtype = dtypes.canonicalize_dtype(inputs.dtype)
    assert dtype in [np.float32, np.float16, jnp.bfloat16]
    assert amax.dtype == np.float32
    assert scale.dtype == np.float32
    assert scale_inv.dtype == np.float32
    out_dtype = _te_dtype_to_jax_dtype(out_ctype)
    # input_cast, input_cast_trans, amax
    return (ShapedArray((inputs.shape[0], inputs.shape[1] // 2),
                        out_dtype,
                        named_shape=inputs.named_shape),
            ShapedArray((1, ), amax.dtype, named_shape=amax.named_shape))


def te_gated_gelu_cuda_lowering(ctx, inputs, amax, scale, scale_inv, *,
                                out_ctype):
    """
    te_gated_gelu_p lowering rules
    """
    i_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
    assert i_aval.dtype in [np.float32, np.float16, jnp.bfloat16]
    assert amax_aval.dtype == np.float32
    assert scale_aval.dtype == np.float32
    assert scale_inv_aval.dtype == np.float32
    i_type = ir.RankedTensorType(inputs.type)
    i_shape = i_type.shape
    out_dtype = dtype_to_ir_type(np.dtype(_te_dtype_to_jax_dtype(out_ctype)))
    amax_type = ir.RankedTensorType(amax.type).element_type

    opaque = transformer_engine_jax.build_te_mat_descriptor(
        i_shape[0], i_shape[1] // 2, _jax_dtype_to_te_dtype(i_aval.dtype),
        out_ctype)
    out = custom_call(b"te_gated_gelu", [
        ir.RankedTensorType.get([i_shape[0], i_shape[1] // 2], out_dtype),
        ir.RankedTensorType.get((1, ), amax_type),
    ], [inputs, amax, scale, scale_inv],
                      backend_config=opaque,
                      operand_layouts=[[1, 0], [0], [0], [0]],
                      result_layouts=[[1, 0], [0]],
                      has_side_effect=False)
    return out


te_gated_gelu_p = core.Primitive("te_gated_gelu")
te_gated_gelu_p.multiple_results = True
te_gated_gelu_p.def_impl(partial(xla.apply_primitive, te_gated_gelu_p))
te_gated_gelu_p.def_abstract_eval(te_gated_gelu_abstract)
mlir.register_lowering(te_gated_gelu_p,
                       te_gated_gelu_cuda_lowering,
                       platform='cuda')


def te_cast_transpose_dgated_gelu(
        inputs: jnp.ndarray, gelu_inputs: jnp.ndarray, amax: jnp.ndarray,
        scale: jnp.ndarray, scale_inv: jnp.ndarray,
        out_ctype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose d_gated_gelu fusion wrapper
    """
    return te_cast_transpose_dgated_gelu_p.bind(inputs,
                                                gelu_inputs,
                                                amax,
                                                scale,
                                                scale_inv,
                                                out_ctype=out_ctype)


def te_cast_transpose_dgated_gelu_abstract(inputs, gelu_inputs, amax, scale,
                                           scale_inv, *, out_ctype):
    """
    te_cast_transpose_dgated_gelu_p abstract
    """
    dtype = dtypes.canonicalize_dtype(inputs.dtype)
    assert dtype in [np.float32, np.float16, jnp.bfloat16]
    assert gelu_inputs.dtype == dtype
    assert inputs.shape[0] == gelu_inputs.shape[0]
    assert inputs.shape[1] * 2 == gelu_inputs.shape[1]
    assert amax.dtype == np.float32
    assert scale.dtype == np.float32
    assert scale_inv.dtype == np.float32
    out_dtype = _te_dtype_to_jax_dtype(out_ctype)
    # input_cast, input_cast_trans, amax
    return (ShapedArray((gelu_inputs.shape[0], gelu_inputs.shape[1]),
                        out_dtype,
                        named_shape=inputs.named_shape),
            ShapedArray((gelu_inputs.shape[1], gelu_inputs.shape[0]),
                        out_dtype,
                        named_shape=inputs.named_shape),
            ShapedArray((1, ), amax.dtype, named_shape=amax.named_shape))


def te_cast_transpose_dgated_gelu_lowering(ctx, inputs, gelu_inputs, amax,
                                           scale, scale_inv, *, out_ctype):
    """
    te_cast_transpose_dgated_gelu_p lowering rules
    """
    i_aval, gi_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
    assert i_aval.dtype in [np.float32, np.float16, jnp.bfloat16]
    assert gi_aval.dtype == i_aval.dtype
    assert amax_aval.dtype == np.float32
    assert scale_aval.dtype == np.float32
    assert scale_inv_aval.dtype == np.float32
    i_type = ir.RankedTensorType(inputs.type)
    i_shape = i_type.shape
    gi_type = ir.RankedTensorType(gelu_inputs.type)
    gi_shape = gi_type.shape
    assert i_shape[0] == gi_shape[0]
    assert i_shape[1] * 2 == gi_shape[1]
    out_dtype = dtype_to_ir_type(np.dtype(_te_dtype_to_jax_dtype(out_ctype)))
    amax_type = ir.RankedTensorType(amax.type).element_type

    opaque = transformer_engine_jax.build_te_mat_descriptor(
        i_shape[0], i_shape[1], _jax_dtype_to_te_dtype(i_aval.dtype),
        out_ctype)
    out = custom_call(b"te_cast_transpose_dgated_gelu", [
        ir.RankedTensorType.get([gi_shape[0], gi_shape[1]], out_dtype),
        ir.RankedTensorType.get([gi_shape[1], gi_shape[0]], out_dtype),
        ir.RankedTensorType.get((1, ), amax_type),
    ], [inputs, gelu_inputs, amax, scale, scale_inv],
                      backend_config=opaque,
                      operand_layouts=[[1, 0], [1, 0], [0], [0], [0]],
                      result_layouts=[[1, 0], [1, 0], [0]],
                      has_side_effect=False)
    return out


te_cast_transpose_dgated_gelu_p = core.Primitive(
    "te_cast_transpose_dgated_gelu")
te_cast_transpose_dgated_gelu_p.multiple_results = True
te_cast_transpose_dgated_gelu_p.def_impl(
    partial(xla.apply_primitive, te_cast_transpose_dgated_gelu_p))
te_cast_transpose_dgated_gelu_p.def_abstract_eval(
    te_cast_transpose_dgated_gelu_abstract)
mlir.register_lowering(te_cast_transpose_dgated_gelu_p,
                       te_cast_transpose_dgated_gelu_lowering,
                       platform='cuda')


def te_gemm(A: jnp.ndarray, A_scale_inverse: jnp.ndarray, A_type: TEDType,
            transa: bool, B: jnp.ndarray, B_scale_inverse: jnp.ndarray,
            B_type: TEDType, transb: bool, D_type: TEDType) -> jnp.ndarray:
    """
    gemm wrapper
    """
    return te_gemm_p.bind(A,
                          B,
                          A_scale_inverse,
                          B_scale_inverse,
                          A_ctype=A_type,
                          B_ctype=B_type,
                          D_ctype=D_type,
                          transa=transa,
                          transb=transb)


def te_gemm_abstract(A, B, A_scale_inv, B_scale_inv, *, A_ctype, B_ctype,
                     D_ctype, transa, transb):
    """
    te_gemm_p abstract
    """
    atype = dtypes.canonicalize_dtype(A.dtype)
    btype = dtypes.canonicalize_dtype(B.dtype)
    assert atype == _te_dtype_to_jax_dtype(A_ctype)
    assert btype == _te_dtype_to_jax_dtype(B_ctype)
    assert A_scale_inv.dtype == np.float32
    assert B_scale_inv.dtype == np.float32

    m = A.shape[0] if transa else A.shape[1]
    k = A.shape[1] if transa else A.shape[0]
    n = B.shape[1] if transb else B.shape[0]
    assert (transb and k == B.shape[0]) or k == B.shape[1]

    out_dtype = _te_dtype_to_jax_dtype(D_ctype)
    return ShapedArray((n, m), out_dtype, named_shape=B.named_shape)


def te_gemm_cuda_lowering(ctx, A, B, A_scale_inv, B_scale_inv, *, A_ctype,
                          B_ctype, D_ctype, transa, transb):
    """
    te_gemm_p lowering rules
    """
    A_aval, B_aval, A_scale_inv_aval, B_scale_inv_aval = ctx.avals_in
    assert A_aval.dtype == _te_dtype_to_jax_dtype(A_ctype)
    assert B_aval.dtype == _te_dtype_to_jax_dtype(B_ctype)
    assert A_scale_inv_aval.dtype == np.float32
    assert B_scale_inv_aval.dtype == np.float32
    A_type = ir.RankedTensorType(A.type)
    B_type = ir.RankedTensorType(B.type)
    A_shape = A_type.shape
    B_shape = B_type.shape

    m = A_shape[0] if transa else A_shape[1]
    k = A_shape[1] if transa else A_shape[0]
    n = B_shape[1] if transb else B_shape[0]
    assert (transb and k == B_shape[0]) or k == B_shape[1]

    out_dtype = dtype_to_ir_type(np.dtype(_te_dtype_to_jax_dtype(D_ctype)))

    opaque = transformer_engine_jax.build_te_gemm_descriptor(
        A_shape[1], B_shape[0], A_shape[0], A_ctype, B_ctype, D_ctype, transa,
        transb)
    out = custom_call(b"te_gemm", [
        ir.RankedTensorType.get([n, m], out_dtype),
    ], [A, B, A_scale_inv, B_scale_inv],
                      backend_config=opaque,
                      operand_layouts=[[1, 0], [1, 0], [0], [0]],
                      result_layouts=[[1, 0]],
                      has_side_effect=False)
    return [out]


te_gemm_p = core.Primitive("te_gemm")
te_gemm_p.multiple_results = False
te_gemm_p.def_impl(partial(xla.apply_primitive, te_gemm_p))
te_gemm_p.def_abstract_eval(te_gemm_abstract)
mlir.register_lowering(te_gemm_p, te_gemm_cuda_lowering, platform='cuda')


def te_rmsnorm_fwd(x, gamma, epsilon):
    """
    Wrapper for TE rmsnorm fwd
    """
    return _rmsnorm_fwd_p.bind(x, gamma, epsilon=epsilon)


def _rmsnorm_fwd_abstract(x, gamma, *, epsilon):  # pylint: disable=unused-argument
    """
    RMSNorm fwd abstract
    """
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    rsigma_dtype = jnp.float32

    hidden = reduce(operator.mul, gamma.shape)
    n = reduce(operator.mul, x.shape) // hidden

    return (
        ShapedArray(x.shape, x_dtype, named_shape=x.named_shape),  # output
        ShapedArray((n, ), rsigma_dtype, named_shape=x.named_shape),  # rsigma
    )


def _rmsnorm_fwd_cuda_lowering(ctx, x, gamma, *, epsilon):  # pylint: disable=unused-argument
    """
    RMSNorm fwd lowering rules
    """
    x_aval, gamma_aval = ctx.avals_in
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(gamma.type)
    w_shape = w_type.shape
    iv_element_type = (ir.F32Type.get() if x_type.element_type in [
        ir.F16Type.get(), ir.BF16Type.get()
    ] else x_type.element_type)

    hidden = reduce(operator.mul, w_shape)
    n = reduce(operator.mul, x_shape) // hidden

    opaque = transformer_engine_jax.build_te_rmsnorm_descriptor(
        n,
        hidden,
        _jax_dtype_to_te_dtype(x_aval.dtype),
        _jax_dtype_to_te_dtype(gamma_aval.dtype),
        epsilon,
    )
    operand_layouts = [
        range(len(shape) - 1, -1, -1) for shape in [x_shape, w_shape]
    ]
    result_layouts = [
        range(len(shape) - 1, -1, -1) for shape in [x_shape, (n, )]
    ]
    out = custom_call(
        b"te_rmsnorm_forward",
        [
            ir.RankedTensorType.get(x_shape, w_type.element_type),
            ir.RankedTensorType.get((n, ), iv_element_type),
        ],
        [x, gamma],
        backend_config=opaque,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )
    return out


_rmsnorm_fwd_p = core.Primitive("te_rmsnorm_forward")
_rmsnorm_fwd_p.multiple_results = True
_rmsnorm_fwd_p.def_impl(partial(xla.apply_primitive, _rmsnorm_fwd_p))
_rmsnorm_fwd_p.def_abstract_eval(_rmsnorm_fwd_abstract)
mlir.register_lowering(
    _rmsnorm_fwd_p,
    _rmsnorm_fwd_cuda_lowering,
    platform="cuda",
)


def te_rmsnorm_fwd_fp8(x, gamma, amax, scale, scale_inverse, epsilon):
    """
    Wrapper for TE rmsnorm fwd (fp8 out)
    """
    return _rmsnorm_fwd_fp8_p.bind(x,
                                   gamma,
                                   amax,
                                   scale,
                                   scale_inverse,
                                   epsilon=epsilon)


def _rmsnorm_fwd_fp8_abstract(x, gamma, amax, scale, scale_inverse, *,
                              epsilon):  # pylint: disable=unused-argument
    """
    RMSNorm fwd (fp8 out) abstract
    """
    x_dtype = dtypes.canonicalize_dtype(x.dtype)

    assert x_dtype in [np.float32, np.float16, jnp.bfloat16]
    assert amax.dtype == np.float32
    assert scale.dtype == np.float32
    assert scale_inverse.dtype == np.float32

    output_dtype = np.int8
    rsigma_dtype = jnp.float32

    hidden = reduce(operator.mul, gamma.shape)
    n = reduce(operator.mul, x.shape) // hidden

    return (
        ShapedArray(x.shape, output_dtype,
                    named_shape=x.named_shape),  # output
        ShapedArray((n, ), rsigma_dtype, named_shape=x.named_shape),  # rsigma
        ShapedArray((1, ), amax.dtype, named_shape=amax.named_shape),  # amax
    )


def _rmsnorm_fwd_fp8_cuda_lowering(ctx, x, gamma, amax, scale, scale_inverse,
                                   *, epsilon):  # pylint: disable=unused-argument
    """
    RMSNorm fwd (fp8 out) lowering rules
    """
    x_aval, gamma_aval, amax_aval, scale_aval, scale_inverse_aval = ctx.avals_in

    assert x_aval.dtype in [np.float32, np.float16, jnp.bfloat16]
    assert amax_aval.dtype == np.float32
    assert scale_aval.dtype == np.float32
    assert scale_inverse_aval.dtype == np.float32

    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(gamma.type)
    w_shape = w_type.shape

    output_dtype = dtype_to_ir_type(np.dtype(np.int8))
    rsigma_type = ir.F32Type.get()
    amax_type = ir.RankedTensorType(amax.type).element_type

    hidden = reduce(operator.mul, w_shape)
    n = reduce(operator.mul, x_shape) // hidden

    opaque = transformer_engine_jax.build_te_rmsnorm_descriptor(
        n,
        hidden,
        _jax_dtype_to_te_dtype(x_aval.dtype),
        _jax_dtype_to_te_dtype(gamma_aval.dtype),
        epsilon,
    )
    operand_layouts = [
        range(len(shape) - 1, -1, -1) for shape in [x_shape, w_shape]
    ] + [0, 0, 0]
    result_layouts = [
        range(len(shape) - 1, -1, -1) for shape in [x_shape, (n, )]
    ] + [0]
    out = custom_call(
        b"te_rmsnorm_forward_fp8",
        [
            ir.RankedTensorType.get(x_shape, output_dtype),
            ir.RankedTensorType.get((n, ), rsigma_type),
            ir.RankedTensorType.get((1, ), amax_type),
        ],
        [x, gamma, amax, scale, scale_inverse],
        backend_config=opaque,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )
    return out


_rmsnorm_fwd_fp8_p = core.Primitive("te_rmsnorm_forward_fp8")
_rmsnorm_fwd_fp8_p.multiple_results = True
_rmsnorm_fwd_fp8_p.def_impl(partial(xla.apply_primitive, _rmsnorm_fwd_fp8_p))
_rmsnorm_fwd_fp8_p.def_abstract_eval(_rmsnorm_fwd_fp8_abstract)
mlir.register_lowering(
    _rmsnorm_fwd_fp8_p,
    _rmsnorm_fwd_fp8_cuda_lowering,
    platform="cuda",
)


def te_rmsnorm_bwd(g, invvar, x, gamma, epsilon):
    """
    Wrapper for TE rmsnorm bwd
    """
    return _rmsnorm_bwd_p.bind(g, invvar, x, gamma, epsilon=epsilon)


def _rmsnorm_bwd_abstract(grad_output, rsigma, x, gamma, *, epsilon):  # pylint: disable=unused-argument
    """
    RMSNorm bwd abstract
    """
    w_dtype = dtypes.canonicalize_dtype(gamma.dtype)
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    rsigma_dtype = dtypes.canonicalize_dtype(rsigma.dtype)

    hidden = reduce(operator.mul, gamma.shape)
    n = reduce(operator.mul, x.shape) // hidden

    assert dtypes.canonicalize_dtype(grad_output.dtype) == w_dtype
    assert grad_output.shape == x.shape
    assert rsigma.shape == (n, )
    assert rsigma_dtype == jnp.float32
    assert grad_output.named_shape == x.named_shape

    gamma_named_shape = (gamma_named_shape
                         if gamma.named_shape else grad_output.named_shape)
    return (
        ShapedArray(x.shape, x_dtype,
                    named_shape=grad_output.named_shape),  # grad input
        ShapedArray(gamma.shape, w_dtype,
                    named_shape=gamma_named_shape),  # grad gamma
    )


def _rmsnorm_bwd_cuda_lowering(ctx, grad_output, invvar, x, gamma, *, epsilon):  # pylint: disable=unused-argument
    """
    RMSNorm bwd lowering rules
    """
    _, _, x_aval, gamma_aval = ctx.avals_in
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(gamma.type)
    w_shape = w_type.shape

    hidden = reduce(lambda x, y: x * y, w_shape)
    n = reduce(lambda x, y: x * y, x_shape) // hidden

    opaque = transformer_engine_jax.build_te_rmsnorm_descriptor(
        n,
        hidden,
        _jax_dtype_to_te_dtype(x_aval.dtype),
        _jax_dtype_to_te_dtype(gamma_aval.dtype),
        epsilon,
    )
    operand_layouts = [
        range(len(shape) - 1, -1, -1)
        for shape in [x_shape, (n, ), x_shape, w_shape]
    ]
    result_layouts = [
        range(len(shape) - 1, -1, -1) for shape in [x_shape, w_shape]
    ]
    out = custom_call(
        b"te_rmsnorm_backward",
        [
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(w_shape, w_type.element_type),
        ],
        [grad_output, invvar, x, gamma],
        backend_config=opaque,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )
    return out


_rmsnorm_bwd_p = core.Primitive("te_rmsnorm_backward")
_rmsnorm_bwd_p.multiple_results = True
_rmsnorm_bwd_p.def_impl(partial(xla.apply_primitive, _rmsnorm_bwd_p))
_rmsnorm_bwd_p.def_abstract_eval(_rmsnorm_bwd_abstract)
mlir.register_lowering(
    _rmsnorm_bwd_p,
    _rmsnorm_bwd_cuda_lowering,
    platform="cuda",
)
