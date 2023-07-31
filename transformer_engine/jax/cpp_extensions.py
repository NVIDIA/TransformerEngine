# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te custom call"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple
from functools import partial, reduce
import operator
import warnings

import numpy as np
from jaxlib.hlo_helpers import custom_call
import jax.numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import xla, mlir
from jax.interpreters.mlir import ir, dtype_to_ir_type

import transformer_engine_jax
from transformer_engine_jax import DType as TEDType
from transformer_engine_jax import NVTE_Bias_Type
from transformer_engine_jax import NVTE_Mask_Type
from transformer_engine_jax import NVTE_QKV_Layout
from transformer_engine_jax import NVTE_Fused_Attn_Backend

for _name, _value in transformer_engine_jax.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")


def te_dtype_to_jax_dtype(te_dtype):
    """
    convert TE dtype to jax dtype
    """
    assert isinstance(te_dtype, TEDType)
    if te_dtype == TEDType.kFloat32:
        return jnp.float32
    if te_dtype == TEDType.kFloat16:
        return jnp.float16
    if te_dtype == TEDType.kBFloat16:
        return jnp.bfloat16
    if te_dtype == TEDType.kInt32:
        return jnp.int32
    if te_dtype == TEDType.kInt64:
        return jnp.int64
    return jnp.int8


def te_dtype_to_ir_dtype(te_dtype):
    """
    convert TE dtype to MLIR dtype
    """
    return dtype_to_ir_type(np.dtype(te_dtype_to_jax_dtype(te_dtype)))


def jax_dtype_to_te_dtype(jax_dtype):
    """
    convert jax dtype to TE dtype
    """
    if jax_dtype == jnp.float32:
        return TEDType.kFloat32
    if jax_dtype == jnp.float16:
        return TEDType.kFloat16
    if jax_dtype == jnp.bfloat16:
        return TEDType.kBFloat16
    raise ValueError(f"Not support the {jax_dtype=}")


def merge_named_shape(base, new):
    """
    merge named shape(ie, dict), no key conflict
    """
    output_named_shape = {**base}
    for key in new:
        if key in output_named_shape:
            assert output_named_shape[key] == new[key], \
                f"The value of named shape with a same name should be equal between" \
                f" base and new in merge_named_shape, but got base[{key}]=" \
                f"{output_named_shape[key]} and {new[key]=}"
        else:
            output_named_shape[key] = new[key]
    return output_named_shape


class BasePrimitive(metaclass=ABCMeta):
    """
    jax premitive
    """

    @staticmethod
    @abstractmethod
    def abstract():
        """
        to describe computing graph
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def lowering():
        """
        to describe MLIR
        """
        return NotImplemented


def register_primitive(cls):
    """
    register jax primitive
    """
    p = core.Primitive(cls.name)
    p.multiple_results = cls.multiple_results
    p.def_impl(partial(xla.apply_primitive, p))
    p.def_abstract_eval(cls.abstract)
    mlir.register_lowering(p, cls.lowering, platform='cuda')
    return p


@dataclass
class CustomCallArgsWrapper:
    """
    wrapper of XLA custom call args
    """

    def __init__(self,
                 output_types,
                 operands,
                 operand_shapes,
                 operand_specific_layouts=None,
                 output_specific_layouts=None):
        self.output_types = output_types
        self.operands = operands
        self.operand_layouts = CustomCallArgsWrapper.generate_layouts(operand_shapes,
                                                                      operand_specific_layouts)
        output_shapes = [x.shape for x in output_types]
        self.output_layouts = CustomCallArgsWrapper.generate_layouts(output_shapes,
                                                                     output_specific_layouts)

    @staticmethod
    def generate_layouts(shapes, specific_layouts):
        """
        setup layouts for XLA custom call
        """

        def default_layout(shape):
            return range(len(shape) - 1, -1, -1)

        if specific_layouts is None:
            specific_layouts = {}

        layouts = []
        for idx, shape in enumerate(shapes):
            if idx in specific_layouts:
                layouts.append(specific_layouts[idx])
            else:
                layouts.append(default_layout(shape))
        return layouts


def custom_caller(name, args, opaque, has_side_effect, **kwargs):
    """
    XLA custom call warpper
    """
    out = custom_call(name,
                      args.output_types,
                      args.operands,
                      operand_layouts=args.operand_layouts,
                      result_layouts=args.output_layouts,
                      backend_config=opaque,
                      has_side_effect=has_side_effect,
                      **kwargs)
    return out


class TransposePrimitive(BasePrimitive):
    """
    Transpose Primitive
    """
    name = "te_transpose"
    multiple_results = False

    @staticmethod
    def abstract(inputs, *, dtype):
        """
        _transpose abstract
        """
        in_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        out_dtype = te_dtype_to_jax_dtype(dtype)

        assert len(inputs.shape) == 2
        assert isinstance(dtype, TEDType)
        assert in_dtype == out_dtype

        return ShapedArray((inputs.shape[1], inputs.shape[0]),
                           in_dtype,
                           named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, *, dtype):
        """
        _transpose cuda lowering
        """

        in_aval = ctx.avals_in[0]
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16, jnp.int8]

        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        ir_out_dtype = te_dtype_to_ir_dtype(dtype)

        out_types = [ir.RankedTensorType.get([ir_in_shape[1], ir_in_shape[0]], ir_out_dtype)]
        operands = [inputs]
        operand_shapes = [ir_in_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        assert len(ir_in_shape) == 2
        opaque = transformer_engine_jax.pack_common_descriptor(ir_in_shape, dtype, dtype)

        out = custom_caller(TransposePrimitive.name, args, opaque, False)

        return [out]


_transpose_p = register_primitive(TransposePrimitive)


def transpose(inputs: jnp.ndarray, dtype: TEDType) -> jnp.ndarray:
    """
    transpose wrapper
    Assume input has two dimension shape
    """
    return _transpose_p.bind(inputs, dtype=dtype)


class CastTransposePrimitive(BasePrimitive):
    """
    Cast Transpose Primitive
    """
    name = "te_cast_transpose"
    multiple_results = True

    @staticmethod
    def abstract(inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert len(inputs.shape) == 2
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32
        out_dtype = te_dtype_to_jax_dtype(out_dtype)
        # input_cast, input_cast_trans, amax
        return (ShapedArray((inputs.shape[0], inputs.shape[1]),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((inputs.shape[1], inputs.shape[0]),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((1,), amax.dtype, named_shape=amax.named_shape))

    @staticmethod
    def lowering(ctx, inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_cast_transpose_p lowering rules
        """
        in_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [
            ir.RankedTensorType.get([ir_in_shape[0], ir_in_shape[1]], ir_out_dtype),
            ir.RankedTensorType.get([ir_in_shape[1], ir_in_shape[0]], ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        assert len(ir_in_shape) == 2
        opaque = transformer_engine_jax.pack_common_descriptor(ir_in_shape,
                                                               jax_dtype_to_te_dtype(in_aval.dtype),
                                                               out_dtype)

        out = custom_caller(CastTransposePrimitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={1: 2})

        return out


_cast_transpose_p = register_primitive(CastTransposePrimitive)


def cast_transpose(inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                   scale_inv: jnp.ndarray,
                   out_dtype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose wrapper
    Return two tensors, FP8(inputs) and FP8(inputs.T), which are scaled by `scale`
    """
    return _cast_transpose_p.bind(inputs, amax, scale, scale_inv, out_dtype=out_dtype)


class GatedGeluPrimitive(BasePrimitive):
    """
    Gated Gelu Primitive
    """
    name = "te_gated_gelu"
    multiple_results = False

    @staticmethod
    def abstract(inputs):
        """
        te_gated_gelu_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        inputs_shape = inputs.shape
        hidden_size = inputs_shape[-1]
        # In Transformer, batch_shape = (batch,  seqlen, )
        batch_shapes = inputs_shape[:-1]
        assert hidden_size % 2 == 0
        inputs_shape = inputs.shape
        out_shape = (batch_shapes) + (hidden_size // 2,)

        return ShapedArray(out_shape, dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs):
        """
        te_gated_gelu_p lowering rules
        """
        (in_aval,) = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        out_shape = ir_in_shape[:-1] + [ir_in_shape[-1] // 2]

        out_types = [
            ir.RankedTensorType.get(out_shape, ir_in_type.element_type),
        ]
        operands = [inputs]
        operand_shapes = [ir_in_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        hidden_size = ir_in_shape[-1]
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, ir_in_shape[:-1])
        in_dtype = jax_dtype_to_te_dtype(in_aval.dtype)
        opaque = transformer_engine_jax.pack_common_descriptor((batch_size, hidden_size // 2),
                                                               in_dtype, in_dtype)

        out = custom_caller(GatedGeluPrimitive.name, args, opaque, False)

        return [out]


_gated_gelu_p = register_primitive(GatedGeluPrimitive)


def gated_gelu(inputs: jnp.ndarray) -> jnp.ndarray:
    """
    gated gelu wrapper
    Return FP8(geglu(inputs))
    Assume inputs has two dimensions shape and the memory layout is (N, 2, H)
    """
    return _gated_gelu_p.bind(inputs)


class GatedGeluFp8Primitive(BasePrimitive):
    """
    Gated Gelu FP8 Primitive
    """
    name = "te_gated_gelu_fp8"
    multiple_results = True

    @staticmethod
    def abstract(inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_gated_gelu_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32
        out_dtype = te_dtype_to_jax_dtype(out_dtype)

        assert len(inputs.shape) == 2
        hidden_size = inputs.shape[1]
        batch_size = inputs.shape[0]    # In Transformer, batch_size = batch x seqlen

        # input_cast, input_cast_trans, amax
        return (ShapedArray((batch_size, hidden_size // 2),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((1,), amax.dtype, named_shape=amax.named_shape))

    @staticmethod
    def lowering(ctx, inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_gated_gelu_p lowering rules
        """
        in_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        hidden_size = ir_in_shape[1]
        batch_size = ir_in_shape[0]    # In Transformer, batch_size = batch x seqlen
        out_types = [
            ir.RankedTensorType.get([batch_size, hidden_size // 2], ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor(
            (ir_in_shape[0], ir_in_shape[1] // 2), jax_dtype_to_te_dtype(in_aval.dtype), out_dtype)

        out = custom_caller(GatedGeluFp8Primitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={1: 1})

        return out


_gated_gelu_fp8_p = register_primitive(GatedGeluFp8Primitive)


def gated_gelu_fp8(inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                   scale_inv: jnp.ndarray,
                   out_dtype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast gated gelu wrapper
    Return FP8(geglu(inputs))
    Assume inputs has two dimensions shape and the memory layout is (N, 2, H)
    """
    return _gated_gelu_fp8_p.bind(inputs, amax, scale, scale_inv, out_dtype=out_dtype)


class DgatedGeluPrimitive(BasePrimitive):
    """
    Dgated Gelu Primitive
    """
    name = "te_dgated_gelu"
    multiple_results = False

    @staticmethod
    def abstract(inputs, gelu_inputs):
        """
        te_dgated_gelu_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gelu_inputs.dtype == dtype
        for axis in range(len(inputs.shape) - 1):
            assert inputs.shape[axis] == gelu_inputs.shape[axis]

        i_hidden_size = inputs.shape[-1]
        g_hidden_szie = gelu_inputs.shape[-1]
        assert i_hidden_size * 2 == g_hidden_szie
        return ShapedArray(gelu_inputs.shape, dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, gelu_inputs):
        """
        te_dgated_gelu_p lowering rules
        """
        in_aval, gi_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gi_aval.dtype == in_aval.dtype
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        gi_type = ir.RankedTensorType(gelu_inputs.type)
        gi_shape = gi_type.shape
        for axis in range(len(ir_in_shape) - 1):
            assert ir_in_shape[axis] == gi_shape[axis]

        # In Transformer, batch_size = batch x seqlen
        ir_batch_szie = reduce(operator.mul, ir_in_shape[:-1])
        i_hidden_size = ir_in_shape[-1]
        g_hidden_szie = gi_shape[-1]
        assert i_hidden_size * 2 == g_hidden_szie
        out_dtype = ir_in_type.element_type
        out_shape = gi_shape

        out_types = [
            ir.RankedTensorType.get(out_shape, out_dtype),
        ]
        operands = [inputs, gelu_inputs]
        operand_shapes = [ir_in_shape, gi_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        in_dtype = jax_dtype_to_te_dtype(in_aval.dtype)
        opaque = transformer_engine_jax.pack_common_descriptor((ir_batch_szie, i_hidden_size),
                                                               in_dtype, in_dtype)

        out = custom_caller(DgatedGeluPrimitive.name, args, opaque, False)

        return [out]


_dgated_gelu_p = register_primitive(DgatedGeluPrimitive)


def dgated_gelu(inputs: jnp.ndarray, gelu_inputs: jnp.ndarray) -> jnp.ndarray:
    """
    dgated_gelu fusion wrapper
    Return dgeglu(inputs)
    """
    return _dgated_gelu_p.bind(inputs, gelu_inputs)


class DgatedGeluCastTransposePrimitive(BasePrimitive):
    """
    Dgated Gelu Cast Transpose Primitive
    """
    name = "te_dgated_gelu_cast_transpose"
    multiple_results = True

    @staticmethod
    def abstract(inputs, gelu_inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_dgated_gelu_cast_transpose_p abstract
        """
        dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gelu_inputs.dtype == dtype
        assert len(inputs.shape) == 2
        assert len(gelu_inputs.shape) == 2
        ir_batch_szie = inputs.shape[0]
        gi_batch_size = gelu_inputs.shape[0]
        assert ir_batch_szie == gi_batch_size
        ir_hidden_szie = inputs.shape[1]
        gi_hidden_size = gelu_inputs.shape[1]
        assert ir_hidden_szie * 2 == gi_hidden_size
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32
        out_dtype = te_dtype_to_jax_dtype(out_dtype)
        # input_cast, input_cast_trans, amax
        return (ShapedArray((gi_batch_size, gi_hidden_size),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((gi_hidden_size, gi_batch_size),
                            out_dtype,
                            named_shape=inputs.named_shape),
                ShapedArray((1,), amax.dtype, named_shape=amax.named_shape))

    @staticmethod
    def lowering(ctx, inputs, gelu_inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_dgated_gelu_cast_transpose_p lowering rules
        """
        in_aval, gi_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in
        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gi_aval.dtype == in_aval.dtype
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32
        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape
        gi_type = ir.RankedTensorType(gelu_inputs.type)
        gi_shape = gi_type.shape
        ir_batch_szie = ir_in_shape[0]
        gi_batch_size = gi_shape[0]
        assert ir_batch_szie == gi_batch_size
        ir_hidden_szie = ir_in_shape[1]
        gi_hidden_size = gi_shape[1]
        assert ir_hidden_szie * 2 == gi_hidden_size
        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [
            ir.RankedTensorType.get([gi_batch_size, gi_hidden_size], ir_out_dtype),
            ir.RankedTensorType.get([gi_hidden_size, gi_batch_size], ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [inputs, gelu_inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, gi_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor((ir_batch_szie, ir_hidden_szie),
                                                               jax_dtype_to_te_dtype(in_aval.dtype),
                                                               out_dtype)

        out = custom_caller(DgatedGeluCastTransposePrimitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={2: 2})

        return out


_dgated_gelu_cast_transpose_p = register_primitive(DgatedGeluCastTransposePrimitive)


def dgated_gelu_cast_transpose(inputs: jnp.ndarray, gelu_inputs: jnp.ndarray, amax: jnp.ndarray,
                               scale: jnp.ndarray, scale_inv: jnp.ndarray,
                               out_dtype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cast transpose d_gated_gelu fusion wrapper
    Return FP8(dgeglu(inputs))
    """
    return _dgated_gelu_cast_transpose_p.bind(inputs,
                                              gelu_inputs,
                                              amax,
                                              scale,
                                              scale_inv,
                                              out_dtype=out_dtype)


class GemmPrimitive(BasePrimitive):
    """
    Gemm Primitive
    """
    name = "te_gemm"
    multiple_results = False

    @staticmethod
    def abstract(A, B, A_scale_inv, B_scale_inv, *, A_dtype, B_dtype, D_dtype, transa, transb,
                 use_split_accumulator):    # pylint: disable=unused-argument
        """
        te_gemm_p abstract
        """
        atype = dtypes.canonicalize_dtype(A.dtype)
        btype = dtypes.canonicalize_dtype(B.dtype)
        assert atype == te_dtype_to_jax_dtype(A_dtype)
        assert btype == te_dtype_to_jax_dtype(B_dtype)
        assert A_scale_inv.dtype == jnp.float32
        assert B_scale_inv.dtype == jnp.float32

        m = A.shape[0] if transa else A.shape[1]
        k = A.shape[1] if transa else A.shape[0]
        n = B.shape[1] if transb else B.shape[0]
        assert (transb and k == B.shape[0]) or k == B.shape[1]

        out_dtype = te_dtype_to_jax_dtype(D_dtype)
        return ShapedArray((n, m),
                           out_dtype,
                           named_shape=merge_named_shape(A.named_shape, B.named_shape))

    @staticmethod
    def lowering(ctx, A, B, A_scale_inv, B_scale_inv, *, A_dtype, B_dtype, D_dtype, transa, transb,
                 use_split_accumulator):
        """
        te_gemm_p lowering rules
        """
        A_aval, B_aval, A_scale_inv_aval, B_scale_inv_aval = ctx.avals_in
        assert A_aval.dtype == te_dtype_to_jax_dtype(A_dtype)
        assert B_aval.dtype == te_dtype_to_jax_dtype(B_dtype)
        assert A_scale_inv_aval.dtype == jnp.float32
        assert B_scale_inv_aval.dtype == jnp.float32
        A_type = ir.RankedTensorType(A.type)
        B_type = ir.RankedTensorType(B.type)
        A_shape = A_type.shape
        B_shape = B_type.shape
        A_scale_inv_shape = ir.RankedTensorType(A_scale_inv.type).shape
        B_scale_inv_shape = ir.RankedTensorType(B_scale_inv.type).shape

        m = A_shape[0] if transa else A_shape[1]
        k = A_shape[1] if transa else A_shape[0]
        n = B_shape[1] if transb else B_shape[0]
        assert (transb and k == B_shape[0]) or k == B_shape[1]

        ir_out_dtype = dtype_to_ir_type(np.dtype(te_dtype_to_jax_dtype(D_dtype)))
        out_types = [
            ir.RankedTensorType.get([n, m], ir_out_dtype),
        ]
        operands = [A, B, A_scale_inv, B_scale_inv]
        operand_shapes = [A_shape, B_shape, A_scale_inv_shape, B_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        # m, n, k here should be equal to transa=False and transb=False,
        # due to te_gemm's implementation.
        # Therefore, m=A_shape[1], n=B_shape[0], k=A_shape[0]
        opaque = transformer_engine_jax.pack_gemm_descriptor(A_shape[1], B_shape[0], A_shape[0],
                                                             A_dtype, B_dtype, D_dtype, transa,
                                                             transb, use_split_accumulator)

        out = custom_caller(GemmPrimitive.name, args, opaque, False)

        return [out]


_gemm_p = register_primitive(GemmPrimitive)


def gemm(A: jnp.ndarray,
         A_scale_inv: jnp.ndarray,
         A_type: TEDType,
         transa: bool,
         B: jnp.ndarray,
         B_scale_inv: jnp.ndarray,
         B_type: TEDType,
         transb: bool,
         D_type: TEDType,
         use_split_accumulator: bool = False) -> jnp.ndarray:
    """
    gemm wrapper
    """
    return _gemm_p.bind(A,
                        B,
                        A_scale_inv,
                        B_scale_inv,
                        A_dtype=A_type,
                        B_dtype=B_type,
                        D_dtype=D_type,
                        transa=transa,
                        transb=transb,
                        use_split_accumulator=use_split_accumulator)


class LayerNormFwdPrimitive(BasePrimitive):
    """
    Layer Normalization Forward Primitive
    """
    name = "te_layernorm_forward"
    multiple_results = True

    @staticmethod
    def abstract(x, gamma, beta, **kwargs):    # pylint: disable=unused-argument
        """
        LayerNorm fwd abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x.dtype)
        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        mu_dtype = jnp.float32
        rsigma_dtype = jnp.float32

        assert gamma.size == beta.size
        hidden_size = gamma.size
        assert x.size % hidden_size == 0
        # In Transformer, batch_size = batch x seqlen
        batch_size = x.size // hidden_size

        return (
            ShapedArray(x.shape, x_dtype, named_shape=x.named_shape),    # output
            ShapedArray((batch_size,), mu_dtype, named_shape=x.named_shape),    # mu
            ShapedArray((batch_size,), rsigma_dtype, named_shape=x.named_shape),    # rsigma
        )

    @staticmethod
    def lowering(ctx, x, gamma, beta, *, zero_centered_gamma, epsilon):
        """
        LayerNorm fwd lowering rules
        """
        x_aval, gamma_aval, beta_aval = ctx.avals_in
        assert gamma_aval.dtype == beta_aval.dtype
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        b_type = ir.RankedTensorType(beta.type)
        b_shape = b_type.shape

        assert w_type == b_type
        assert w_shape == b_shape

        # Output shape is same as the input shape, but the output type is same as the weight type.
        # See ln_api.cpp
        out_shape = x_shape
        output_type = w_type.element_type
        ir_mu_dtype = ir.F32Type.get()
        ir_rsigma_dtype = ir.F32Type.get()

        hidden_size = reduce(operator.mul, w_shape)
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(out_shape, output_type),
            ir.RankedTensorType.get((batch_size,), ir_mu_dtype),
            ir.RankedTensorType.get((batch_size,), ir_rsigma_dtype),
        ]
        operands = [x, gamma, beta]
        operand_shapes = [x_shape, w_shape, b_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            zero_centered_gamma,
            epsilon,
        )

        out = custom_caller(LayerNormFwdPrimitive.name, args, opaque, False)

        return out


_layernorm_fwd_p = register_primitive(LayerNormFwdPrimitive)


def layernorm_fwd(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, zero_centered_gamma: bool,
                  epsilon: float):
    """
    Wrapper for TE layernorm fwd
    """
    return _layernorm_fwd_p.bind(x,
                                 gamma,
                                 beta,
                                 zero_centered_gamma=zero_centered_gamma,
                                 epsilon=epsilon)


class LayerNormFwdFp8Primitive(BasePrimitive):
    """
    Layer Normalization Forward FP8 Primitive
    """
    name = "te_layernorm_forward_fp8"
    multiple_results = True

    @staticmethod
    def abstract(
            x,
            gamma,
            beta,
            amax,
            scale,
            scale_inv,
            **kwargs    # pylint: disable=unused-argument
    ):
        """
        LayerNorm fwd (fp8 out) abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32

        out_dtype = jnp.int8
        mu_dtype = jnp.float32
        rsigma_dtype = jnp.float32

        assert gamma.size == beta.size

        hidden_szie = gamma.size
        # In Transformer, batch_size = batch x seqlen
        batch_size = x.size // hidden_szie

        return (
            ShapedArray(x.shape, out_dtype, named_shape=x.named_shape),    # output
            ShapedArray((batch_size,), mu_dtype, named_shape=x.named_shape),    # mu
            ShapedArray((batch_size,), rsigma_dtype, named_shape=x.named_shape),    # rsigma
            ShapedArray((1,), amax.dtype, named_shape=amax.named_shape),    # amax
        )

    @staticmethod
    def lowering(ctx, x, gamma, beta, amax, scale, scale_inv, *, zero_centered_gamma, epsilon):
        """
        LayerNorm fwd (fp8 out) lowering rules
        """
        x_aval, gamma_aval, beta_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert gamma_aval.dtype == beta_aval.dtype
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        b_type = ir.RankedTensorType(beta.type)
        b_shape = b_type.shape

        ir_out_dtype = dtype_to_ir_type(np.dtype(np.int8))
        ir_mu_dtype = ir.F32Type.get()
        ir_rsigma_dtype = ir.F32Type.get()
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        hidden_size = reduce(operator.mul, w_shape)
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, ir_out_dtype),
            ir.RankedTensorType.get((batch_size,), ir_mu_dtype),
            ir.RankedTensorType.get((batch_size,), ir_rsigma_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, gamma, beta, amax, scale, scale_inv]
        operand_shapes = [
            x_shape, w_shape, b_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape
        ]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            zero_centered_gamma,
            epsilon,
        )

        out = custom_caller(LayerNormFwdFp8Primitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={3: 3})

        return out


_layernorm_fwd_fp8_p = register_primitive(LayerNormFwdFp8Primitive)


def layernorm_fwd_fp8(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, amax: jnp.ndarray,
                      scale: jnp.ndarray, scale_inv: jnp.ndarray, zero_centered_gamma: bool,
                      epsilon: float):
    """
    Wrapper for TE layernorm fwd (fp8 out)
    """
    return _layernorm_fwd_fp8_p.bind(x,
                                     gamma,
                                     beta,
                                     amax,
                                     scale,
                                     scale_inv,
                                     zero_centered_gamma=zero_centered_gamma,
                                     epsilon=epsilon)


class LayerNormBwdPrimitive(BasePrimitive):
    """
    Layer Normalization Backward Primitive
    """
    name = "te_layernorm_backward"
    multiple_results = True

    @staticmethod
    def abstract(grad_output, mu, rsigma, x, gamma, **kwargs):    # pylint: disable=unused-argument
        """
        Layernorm bwd abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x.dtype)
        w_dtype = dtypes.canonicalize_dtype(gamma.dtype)
        mu_dtype = dtypes.canonicalize_dtype(mu.dtype)
        rsigma_dtype = dtypes.canonicalize_dtype(rsigma.dtype)

        hidden_size = gamma.size
        # In Transformer, batch_size = batch x seqlen
        batch_size = x.size // hidden_size

        assert dtypes.canonicalize_dtype(grad_output.dtype) == w_dtype
        assert grad_output.shape == x.shape
        assert mu.shape == rsigma.shape == (batch_size,)
        assert mu_dtype == rsigma_dtype == jnp.float32
        assert grad_output.named_shape == x.named_shape

        return (
            ShapedArray(x.shape, x_dtype, named_shape=grad_output.named_shape),    # grad input
            ShapedArray(gamma.shape, w_dtype, named_shape=gamma.named_shape),    # grad gamma
            ShapedArray(gamma.shape, w_dtype, named_shape=gamma.named_shape),    # grad beta
        )

    @staticmethod
    def lowering(ctx, grad_output, mu, rsigma, x, gamma, *, zero_centered_gamma, epsilon):
        """
        Layernorm bwd lowering rules
        """
        _, _, _, x_aval, gamma_aval = ctx.avals_in
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        b_type = ir.RankedTensorType(gamma.type)
        b_shape = b_type.shape
        assert w_type == b_type
        assert w_shape == b_shape

        go_shape = ir.RankedTensorType(grad_output.type).shape
        mu_shape = ir.RankedTensorType(mu.type).shape
        rsigma_shape = ir.RankedTensorType(rsigma.type).shape

        hidden_size = reduce(operator.mul, w_shape)
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(w_shape, w_type.element_type),
            ir.RankedTensorType.get(b_shape, b_type.element_type),
        ]
        operands = [grad_output, mu, rsigma, x, gamma]
        operand_shapes = [go_shape, mu_shape, rsigma_shape, x_shape, w_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            zero_centered_gamma,
            epsilon,
        )

        out = custom_caller(LayerNormBwdPrimitive.name, args, opaque, False)

        return out


_layernorm_bwd_p = register_primitive(LayerNormBwdPrimitive)


def layernorm_bwd(g: jnp.ndarray, mu: jnp.ndarray, rsigma: jnp.ndarray, x: jnp.ndarray,
                  gamma: jnp.ndarray, zero_centered_gamma: bool, epsilon: float):
    """
    Wrapper for TE layernorm bwd
    """
    return _layernorm_bwd_p.bind(g,
                                 mu,
                                 rsigma,
                                 x,
                                 gamma,
                                 zero_centered_gamma=zero_centered_gamma,
                                 epsilon=epsilon)


class RmsNormFwdPrimitive(BasePrimitive):
    """
    RMS Normalization Forward Primitive
    """
    name = "te_rmsnorm_forward"
    multiple_results = True

    @staticmethod
    def abstract(x, gamma, **kwargs):    # pylint: disable=unused-argument
        """
        RMSNorm fwd abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x.dtype)
        rsigma_dtype = jnp.float32

        hidden_size = gamma.size
        # In Transformer, batch_size = batch x seqlen
        batch_size = x.size // hidden_size

        return (
            ShapedArray(x.shape, x_dtype, named_shape=x.named_shape),    # output
            ShapedArray((batch_size,), rsigma_dtype, named_shape=x.named_shape),    # rsigma
        )

    @staticmethod
    def lowering(ctx, x, gamma, *, epsilon):
        """
        RMSNorm fwd lowering rules
        """
        x_aval, gamma_aval = ctx.avals_in
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        iv_element_type = ir.F32Type.get()

        hidden_size = reduce(operator.mul, w_shape)
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, w_type.element_type),
            ir.RankedTensorType.get((batch_size,), iv_element_type),
        ]
        operands = [x, gamma]
        operand_shapes = [x_shape, w_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            False,    # RMSNorm doesn't support zero_centered_gamma
            epsilon,
        )

        out = custom_caller(RmsNormFwdPrimitive.name, args, opaque, False)

        return out


_rmsnorm_fwd_p = register_primitive(RmsNormFwdPrimitive)


def rmsnorm_fwd(x: jnp.ndarray, gamma: jnp.ndarray, epsilon: float):
    """
    Wrapper for TE rmsnorm fwd
    """
    return _rmsnorm_fwd_p.bind(x, gamma, epsilon=epsilon)


class RmsNormFwdFp8Primitive(BasePrimitive):
    """
    RMS Normalization Forward FP8 Primitive
    """
    name = "te_rmsnorm_forward_fp8"
    multiple_results = True

    @staticmethod
    def abstract(
            x,
            gamma,
            amax,
            scale,
            scale_inv,
            **kwargs    # pylint: disable=unused-argument
    ):
        """
        RMSNorm fwd (fp8 out) abstract
        """
        x_dtype = dtypes.canonicalize_dtype(x.dtype)

        assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32

        out_dtype = jnp.int8
        rsigma_dtype = jnp.float32

        hidden_size = gamma.size
        # In Transformer, batch_size = batch x seqlen
        batch_size = x.size // hidden_size

        return (
            ShapedArray(x.shape, out_dtype, named_shape=x.named_shape),    # output
            ShapedArray((batch_size,), rsigma_dtype, named_shape=x.named_shape),    # rsigma
            ShapedArray((1,), amax.dtype, named_shape=amax.named_shape),    # amax
        )

    @staticmethod
    def lowering(ctx, x, gamma, amax, scale, scale_inv, *, epsilon):
        """
        RMSNorm fwd (fp8 out) lowering rules
        """
        x_aval, gamma_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert x_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape

        ir_out_dtype = dtype_to_ir_type(np.dtype(np.int8))
        ir_rsigma_dtype = ir.F32Type.get()
        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_dtype = ir_amax_type.element_type
        ir_amax_shape = ir_amax_type.shape
        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        hidden_size = reduce(operator.mul, w_shape)
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, ir_out_dtype),
            ir.RankedTensorType.get((batch_size,), ir_rsigma_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [x, gamma, amax, scale, scale_inv]
        operand_shapes = [x_shape, w_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            False,    # RMSNorm doesn't support zero_centered_gamma
            epsilon,
        )

        out = custom_caller(RmsNormFwdFp8Primitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={2: 2})

        return out


_rmsnorm_fwd_fp8_p = register_primitive(RmsNormFwdFp8Primitive)


def rmsnorm_fwd_fp8(x: jnp.ndarray, gamma: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                    scale_inv: jnp.ndarray, epsilon: float):
    """
    Wrapper for TE rmsnorm fwd (fp8 out)
    """
    return _rmsnorm_fwd_fp8_p.bind(x, gamma, amax, scale, scale_inv, epsilon=epsilon)


class RmsNormBwdPrimitive(BasePrimitive):
    """
    RMS Normalization Backward Primitive
    """
    name = "te_rmsnorm_backward"
    multiple_results = True

    @staticmethod
    def abstract(
            grad_output,
            rsigma,
            x,
            gamma,
            **kwargs    # pylint: disable=unused-argument
    ):
        """
        RMSNorm bwd abstract
        """
        w_dtype = dtypes.canonicalize_dtype(gamma.dtype)
        x_dtype = dtypes.canonicalize_dtype(x.dtype)
        rsigma_dtype = dtypes.canonicalize_dtype(rsigma.dtype)

        hidden_size = gamma.size
        # In Transformer, batch_size = batch x seqlen
        batch_size = x.size // hidden_size

        assert dtypes.canonicalize_dtype(grad_output.dtype) == w_dtype
        assert grad_output.shape == x.shape
        assert rsigma.shape == (batch_size,)
        assert rsigma_dtype == jnp.float32
        assert grad_output.named_shape == x.named_shape

        return (
            ShapedArray(x.shape, x_dtype, named_shape=grad_output.named_shape),    # grad input
            ShapedArray(gamma.shape, w_dtype, named_shape=gamma.named_shape),    # grad gamma
        )

    @staticmethod
    def lowering(ctx, grad_output, inv_var, x, gamma, *, epsilon):
        """
        RMSNorm bwd lowering rules
        """
        _, _, x_aval, gamma_aval = ctx.avals_in
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        go_shape = ir.RankedTensorType(grad_output.type).shape
        inv_var_shape = ir.RankedTensorType(inv_var.type).shape

        hidden_size = reduce(operator.mul, w_shape)
        # In Transformer, batch_size = batch x seqlen
        batch_size = reduce(operator.mul, x_shape) // hidden_size

        out_types = [
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(w_shape, w_type.element_type),
        ]
        operands = [grad_output, inv_var, x, gamma]
        operand_shapes = [go_shape, inv_var_shape, x_shape, w_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_norm_descriptor(
            batch_size,
            hidden_size,
            jax_dtype_to_te_dtype(x_aval.dtype),
            jax_dtype_to_te_dtype(gamma_aval.dtype),
            False,    # RMSNorm doesn't support zero_centered_gamma
            epsilon,
        )

        out = custom_caller(RmsNormBwdPrimitive.name, args, opaque, False)

        return out


_rmsnorm_bwd_p = register_primitive(RmsNormBwdPrimitive)


def rmsnorm_bwd(grad: jnp.ndarray, inv_var: jnp.ndarray, x: jnp.ndarray, gamma: jnp.ndarray,
                epsilon: float):
    """
    Wrapper for TE rmsnorm bwd
    """
    return _rmsnorm_bwd_p.bind(grad, inv_var, x, gamma, epsilon=epsilon)


class QuantizePrimitive(BasePrimitive):
    """
    Quantize Primitive
    """
    name = "te_quantize"
    multiple_results = True

    @staticmethod
    def abstract(inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_quantize abstract
        """
        in_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert in_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        assert isinstance(out_dtype, TEDType)
        out_dtype = te_dtype_to_jax_dtype(out_dtype)

        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32

        return (ShapedArray(inputs.shape, out_dtype, named_shape=inputs.named_shape),
                ShapedArray((1,), amax.dtype, named_shape=amax.named_shape))

    @staticmethod
    def lowering(ctx, inputs, amax, scale, scale_inv, *, out_dtype):
        """
        te_quantize lowering rules
        """
        in_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert in_aval.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape

        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_out_shape = ir_in_shape

        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_shape = ir_amax_type.shape
        ir_amax_dtype = ir_amax_type.element_type

        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [
            ir.RankedTensorType.get(ir_out_shape, ir_out_dtype),
            ir.RankedTensorType.get(ir_amax_shape, ir_amax_dtype),
        ]
        operands = [inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor(in_aval.shape,
                                                               jax_dtype_to_te_dtype(in_aval.dtype),
                                                               out_dtype)

        out = custom_caller(QuantizePrimitive.name,
                            args,
                            opaque,
                            False,
                            operand_output_aliases={1: 1})

        return out


_quantize_p = register_primitive(QuantizePrimitive)


def quantize(inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
             out_dtype: TEDType) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    quantize wrapper
    Return FP8 tensor
    """
    return _quantize_p.bind(inputs, amax, scale, scale_inv, out_dtype=out_dtype)


class DequantizePrimitive(BasePrimitive):
    """
    Dequantize Primitive
    """
    name = "te_dequantize"
    multiple_results = False

    @staticmethod
    def abstract(inputs, amax, scale, scale_inv, *, fp8_dtype, out_dtype):
        """
        te_dquantize abstract
        """
        in_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert in_dtype == jnp.int8
        assert isinstance(fp8_dtype, TEDType)

        assert isinstance(out_dtype, TEDType)
        out_dtype = te_dtype_to_jax_dtype(out_dtype)
        assert out_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

        assert amax.dtype == jnp.float32
        assert scale.dtype == jnp.float32
        assert scale_inv.dtype == jnp.float32

        return ShapedArray(inputs.shape, out_dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, amax, scale, scale_inv, *, fp8_dtype, out_dtype):
        """
        te_dquantize lowering rules
        """
        in_aval, amax_aval, scale_aval, scale_inv_aval = ctx.avals_in

        assert in_aval.dtype == jnp.int8
        assert amax_aval.dtype == jnp.float32
        assert scale_aval.dtype == jnp.float32
        assert scale_inv_aval.dtype == jnp.float32

        ir_in_type = ir.RankedTensorType(inputs.type)
        ir_in_shape = ir_in_type.shape

        ir_out_dtype = te_dtype_to_ir_dtype(out_dtype)
        ir_out_shape = ir_in_shape

        ir_amax_type = ir.RankedTensorType(amax.type)
        ir_amax_shape = ir_amax_type.shape

        ir_scale_shape = ir_amax_shape
        ir_scale_inv_shape = ir_amax_shape

        out_types = [ir.RankedTensorType.get(ir_out_shape, ir_out_dtype)]
        operands = [inputs, amax, scale, scale_inv]
        operand_shapes = [ir_in_shape, ir_amax_shape, ir_scale_shape, ir_scale_inv_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_common_descriptor(in_aval.shape, fp8_dtype, out_dtype)

        out = custom_caller(DequantizePrimitive.name, args, opaque, False)

        return [out]


_dequantize_p = register_primitive(DequantizePrimitive)


def dequantize(inputs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray, scale_inv: jnp.ndarray,
               fp8_dtype: TEDType, out_dtype: TEDType) -> jnp.ndarray:
    """
    dequantize wrapper
    Return FP16/BF16/FP32 tensor
    """
    return _dequantize_p.bind(inputs,
                              amax,
                              scale,
                              scale_inv,
                              fp8_dtype=fp8_dtype,
                              out_dtype=out_dtype)


class SoftmaxPrimitive(BasePrimitive):
    """
    Softmax Primitive
    """
    max_k_seqlen_supported = 4096

    @staticmethod
    def get_batch_per_block(k_seqlen: int) -> int:
        """Get batch per CTA in Softmax kernels"""
        threads_per_warp = 32
        threads_per_block = 128    # Depends on the kernel implmentation

        pow2 = 1 << (k_seqlen - 1).bit_length()
        warp_size = pow2 if pow2 < threads_per_warp else threads_per_warp
        batches_per_warp = 2 if pow2 <= 128 else 1
        warps_per_block = threads_per_block // warp_size
        batches_per_block = warps_per_block * batches_per_warp
        return batches_per_block

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        raise NotImplementedError

    @staticmethod
    def softmax_backward_abstract(grad_outputs, softmax_outputs, scale_factor=None):    # pylint: disable=unused-argument
        """
        MLIR abstract
        """
        grad_outputs_dtype = dtypes.canonicalize_dtype(grad_outputs.dtype)
        softmax_outputs_dtype = dtypes.canonicalize_dtype(softmax_outputs.dtype)
        assert grad_outputs_dtype == softmax_outputs_dtype
        assert grad_outputs_dtype in [jnp.float16, jnp.bfloat16]
        assert softmax_outputs_dtype in [jnp.float16, jnp.bfloat16]

        assert grad_outputs.shape == softmax_outputs.shape

        return ShapedArray(softmax_outputs.shape,
                           softmax_outputs_dtype,
                           named_shape=softmax_outputs.named_shape)

    @staticmethod
    def softmax_backward_lowering(name, ctx, grad_outputs, softmax_outputs, scale_factor):
        """
        MLIR abstract
        """
        grad_outputs_aval, _ = ctx.avals_in

        grad_outputs_type = ir.RankedTensorType(grad_outputs.type)
        grad_outputs_shape = grad_outputs_type.shape

        batch = grad_outputs_shape[0]
        pad_batch = batch    # unused
        heads = grad_outputs_shape[1]
        q_seqlen = grad_outputs_shape[2]
        k_seqlen = grad_outputs_shape[3]

        softmax_outputs_type = ir.RankedTensorType(softmax_outputs.type)
        softmax_outputs_shape = softmax_outputs_type.shape

        out_types = [
            ir.RankedTensorType.get(softmax_outputs_shape, softmax_outputs_type.element_type)
        ]
        operands = [grad_outputs, softmax_outputs]
        operand_shapes = [grad_outputs_shape, softmax_outputs_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(
            batch, pad_batch, heads, q_seqlen, k_seqlen,
            jax_dtype_to_te_dtype(grad_outputs_aval.dtype), scale_factor)

        out = custom_caller(name, args, opaque, False)

        return [out]


class ScaledSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Softmax Fwd Primitive
    """
    name = "te_scaled_softmax_forward"
    multiple_results = False

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        if (dtype in [jnp.float16, jnp.bfloat16]
                and 16 < k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        # k_seqlen must be 16 ~ 4096
                and q_seqlen % 4 == 0    # q_seqlen must be divisor of 4
                and attn_batches % 4 == 0    # batch * heads must be divisor of 4
           ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return q_seqlen % batch_per_block == 0
        return False

    @staticmethod
    def abstract(inputs, *, scale_factor):    # pylint: disable=unused-argument
        """
        te_scaled_softmax_forward abstract
        """
        shape_rank = 4    # batch, heads, q_seqlen and k_seqlen

        i_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert i_dtype in [jnp.float16, jnp.bfloat16]
        i_shape = inputs.shape
        assert len(i_shape) == shape_rank
        q_seqlen = i_shape[2]
        k_seqlen = i_shape[3]
        assert k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        assert q_seqlen > 1

        return ShapedArray(inputs.shape, i_dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, *, scale_factor):
        """
        te_scaled_softmax_forward lowering rules
        """
        shape_rank = 4    # batch, heads, q_seqlen and k_seqlen

        i_aval, = ctx.avals_in
        i_type = ir.RankedTensorType(inputs.type)
        i_shape = i_type.shape
        assert len(i_shape) == shape_rank
        batch = i_shape[0]
        pad_batch = batch
        heads = i_shape[1]
        q_seqlen = i_shape[2]
        k_seqlen = i_shape[3]

        out_types = [ir.RankedTensorType.get(i_shape, i_type.element_type)]
        operands = [inputs]
        operand_shapes = [i_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(batch, pad_batch, heads, q_seqlen,
                                                                k_seqlen,
                                                                jax_dtype_to_te_dtype(i_aval.dtype),
                                                                scale_factor)

        out = custom_caller(ScaledSoftmaxFwdPrimitive.name, args, opaque, False)

        return [out]


_scaled_softmax_fwd_p = register_primitive(ScaledSoftmaxFwdPrimitive)


def scaled_softmax_fwd(inputs: jnp.ndarray, scale_factor: float) -> jnp.ndarray:
    """
    scaled_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return _scaled_softmax_fwd_p.bind(inputs, scale_factor=scale_factor)


class ScaledSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Softmax Bwd Primitive
    """
    name = "te_scaled_softmax_backward"
    multiple_results = False

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(batch, heads, q_seqlen, k_seqlen,
                                                             dtype)

    @staticmethod
    def abstract(grad_outputs, softmax_outputs, *, scale_factor):
        """
        te_scaled_softmax_backward abstract
        """
        return SoftmaxPrimitive.softmax_backward_abstract(grad_outputs, softmax_outputs,
                                                          scale_factor)

    @staticmethod
    def lowering(ctx, grad_outputs, softmax_outputs, *, scale_factor):
        """
        te_scaled_softmax_backward lowering rules
        """
        out = SoftmaxPrimitive.softmax_backward_lowering(ScaledSoftmaxBwdPrimitive.name, ctx,
                                                         grad_outputs, softmax_outputs,
                                                         scale_factor)

        return out    # out is iterable already


_scaled_softmax_bwd_p = register_primitive(ScaledSoftmaxBwdPrimitive)


def scaled_softmax_bwd(grad_outputs: jnp.ndarray, softmax_outputs: jnp.ndarray,
                       scale_factor: float) -> jnp.ndarray:
    """
    scaled_softmax_backward wrapper
    Return FP16/BF16 tensor
    """
    return _scaled_softmax_bwd_p.bind(grad_outputs, softmax_outputs, scale_factor=scale_factor)


class ScaledMaskedSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Masked Softmax Fwd Primitive
    """
    name = "te_scaled_masked_softmax_forward"
    multiple_results = False

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        if (dtype in [jnp.float16, jnp.bfloat16]
                and 16 < k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        # k_seqlen must be 16 ~ 4096
                and q_seqlen % 4 == 0    # q_seqlen must be divisor of 4
                and attn_batches % 4 == 0    # batch * heads must be divisor of 4
           ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return q_seqlen % batch_per_block == 0
        return False

    @staticmethod
    def abstract(inputs, mask, *, scale_factor):    # pylint: disable=unused-argument
        """
        te_scaled_masked_softmax_forward abstract
        """
        shape_rank = 4    # batch, heads, q_seqlen and k_seqlen

        i_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert i_dtype in [jnp.float16, jnp.bfloat16]
        i_shape = inputs.shape
        assert len(i_shape) == shape_rank
        batch = i_shape[0]
        q_seqlen = i_shape[2]
        k_seqlen = i_shape[3]
        assert k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        assert q_seqlen > 1

        mask_dtype = dtypes.canonicalize_dtype(mask.dtype)
        assert mask_dtype in [
            jnp.uint8,
        ]
        mask_shape = mask.shape
        assert len(mask_shape) == shape_rank
        pad_batch = mask_shape[0]
        assert pad_batch in (1, batch)    # 1 means broadcast
        assert mask_shape[1] == 1    # 1 means broadcast
        assert mask_shape[2] == q_seqlen
        assert mask_shape[3] == k_seqlen

        return ShapedArray(inputs.shape, i_dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, mask, *, scale_factor):
        """
        te_scaled_masked_softmax_forward lowering rules
        """
        shape_rank = 4    # batch, heads, q_seqlen and k_seqlen

        i_aval, _ = ctx.avals_in
        i_type = ir.RankedTensorType(inputs.type)
        i_shape = i_type.shape
        assert len(i_shape) == shape_rank
        batch = i_shape[0]
        heads = i_shape[1]
        q_seqlen = i_shape[2]
        k_seqlen = i_shape[3]

        mask_type = ir.RankedTensorType(mask.type)
        mask_shape = mask_type.shape
        assert len(mask_shape) == shape_rank
        pad_batch = mask_shape[0]

        out_types = [ir.RankedTensorType.get(i_shape, i_type.element_type)]
        operands = [inputs, mask]
        operand_shapes = [i_shape, mask_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(batch, pad_batch, heads, q_seqlen,
                                                                k_seqlen,
                                                                jax_dtype_to_te_dtype(i_aval.dtype),
                                                                scale_factor)

        out = custom_caller(ScaledMaskedSoftmaxFwdPrimitive.name, args, opaque, False)

        return [out]


_scaled_masked_softmax_fwd_p = register_primitive(ScaledMaskedSoftmaxFwdPrimitive)


def scaled_masked_softmax_fwd(inputs: jnp.ndarray, mask: jnp.ndarray,
                              scale_factor: float) -> jnp.ndarray:
    """
    scaled_masked_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return _scaled_masked_softmax_fwd_p.bind(inputs, mask, scale_factor=scale_factor)


class ScaledMaskedSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Masked Softmax Bwd Primitive
    """
    name = "te_scaled_masked_softmax_backward"
    multiple_results = False

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(batch, heads, q_seqlen, k_seqlen,
                                                             dtype)

    @staticmethod
    def abstract(grad_outputs, softmax_outputs, *, scale_factor):
        """
        te_scaled_masked_softmax_backward abstract
        """
        return SoftmaxPrimitive.softmax_backward_abstract(grad_outputs, softmax_outputs,
                                                          scale_factor)

    @staticmethod
    def lowering(ctx, grad_outputs, softmax_outputs, *, scale_factor):
        """
        te_scaled_masked_softmax_backward lowering rules
        """
        out = SoftmaxPrimitive.softmax_backward_lowering(ScaledMaskedSoftmaxBwdPrimitive.name, ctx,
                                                         grad_outputs, softmax_outputs,
                                                         scale_factor)

        return out    # out is iterable already


_scaled_masked_softmax_bwd_p = register_primitive(ScaledMaskedSoftmaxBwdPrimitive)


def scaled_masked_softmax_bwd(grad_outputs: jnp.ndarray, softmax_outputs: jnp.ndarray,
                              scale_factor: float) -> jnp.ndarray:
    """
    scaled_masked_softmax_backward wrapper
    Return FP16/BF16 tensor
    """
    return _scaled_masked_softmax_bwd_p.bind(grad_outputs,
                                             softmax_outputs,
                                             scale_factor=scale_factor)


class ScaledUpperTriangMaskedSoftmaxFwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Upper Triang Masked Softmax Fwd Primitive
    """
    name = "te_scaled_upper_triang_masked_softmax_forward"
    multiple_results = False

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        attn_batches = batch * heads

        if (dtype in [jnp.float16, jnp.bfloat16]
                and 16 < k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        # k_seqlen must be 16 ~ 4096
                and q_seqlen % 4 == 0    # q_seqlen must be divisor of 4
                and attn_batches % 4 == 0    # batch * heads must be divisor of 4
           ):
            if 0 <= k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported:
                batch_per_block = SoftmaxPrimitive.get_batch_per_block(k_seqlen)
                return attn_batches % batch_per_block == 0
        return False

    @staticmethod
    def abstract(inputs, *, scale_factor):    # pylint: disable=unused-argument
        """
        te_scaled_upper_triang_masked_softmax_forward abstract
        """
        shape_rank = 4    # batch, heads, q_seqlen and k_seqlen

        i_dtype = dtypes.canonicalize_dtype(inputs.dtype)
        assert i_dtype in [jnp.float16, jnp.bfloat16]
        i_shape = inputs.shape
        assert len(i_shape) == shape_rank
        q_seqlen = i_shape[2]
        k_seqlen = i_shape[3]
        assert q_seqlen == k_seqlen
        assert k_seqlen <= SoftmaxPrimitive.max_k_seqlen_supported
        assert q_seqlen > 1

        return ShapedArray(inputs.shape, i_dtype, named_shape=inputs.named_shape)

    @staticmethod
    def lowering(ctx, inputs, *, scale_factor):
        """
        te_scaled_upper_triang_masked_softmax_forward lowering rules
        """
        shape_rank = 4    # batch, heads, q_seqlen and k_seqlen

        i_aval, = ctx.avals_in
        i_type = ir.RankedTensorType(inputs.type)
        i_shape = i_type.shape
        assert len(i_shape) == shape_rank
        batch = i_shape[0]
        pad_batch = batch
        heads = i_shape[1]
        q_seqlen = i_shape[2]
        k_seqlen = i_shape[3]

        out_types = [ir.RankedTensorType.get(i_shape, i_type.element_type)]
        operands = [inputs]
        operand_shapes = [i_shape]
        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_softmax_descriptor(batch, pad_batch, heads, q_seqlen,
                                                                k_seqlen,
                                                                jax_dtype_to_te_dtype(i_aval.dtype),
                                                                scale_factor)

        out = custom_caller(ScaledUpperTriangMaskedSoftmaxFwdPrimitive.name, args, opaque, False)

        return [out]

_scaled_upper_triang_masked_softmax_fwd_p = \
    register_primitive(ScaledUpperTriangMaskedSoftmaxFwdPrimitive)


def scaled_upper_triang_masked_softmax_fwd(inputs: jnp.ndarray, scale_factor: float) -> jnp.ndarray:
    """
    scaled_upper_triang_masked_softmax_forward wrapper
    Return FP16/BF16 tensor
    """
    return _scaled_upper_triang_masked_softmax_fwd_p.bind(inputs, scale_factor=scale_factor)


class ScaledUpperTriangMaskedSoftmaxBwdPrimitive(SoftmaxPrimitive):
    """
    Scaled Upper Triang Masked Softmax Bwd Primitive
    """
    name = "te_scaled_upper_triang_masked_softmax_backward"
    multiple_results = False

    @staticmethod
    def is_kernel_available(batch: int, heads: int, q_seqlen: int, k_seqlen: int,
                            dtype: jnp.dtype) -> bool:
        """Check Softmax kernel availability based on size"""
        return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype)

    @staticmethod
    def abstract(grad_outputs, softmax_outputs, *, scale_factor):
        """
        te_scaled_upper_triang_masked_softmax_backward abstract
        """
        return SoftmaxPrimitive.softmax_backward_abstract(grad_outputs, softmax_outputs,
                                                          scale_factor)

    @staticmethod
    def lowering(ctx, grad_outputs, softmax_outputs, *, scale_factor):
        """
        te_scaled_upper_triang_masked_softmax_backward lowering rules
        """
        out = SoftmaxPrimitive.softmax_backward_lowering(
            ScaledUpperTriangMaskedSoftmaxBwdPrimitive.name, ctx, grad_outputs, softmax_outputs,
            scale_factor)

        return out    # out is iterable already

_scaled_upper_triang_masked_softmax_bwd_p = \
    register_primitive(ScaledUpperTriangMaskedSoftmaxBwdPrimitive)


def scaled_upper_triang_masked_softmax_bwd(grad_outputs: jnp.ndarray, softmax_outputs: jnp.ndarray,
                                           scale_factor: float) -> jnp.ndarray:
    """
    scaled_upper_triang_masked_softmax_backward wrapper
    Return FP16/BF16 tensor
    """
    return _scaled_upper_triang_masked_softmax_bwd_p.bind(grad_outputs,
                                                          softmax_outputs,
                                                          scale_factor=scale_factor)


@dataclass(frozen=True)
class _FusedAttnRNGStateChecker:
    """
    Checker for guarding the fused attention rng state.
    The fused attention backend requires a 64 bits seed and a 64 bits offset.
    However, JAX doesn't enable 64 bits by default,
    so we have to emulate seed as two 32 bits array.
    The offset calculation is maintained in the backend.
    """
    rng_state_dtype: jnp.dtype = jnp.uint32
    # (seed,) with internal dtype int64
    seed_size: int = 2
    # (seed, offset) with internal dtype int64
    rng_state_size: int = 2 * 2

    def check_seed(self, seed, dropout_probability, is_training):
        """
        Check the seed and convert the data type of seed if possible.
        """
        # Jax can't bind None, create a dummy tensor for None
        if seed is None:
            dropout_enabled = dropout_probability > 0 and is_training
            assert not dropout_enabled, "seed is not allowed to be None when dropout is enabled."
            seed = jnp.zeros(2, dtype=self.rng_state_dtype)

        if seed.dtype != self.rng_state_dtype:
            warnings.warn(
                f"Requested {seed.dtype=} is not available, and will be "
                f"casted to dtype {self.rng_state_dtype}. "
                f"Please use threefry/rbg/unsafe_rbg PRNG implementations to remove this warning.")
            seed = seed.astype(self.rng_state_dtype)

        assert seed.dtype == self.rng_state_dtype
        # Backend takes an int64_t seed, so only the first two u32 elements are taken
        assert seed.size >= self.seed_size

        return seed


class SelfFusedAttnFwdPrimitive(BasePrimitive):
    """
    Self Fused Attention Forward Primitive
    """
    name = "te_self_fused_attn_forward"
    multiple_results = True

    @staticmethod
    def abstract(
            qkv,
            bias,
            cu_seqlen,    # pylint: disable=unused-argument
            seed,    # pylint: disable=unused-argument
            *,
            attn_bias_type,    # pylint: disable=unused-argument
            attn_mask_type,    # pylint: disable=unused-argument
            scaling_factor,    # pylint: disable=unused-argument
            dropout_probability,    # pylint: disable=unused-argument
            is_training    # pylint: disable=unused-argument
    ):
        """
        Self fused attention fwd abstract
        """
        qkv_dtype = dtypes.canonicalize_dtype(qkv.dtype)
        batch, max_seqlen, nqkv, num_head, head_dim = qkv.shape
        assert nqkv == 3
        assert qkv.dtype == bias.dtype

        output_shape = (batch, max_seqlen, num_head, head_dim)
        output_dtype = qkv_dtype

        backend = transformer_engine_jax.get_fused_attn_backend(
            jax_dtype_to_te_dtype(qkv_dtype), jax_dtype_to_te_dtype(qkv_dtype),
            NVTE_QKV_Layout.NVTE_QKV_INTERLEAVED, attn_bias_type, attn_mask_type,
            dropout_probability, max_seqlen, max_seqlen, head_dim)

        if backend == NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen:
            softmax_aux_shape = (batch, num_head, max_seqlen, max_seqlen)
            softmax_dtype = qkv_dtype
        elif backend == NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
            softmax_aux_shape = (batch, num_head, max_seqlen, 1)
            softmax_dtype = dtypes.canonicalize_dtype(jnp.float32)
        else:
            raise ValueError(f'Not supported {backend=}')

        checker = _FusedAttnRNGStateChecker()
        seed_dtype = dtypes.canonicalize_dtype(seed.dtype)
        assert seed_dtype == checker.rng_state_dtype
        rng_state_shape = (checker.rng_state_size,)
        rng_state_dtype = seed_dtype

        return (
            ShapedArray(output_shape, output_dtype, named_shape=qkv.named_shape),    # output
            ShapedArray(softmax_aux_shape, softmax_dtype,
                        named_shape=qkv.named_shape),    # softmax_aux
            ShapedArray(rng_state_shape, rng_state_dtype,
                        named_shape=seed.named_shape),    # rng_state
        )

    @staticmethod
    def lowering(ctx, qkv, bias, cu_seqlen, seed, *, attn_bias_type, attn_mask_type, scaling_factor,
                 dropout_probability, is_training):
        """
        Self fused attention fwd lowering rules
        """
        qkv_aval, _, _, _ = ctx.avals_in

        batch, max_seqlen, _, num_head, head_dim = qkv_aval.shape

        operands = [qkv, bias, cu_seqlen, seed]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            batch, num_head, max_seqlen, max_seqlen, head_dim, scaling_factor, dropout_probability,
            attn_bias_type, attn_mask_type, jax_dtype_to_te_dtype(qkv_aval.dtype), is_training)

        out = custom_caller(SelfFusedAttnFwdPrimitive.name, args, opaque, has_side_effect=False)

        return out


_self_fused_attn_fwd_p = register_primitive(SelfFusedAttnFwdPrimitive)


def self_fused_attn_fwd(qkv: jnp.ndarray, bias: jnp.ndarray, cu_seqlen: jnp.ndarray,
                        seed: jnp.ndarray, attn_bias_type: NVTE_Bias_Type,
                        attn_mask_type: NVTE_Mask_Type, scaling_factor: float,
                        dropout_probability: float, is_training: bool):
    """
    Wrapper for TE self fused attention fwd
    Return BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    """
    checker = _FusedAttnRNGStateChecker()
    seed = checker.check_seed(seed, dropout_probability, is_training)

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=qkv.dtype)
    return _self_fused_attn_fwd_p.bind(qkv,
                                       bias,
                                       cu_seqlen,
                                       seed,
                                       attn_bias_type=attn_bias_type,
                                       attn_mask_type=attn_mask_type,
                                       scaling_factor=scaling_factor,
                                       dropout_probability=dropout_probability,
                                       is_training=is_training)


class SelfFusedAttnBwdPrimitive(BasePrimitive):
    """
    Self Fused Attention Backward Primitive
    """
    name = "te_self_fused_attn_backward"
    multiple_results = True

    @staticmethod
    def abstract(
            qkv,
            softmax_aux,    # pylint: disable=unused-argument
            rng_state,    # pylint: disable=unused-argument
            output,    # pylint: disable=unused-argument
            doutput,
            cu_seqlen,    # pylint: disable=unused-argument
            *,
            attn_bias_type,    # pylint: disable=unused-argument
            attn_mask_type,    # pylint: disable=unused-argument
            scaling_factor,    # pylint: disable=unused-argument
            dropout_probability,    # pylint: disable=unused-argument
            is_training    # pylint: disable=unused-argument
    ):
        """
        Self fused attention bwd abstract
        """
        qkv_dtype = dtypes.canonicalize_dtype(qkv.dtype)
        assert qkv.dtype == doutput.dtype

        _, seqlen, _, num_head, _ = qkv.shape

        if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
            bias_shape = (0,)
        else:
            bias_shape = (1, num_head, seqlen, seqlen)
        bias_dtype = qkv_dtype

        return (
            ShapedArray(qkv.shape, qkv_dtype, named_shape=qkv.named_shape),    # dqkv
            ShapedArray(bias_shape, bias_dtype, named_shape=qkv.named_shape))

    @staticmethod
    def lowering(ctx, qkv, softmax_aux, rng_state, output, doutput, cu_seqlen, *, attn_bias_type,
                 attn_mask_type, scaling_factor, dropout_probability, is_training):
        """
        Self fused attention bwd lowering rules
        """
        qkv_aval, _, _, _, _, _ = ctx.avals_in

        batch, max_seqlen, _, num_head, head_dim = qkv_aval.shape

        operands = [qkv, softmax_aux, rng_state, output, doutput, cu_seqlen]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            batch, num_head, max_seqlen, max_seqlen, head_dim, scaling_factor, dropout_probability,
            attn_bias_type, attn_mask_type, jax_dtype_to_te_dtype(qkv_aval.dtype), is_training)

        out = custom_caller(SelfFusedAttnBwdPrimitive.name, args, opaque, has_side_effect=False)

        return out


_self_fused_attn_bwd_p = register_primitive(SelfFusedAttnBwdPrimitive)


def self_fused_attn_bwd(qkv: jnp.ndarray, softmax_aux: jnp.ndarray, rng_state: jnp.ndarray,
                        output: jnp.ndarray, doutput: jnp.ndarray, cu_seqlen: jnp.ndarray,
                        attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                        scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE self fused attention bwd
    Return the gradients of self fused attention with packed qkv input
    """
    return _self_fused_attn_bwd_p.bind(qkv,
                                       softmax_aux,
                                       rng_state,
                                       output,
                                       doutput,
                                       cu_seqlen,
                                       attn_bias_type=attn_bias_type,
                                       attn_mask_type=attn_mask_type,
                                       scaling_factor=scaling_factor,
                                       dropout_probability=dropout_probability,
                                       is_training=is_training)


class CrossFusedAttnFwdPrimitive(BasePrimitive):
    """
    Cross Fused Attention Forward Primitive
    """
    name = "te_cross_fused_attn_forward"
    multiple_results = True

    @staticmethod
    def abstract(
            q,
            kv,
            q_cu_seqlen,
            kv_cu_seqlen,
            seed,    # pylint: disable=unused-argument
            *,
            attn_bias_type,    # pylint: disable=unused-argument
            attn_mask_type,    # pylint: disable=unused-argument
            scaling_factor,    # pylint: disable=unused-argument
            dropout_probability,    # pylint: disable=unused-argument
            is_training    # pylint: disable=unused-argument
    ):
        """
        Cross fused attention fwd abstract
        """
        q_dtype = dtypes.canonicalize_dtype(q.dtype)
        batch_q, q_max_seqlen, num_head_q, head_dim_q = q.shape
        kv_dtype = dtypes.canonicalize_dtype(kv.dtype)
        batch_kv, kv_max_seqlen, nkv, num_head_kv, head_dim_kv = kv.shape

        assert q_dtype == kv_dtype
        assert batch_q == batch_kv
        assert num_head_q == num_head_kv
        assert head_dim_q == head_dim_kv
        assert nkv == 2
        assert q_cu_seqlen.dtype == kv_cu_seqlen.dtype

        output_shape = q.shape
        output_dtype = q_dtype
        softmax_aux_shape = (batch_q, num_head_q, q_max_seqlen, kv_max_seqlen)
        softmax_aux_dtype = q_dtype

        return (
            ShapedArray(output_shape, output_dtype, named_shape=q.named_shape),    # output
            ShapedArray(softmax_aux_shape, softmax_aux_dtype,
                        named_shape=q.named_shape),    # softmax_aux
        )

    @staticmethod
    def lowering(ctx, q, kv, q_cu_seqlen, kv_cu_seqlen, seed, *, attn_bias_type, attn_mask_type,
                 scaling_factor, dropout_probability, is_training):
        """
        Cross fused attention fwd lowering rules
        """
        q_aval, kv_aval, _, _, _ = ctx.avals_in
        assert q_aval.dtype == kv_aval.dtype

        batch, q_max_seqlen, num_head, head_dim = q_aval.shape
        kv_max_seqlen = kv_aval.shape[1]

        operands = [q, kv, q_cu_seqlen, kv_cu_seqlen, seed]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)
        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            batch, num_head, q_max_seqlen, kv_max_seqlen, head_dim,
            scaling_factor, dropout_probability, attn_bias_type, attn_mask_type,
            jax_dtype_to_te_dtype(q_aval.dtype), is_training)

        out = custom_caller(CrossFusedAttnFwdPrimitive.name, args, opaque, has_side_effect=False)

        return out


_cross_fused_attn_fwd_p = register_primitive(CrossFusedAttnFwdPrimitive)


def cross_fused_attn_fwd(q: jnp.ndarray, kv: jnp.ndarray, q_cu_seqlen: jnp.ndarray,
                         kv_cu_seqlen: jnp.ndarray, seed: jnp.ndarray,
                         attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                         scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE cross fused attention fwd
    Return BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    """
    checker = _FusedAttnRNGStateChecker()
    seed = checker.check_seed(seed, dropout_probability, is_training)

    return _cross_fused_attn_fwd_p.bind(q,
                                        kv,
                                        q_cu_seqlen,
                                        kv_cu_seqlen,
                                        seed,
                                        attn_bias_type=attn_bias_type,
                                        attn_mask_type=attn_mask_type,
                                        scaling_factor=scaling_factor,
                                        dropout_probability=dropout_probability,
                                        is_training=is_training)


class CrossFusedAttnBwdPrimitive(BasePrimitive):
    """
    Cross Fused Attention Backward Primitive
    """
    name = "te_cross_fused_attn_backward"
    multiple_results = True

    @staticmethod
    def abstract(
            q,
            kv,
            softmax_aux,
            doutput,
            q_cu_seqlen,
            kv_cu_seqlen,
            *,
            attn_bias_type,    # pylint: disable=unused-argument
            attn_mask_type,    # pylint: disable=unused-argument
            scaling_factor,    # pylint: disable=unused-argument
            dropout_probability,    # pylint: disable=unused-argument
            is_training    # pylint: disable=unused-argument
    ):
        """
        Cross fused attention bwd abstract
        """
        q_dtype = dtypes.canonicalize_dtype(q.dtype)
        kv_dtype = dtypes.canonicalize_dtype(kv.dtype)
        softmax_aux_dtype = dtypes.canonicalize_dtype(softmax_aux.dtype)
        doutput_dtype = dtypes.canonicalize_dtype(doutput.dtype)
        assert q_dtype == kv_dtype == softmax_aux_dtype == doutput_dtype
        assert q_cu_seqlen.dtype == kv_cu_seqlen.dtype

        return (
            ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),    # dq
            ShapedArray(kv.shape, kv_dtype, named_shape=kv.named_shape),    # dkv
        )

    @staticmethod
    def lowering(ctx, q, kv, softmax_aux, doutput, q_cu_seqlen, kv_cu_seqlen, *, attn_bias_type,
                 attn_mask_type, scaling_factor, dropout_probability, is_training):
        """
        Cross fused attention bwd lowering rules
        """
        q_aval, kv_aval, _, _, _, _ = ctx.avals_in
        assert q_aval.dtype == kv_aval.dtype

        batch, q_max_seqlen, num_head, head_dim = q_aval.shape
        kv_max_seqlen = kv_aval.shape[1]

        operands = [q, kv, softmax_aux, doutput, q_cu_seqlen, kv_cu_seqlen]
        operand_shapes = map(lambda x: x.type.shape, operands)
        out_types = [
            ir.RankedTensorType.get(output.shape, mlir.dtype_to_ir_type(output.dtype))
            for output in ctx.avals_out
        ]

        args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

        # the dropout elements are encoded in the forward auxiliary tensor
        # so seed is not needed in backward
        opaque = transformer_engine_jax.pack_fused_attn_descriptor(
            batch, num_head, q_max_seqlen, kv_max_seqlen, head_dim,
            scaling_factor, dropout_probability, attn_bias_type, attn_mask_type,
            jax_dtype_to_te_dtype(q_aval.dtype), is_training)

        out = custom_caller(CrossFusedAttnBwdPrimitive.name, args, opaque, has_side_effect=False)

        return out


_cross_fused_attn_bwd_p = register_primitive(CrossFusedAttnBwdPrimitive)


def cross_fused_attn_bwd(q: jnp.ndarray, kv: jnp.ndarray, softmax_aux: jnp.ndarray,
                         doutput: jnp.ndarray, q_cu_seqlen: jnp.ndarray, kv_cu_seqlen: jnp.ndarray,
                         attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                         scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE cross fused attention bwd
    Return the gradients of cross fused attention with packed kv input
    """
    return _cross_fused_attn_bwd_p.bind(q,
                                        kv,
                                        softmax_aux,
                                        doutput,
                                        q_cu_seqlen,
                                        kv_cu_seqlen,
                                        attn_bias_type=attn_bias_type,
                                        attn_mask_type=attn_mask_type,
                                        scaling_factor=scaling_factor,
                                        dropout_probability=dropout_probability,
                                        is_training=is_training)
