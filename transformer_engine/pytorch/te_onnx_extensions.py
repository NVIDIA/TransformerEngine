# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
ONNX symbolic functions for Transformer Engine

Warnings of the type pasted below are a known Pytorch issue
(https://github.com/pytorch/pytorch/issues/81693):

tests/test_onnx_export.py::test_export_cast_ops[112]
  /opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py:649:
  UserWarning: The shape inference of trt::TRT_FP8DequantizeLinear type is missing,
  so it may result in wrong shape inference for the exported graph.
  Please consider adding it in symbolic function. (Triggered internally at
  /opt/pytorch/pytorch/torch/csrc/jit/passes/onnx/shape_type_inference.cpp:1880.)
    _C._jit_pass_onnx_graph_shape_type_inference(


Scale tensors are treated as lists ("fs") instead of tensors ("v") because we need to access
specific entries using the index passes as `fp8_tensor`. If you fail to do this you will get
the following error when accessing a sepcific scale element (e.g. `scale_inv[fp8_tensor]`):
    TypeError: 'torch._C.Value' object is not subscriptable
"""

import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic, _type_utils
import torch._C._onnx as _C_onnx

# Monkey-patch graph manipulation methods on Graph, used for the ONNX symbolics
from torch.onnx._internal import jit_utils

import transformer_engine_torch as tex


# This file registers custom op symbolic ONNX functions and does not export any symbols.
__all__ = []


# Custom ops spec version
VER = 1
UNSPECIFIED_TYPE = -1


def make_op_name(op_name: str) -> str:
    """custom op name"""
    return "trt::" + op_name


def get_TensorProtoDataType(t):
    """Return the _C_onnx.TensorProtoDataType of the input tensor"""
    try:
        return {
            "Float": _C_onnx.TensorProtoDataType.FLOAT,
            "Half": _C_onnx.TensorProtoDataType.FLOAT16,
            "BFloat16": _C_onnx.TensorProtoDataType.BFLOAT16,
        }[t.type().scalarType()]
    except KeyError as e:
        raise TypeError(f"Onnx export for dtype {t.type().scalarType()} not supported.") from e


def is_dtype_fp32(t):
    """Check fp32 dtype"""
    return t.type().scalarType() == "Float"


def is_dtype_fp16(t):
    """Check fp16 dtype"""
    return t.type().scalarType() == "Half"


def is_dtype_bf16(t):
    """Check bf16 dtype"""
    return t.type().scalarType() == "BFloat16"


def quantize(g, inputs, scale_inv, fp8_tensor):
    """Helper Function for Quantization"""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)

    # Q inputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the input if needed.
    if not is_dtype_fp32(inputs):
        inputs = g.op("Cast", inputs, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    scale = g.op("Constant", value_t=torch.tensor(scale_inv[fp8_tensor]))
    q_op = g.op(make_op_name("TRT_FP8QuantizeLinear"), inputs, scale).setType(
        inputs.type().with_dtype(torch.uint8).with_sizes(output_shape)
    )
    return q_op


def dequantize(g, inputs, scale_inv, fp8_tensor, otype):
    """Helper Function for Dequantization"""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)

    scale = g.op("Constant", value_t=torch.tensor(scale_inv[fp8_tensor]))
    out = g.op(make_op_name("TRT_FP8DequantizeLinear"), inputs, scale).setType(
        inputs.type().with_dtype(torch.float32).with_sizes(output_shape)
    )

    # DQ outputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the output if needed.
    if otype == int(tex.DType.kFloat16):
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
    elif otype == int(tex.DType.kBFloat16):
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)
    return out


def compute_in_fp32(g, inp, subgraph, *args, **kwargs):
    """Wrap subgraph with casts to/from FP32 so that its precision is FP32.

    If `inp` data type is not FP32, add a cast of `inp` to FP32 and feed that into `subgraph`;
    then cast subgraphs's output back to `inp` data type.
    """
    inp_dtype = get_TensorProtoDataType(inp)
    is_fp32 = inp_dtype == _type_utils.JitScalarType.FLOAT
    if not is_fp32:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    sg_out = subgraph(g, inp, *args, **kwargs)
    if not is_fp32:
        sg_out = g.op("Cast", sg_out, to_i=inp_dtype)
    return sg_out


@symbolic_helper.parse_args("v", "v", "v", "fs", "i", "i")
def onnx_cast_to_fp8(g, inputs, scale, amax, scale_inv, fp8_tensor, otype):
    """ONNX graph for cast_to_fp8"""
    # pylint: disable=unused-argument
    return quantize(g, inputs, scale_inv, fp8_tensor)


@symbolic_helper.parse_args("v", "v", "v", "v", "fs", "i", "i")
def onnx_cast_to_fp8_noalloc(g, inputs, scale, output, amax, scale_inv, fp8_tensor, otype):
    """ONNX graph for cast_to_fp8_noalloc"""
    # pylint: disable=unused-argument
    return quantize(g, inputs, scale_inv, fp8_tensor)


@symbolic_helper.parse_args("v", "fs", "i", "i", "i")
def onnx_cast_from_fp8(g, inputs, scale_inv, fp8_tensor, itype, otype):
    """ONNX graph for cast_from_fp8"""
    # pylint: disable=unused-argument
    return dequantize(g, inputs, scale_inv, fp8_tensor, otype)


@symbolic_helper.parse_args("v", "v", "v", "fs", "i", "i")
def onnx_fp8_gelu(g, inputs, scale, amax, scale_inv, fp8_tensor, otype):
    """ONNX graph for fp8_gelu"""
    # pylint: disable=unused-argument
    # TE computes GELU using float32 precision so wrap the GELU subgraph with
    # conversion to/from float32.
    gelu = compute_in_fp32(g, inputs, torch.onnx.symbolic_opset9.gelu, "tanh")
    if scale_inv:
        gelu = quantize(g, gelu, scale_inv, fp8_tensor)
    return gelu


@symbolic_helper.parse_args("v", "v", "v", "fs", "i", "i")
def onnx_fp8_relu(g, inputs, scale, amax, scale_inv, fp8_tensor, otype):
    """ONNX graph for fp8_relu"""
    # pylint: disable=unused-argument
    relu = compute_in_fp32(g, inputs, torch.onnx.symbolic_opset9.relu)
    if scale_inv:
        relu = quantize(g, relu, scale_inv, fp8_tensor)
    return relu


@symbolic_helper.parse_args("v", "i")
def onnx_swiglu(g: jit_utils.GraphContext, inp, dim):
    """ONNX graph for swiglu"""
    dim_size = symbolic_helper._get_tensor_dim_size(inp, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op("Split", inp, axis_i=dim, outputs=2)
    return g.op("Mul", g.op("Sigmoid", first), second)


@symbolic_helper.parse_args("v", "v", "v", "fs", "i", "i")
def onnx_fp8_swiglu(g, inputs, scale, amax, scale_inv, fp8_tensor, otype):
    """ONNX graph for fp8_swiglu"""
    # pylint: disable=unused-argument
    swiglu = compute_in_fp32(g, inputs, onnx_swiglu, 1)
    if scale_inv:
        swiglu = quantize(g, swiglu, scale_inv, fp8_tensor)
    return swiglu


@symbolic_helper.parse_args("v", "i")
def onnx_reglu(g: jit_utils.GraphContext, inp, dim):
    """ONNX graph for reglu"""
    dim_size = symbolic_helper._get_tensor_dim_size(inp, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op("Split", inp, axis_i=dim, outputs=2)
    return g.op("Mul", g.op("Relu", first), second)


@symbolic_helper.parse_args("v", "v", "v", "fs", "i", "i")
def onnx_fp8_reglu(g, inputs, scale, amax, scale_inv, fp8_tensor, otype):
    """ONNX graph for fp8_reglu"""
    # pylint: disable=unused-argument
    reglu = compute_in_fp32(g, inputs, onnx_reglu, 1)
    if scale_inv:
        reglu = quantize(g, reglu, scale_inv, fp8_tensor)
    return reglu


@symbolic_helper.parse_args("v", "i")
def onnx_geglu(g: jit_utils.GraphContext, inp, dim):
    """ONNX graph for geglu"""
    dim_size = symbolic_helper._get_tensor_dim_size(inp, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op("Split", inp, axis_i=dim, outputs=2)
    first_gelu = torch.onnx.symbolic_opset9.gelu(g, first, "tanh")
    return g.op("Mul", first_gelu, second)


@symbolic_helper.parse_args("v", "v", "v", "fs", "i", "i")
def onnx_fp8_geglu(g, inputs, scale, amax, scale_inv, fp8_tensor, otype):
    """ONNX graph for fp8_geglu"""
    # pylint: disable=unused-argument
    geglu = compute_in_fp32(g, inputs, onnx_geglu, 1)
    if scale_inv:
        geglu = quantize(g, geglu, scale_inv, fp8_tensor)
    return geglu


@symbolic_helper.parse_args(
    "v",
    "fs",
    "i",
    "i",
    "i",
    "v",
    "fs",
    "i",
    "i",
    "i",
    "v",
    "fs",
    "i",
    "fs",
    "v",
    "i",
    "v",
    "i",
    "v",
    "i",
    "i",
    "i",
)
def onnx_te_gemm(
    g,
    weight,
    weight_scale_inverse,
    weight_fp8_tensor,
    weight_type,
    trans_weight,
    inputs,
    input_scale_inverse,
    input_fp8_tensor,
    input_type,
    trans_input,
    out,
    out_scale,
    out_type,
    out_amax,
    bias,
    bias_type,
    pre_gelu_out,
    grad,
    workspace,
    workspaceSize,
    accumulate,
    use_split_accumulator,
):
    """ONNX graph for te_gemm"""
    # pylint: disable=unused-argument
    is_fp16 = is_dtype_fp16(inputs)
    is_bf16 = is_dtype_bf16(inputs)
    if input_type == int(tex.DType.kFloat8E4M3):
        inputs = dequantize(g, inputs, input_scale_inverse, input_fp8_tensor, out_type)

    if weight_type == int(tex.DType.kFloat8E4M3):
        weight = dequantize(g, weight, weight_scale_inverse, weight_fp8_tensor, out_type)

    empty_tensor_size = [0]
    bias_empty = torch.onnx.symbolic_helper._get_tensor_sizes(bias) == empty_tensor_size
    pre_gelu_out_empty = (
        torch.onnx.symbolic_helper._get_tensor_sizes(pre_gelu_out) == empty_tensor_size
    )

    if not bias_empty:
        output = g.op("Gemm", inputs, weight, bias, transA_i=trans_input, transB_i=trans_weight)
    else:
        output = g.op("Gemm", inputs, weight, transA_i=trans_input, transB_i=trans_weight)
    if not bias_empty:
        if not pre_gelu_out_empty:
            # TE computes GELU using float32 precision so wrap the GELU subgraph with
            # conversion to/from float32.
            output = compute_in_fp32(g, output, torch.onnx.symbolic_opset9.gelu, "tanh")
    else:
        if is_fp16:
            output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
        elif is_bf16:
            output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)
    return output


def _ones_like(g, inp, dtype):
    """Returns a tensor filled with the scalar value 1, with the same size as input and
    with dtype data-type"""
    shape = g.op("Shape", inp)
    # WAR ONNX spec: ConstantOfShape accepts all data types except for BF16. To WAR
    # create a ConstantOfShape with type FP32 and then add a Cast to BF16.
    is_bf16 = dtype == torch.bfloat16
    one = g.op(
        "ConstantOfShape",
        shape,
        value_t=torch.tensor([1], dtype=torch.float32 if is_bf16 else dtype),
    )
    if is_bf16:
        one = g.op("Cast", one, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)
    return one


@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v", "fs", "i", "i", "i", "b")
def onnx_layernorm_fwd_fp8(
    g,
    inputs,
    weight,
    bias,
    eps,
    scale,
    amax,
    scale_inv,
    fp8_tensor,
    otype,
    sm_margin,
    zero_centered_gamma,
):
    """ONNX graph for layernorm_fwd_fp8"""
    # pylint: disable=unused-argument
    inp_dtype = get_TensorProtoDataType(inputs)

    if inp_dtype != get_TensorProtoDataType(weight):
        weight = g.op("Cast", weight, to_i=inp_dtype)
    if inp_dtype != get_TensorProtoDataType(bias):
        bias = g.op("Cast", bias, to_i=inp_dtype)

    ln = onnx_layernorm_fwd(g, inputs, weight, bias, eps, sm_margin, zero_centered_gamma)
    fp8_ln = quantize(g, ln, scale_inv, fp8_tensor)
    return fp8_ln


@symbolic_helper.parse_args("v", "v", "v", "f", "i", "b")
def onnx_layernorm_fwd(g, inputs, weight, bias, eps, sm_margin, zero_centered_gamma):
    """ONNX graph for layernorm_fwd"""
    # pylint: disable=unused-argument

    normalized_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)
    if normalized_shape is None:
        ndim = torch.onnx.symbolic_helper._get_tensor_rank(inputs)
        assert ndim is not None
        normalized_shape = list(range(0, ndim))
    # Normalization axis = 0, so normalized_shape uses all dims except dim = 0
    normalized_shape = normalized_shape[1:]

    if zero_centered_gamma:
        inputs_dtype = inputs.type().dtype()
        one = _ones_like(g, weight, inputs_dtype)
        weight = g.op("Add", weight, one)

    axis = -len(normalized_shape)
    ln = g.op(
        "LayerNormalization",
        inputs,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=axis,
        # This sets the LN compute precision - use FP32 always as does TE.
        stash_type_i=_C_onnx.TensorProtoDataType.FLOAT,
    )
    return ln


@symbolic_helper.parse_args("v", "v", "f", "v", "v", "fs", "i", "i", "i", "b")
def onnx_rmsnorm_fwd_fp8(
    g,
    inputs,
    weight,
    eps,
    scale,
    amax,
    scale_inv,
    fp8_tensor,
    otype,
    sm_margin,
    zero_centered_gamma,
):
    """ONNX graph for rmsnorm_fwd_fp8"""
    # pylint: disable=unused-argument
    inp_dtype = get_TensorProtoDataType(inputs)

    if inp_dtype != get_TensorProtoDataType(weight):
        weight = g.op("Cast", weight, to_i=inp_dtype)

    ln = onnx_rmsnorm_fwd(g, inputs, weight, eps, sm_margin, zero_centered_gamma)
    fp8_ln = quantize(g, ln, scale_inv, fp8_tensor)
    return fp8_ln


@symbolic_helper.parse_args("v", "v", "f", "i", "b")
def onnx_rmsnorm_fwd(g, inputs, weight, eps, sm_margin, zero_centered_gamma):
    """ONNX graph for rmsnorm_fwd"""
    # pylint: disable=unused-argument

    normalized_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)
    if normalized_shape is None:
        ndim = torch.onnx.symbolic_helper._get_tensor_rank(inputs)
        assert ndim is not None
        normalized_shape = list(range(0, ndim))
    # Normalization axis = 0, so normalized_shape uses all dims except dim = 0
    normalized_shape = normalized_shape[1:]

    if zero_centered_gamma:
        inputs_dtype = inputs.type().dtype()
        one = _ones_like(g, weight, inputs_dtype)
        weight = g.op("Add", weight, one)

    axis = -len(normalized_shape)

    inputs_float = g.op("Cast", inputs, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    sum_square = g.op("ReduceSumSquare", inputs_float, axes_i=[axis])
    shape = g.op("Shape", inputs_float, start_i=-1)
    shape_f = g.op("Cast", shape, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    mean_squared = g.op("Div", sum_square, shape_f)
    eps_tensor = g.op("ConstantOfShape", shape, value_t=torch.tensor([eps], dtype=torch.float32))
    rms_squared = g.op("Add", mean_squared, eps_tensor)
    rms_eps = g.op("Sqrt", rms_squared)
    normalized_input = g.op("Div", inputs_float, rms_eps)
    result = g.op("Mul", weight, normalized_input)
    result = g.op("Cast", result, to_i=get_TensorProtoDataType(inputs))

    return result


register_custom_op_symbolic("tex_ts::cast_to_fp8_ts", onnx_cast_to_fp8, VER)
register_custom_op_symbolic("tex_ts::cast_to_fp8_noalloc_ts", onnx_cast_to_fp8_noalloc, VER)
register_custom_op_symbolic("tex_ts::cast_from_fp8_ts", onnx_cast_from_fp8, VER)
register_custom_op_symbolic("tex_ts::gelu_ts", onnx_fp8_gelu, VER)
register_custom_op_symbolic("tex_ts::relu_ts", onnx_fp8_relu, VER)
register_custom_op_symbolic("tex_ts::reglu_ts", onnx_fp8_reglu, VER)
register_custom_op_symbolic("tex_ts::geglu_ts", onnx_fp8_geglu, VER)
register_custom_op_symbolic("tex_ts::swiglu_ts", onnx_fp8_swiglu, VER)
register_custom_op_symbolic("tex_ts::te_gemm_ts", onnx_te_gemm, VER)
register_custom_op_symbolic("tex_ts::layernorm_fwd_fp8_inf_ts", onnx_layernorm_fwd_fp8, VER)
register_custom_op_symbolic("tex_ts::layernorm_fwd_inf_ts", onnx_layernorm_fwd, VER)
register_custom_op_symbolic("tex_ts::rmsnorm_fwd_fp8_inf_ts", onnx_rmsnorm_fwd_fp8, VER)
register_custom_op_symbolic("tex_ts::rmsnorm_fwd_inf_ts", onnx_rmsnorm_fwd, VER)
