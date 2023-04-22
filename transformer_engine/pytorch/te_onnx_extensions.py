# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import transformer_engine_extensions as tex


# This file registers custom op symbolic ONNX functions and does not export any symbols.
__all__ = []


# Custom ops spec version
VER = 1

UNSPECIFIED_TYPE = -1


def make_op_name(op_name: str) -> str:
    """custom op name"""
    return "trt::" + op_name


def quantize(g, inputs, scale_inv, fp8_tensor):
    """Helper Function for Quantization"""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)

    # Q inputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the input if needed.
    if inputs.type().scalarType() == "Half" or inputs.type().scalarType() == "BFloat16":
        inputs = g.op("Cast", inputs, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    scale = g.op("Constant", value_t=torch.tensor(scale_inv[fp8_tensor]))
    q_op = g.op(
        make_op_name("TRT_FP8QuantizeLinear"), inputs, scale).setType(
            inputs.type().with_dtype(torch.uint8).with_sizes(output_shape))
    return q_op


def dequantize(g, inputs, scale_inv, fp8_tensor, otype):
    """Helper Function for Dequantization"""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)

    scale = g.op("Constant", value_t=torch.tensor(scale_inv[fp8_tensor]))
    out = g.op(make_op_name("TRT_FP8DequantizeLinear"), inputs, scale).setType(
        inputs.type().with_dtype(torch.float32).with_sizes(output_shape))

    # DQ outputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the output if needed.
    if otype == int(tex.DType.kFloat16):
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
    elif otype == int(tex.DType.kBFloat16):
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)
    return out


def compute_in_fp32(g, inp, subgraph, cast_outp):
    """Wrap subgraph with casts to/from FP32 so that its precision is FP32.

    If `inp` data type is not FP32, add a cast of `inp` to FP32 and feed that into `subgraph`.
    Then, if `cast_output` is true, cast subgraphs's output back to `inp` data type.
    """
    try:
        inp_dtype = {
            "Float": _C_onnx.TensorProtoDataType.FLOAT,
            "Half": _C_onnx.TensorProtoDataType.FLOAT16,
            "BFloat16": _C_onnx.TensorProtoDataType.BFLOAT16,
        }[inp.type().scalarType()]
    except KeyError as e:
        raise TypeError(f"Onnx export for dtype {inp.type().scalarType()} not supported.") from e

    is_fp32 = inp_dtype == _type_utils.JitScalarType.FLOAT
    if not is_fp32:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    sg_out = subgraph(inp)
    if not is_fp32 and cast_outp:
        sg_out = g.op("Cast", sg_out, to_i=inp_dtype)
    return sg_out


@symbolic_helper.parse_args("v", "v", "v", "fs", "i", "i")
def onnx_cast_to_fp8(g, inputs, scale, amax, scale_inv, fp8_tensor, otype):
    """ONNX graph for cast_to_fp8"""
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
    wrapped_gelu = lambda inputs: torch.onnx.symbolic_opset9.gelu(g, inputs, "tanh")
    # TE computes GELU using float32 precision so wrap the GELU subgraph with
    # conversion to/from float32.
    gelu = compute_in_fp32(g, inputs, wrapped_gelu, cast_outp=False)
    out = quantize(g, gelu, scale_inv, fp8_tensor)
    return out


@symbolic_helper.parse_args("v", "fs", "i", "i", "i",
                            "v", "fs", "i", "i", "i",
                            "v", "fs", "i", "fs", "v", "i", "v", "i",
                            "v", "i", "i", "i")
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
    use_split_accumulator):
    """ONNX graph for te_gemm"""
    # pylint: disable=unused-argument
    is_fp16 = bias.type().scalarType() == "Half"
    if input_type == int(tex.DType.kFloat8E4M3):
        inputs = dequantize(g, inputs, input_scale_inverse, input_fp8_tensor, UNSPECIFIED_TYPE)

    if weight_type == int(tex.DType.kFloat8E4M3):
        weight = dequantize(g, weight, weight_scale_inverse, weight_fp8_tensor, UNSPECIFIED_TYPE)

    output = g.op("Gemm", inputs, weight, transA_i=trans_input, transB_i=trans_weight)

    empty_tensor_size = [0]
    bias_empty = torch.onnx.symbolic_helper._get_tensor_sizes(bias) == empty_tensor_size
    pre_gelu_out_empty = torch.onnx.symbolic_helper._get_tensor_sizes(pre_gelu_out) \
        == empty_tensor_size
    if not bias_empty:
        if pre_gelu_out_empty:
            if is_fp16:
                output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
            output = g.op('Add', output, bias)
        else:
            if is_fp16:
                output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
            output = g.op('Add', output, bias)
            output = torch.onnx.symbolic_opset9.gelu(g, output)
    else:
        if is_fp16:
            output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
    return output


@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v", "fs", "i", "i", "b")
def onnx_layernorm_fwd_fp8(g, inputs, weight, bias, eps, scale, amax,
                           scale_inv, fp8_tensor, otype, zero_centered_gamma):
    """ONNX graph for layernorm_fwd_fp8"""
    # pylint: disable=unused-argument
    ln = onnx_layernorm_fwd(g, inputs, weight, bias, eps, zero_centered_gamma)
    fp8_ln = quantize(g, ln, scale_inv, fp8_tensor)
    return fp8_ln


@symbolic_helper.parse_args("v", "v", "v", "f", "b")
def onnx_layernorm_fwd(g, inputs, weight, bias, eps, zero_centered_gamma):
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
        inputs_dtype= inputs.type().dtype()
        one = g.op("Constant", value_t=torch.tensor([1.], dtype=inputs_dtype, device="cuda"))
        weight = g.op("Add", weight, one)

    axis = -len(normalized_shape)
    ln =  g.op(
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


register_custom_op_symbolic('tex_ts::cast_to_fp8_ts', onnx_cast_to_fp8, VER)
register_custom_op_symbolic('tex_ts::cast_from_fp8_ts', onnx_cast_from_fp8, VER)
register_custom_op_symbolic('tex_ts::fp8_gelu_ts', onnx_fp8_gelu, VER)
register_custom_op_symbolic('tex_ts::te_gemm_ts', onnx_te_gemm, VER)
register_custom_op_symbolic('tex_ts::layernorm_fwd_fp8_inf_ts', onnx_layernorm_fwd_fp8, VER)
register_custom_op_symbolic('tex_ts::layernorm_fwd_inf_ts', onnx_layernorm_fwd, VER)
