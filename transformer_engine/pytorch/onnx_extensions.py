# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""

File containing torch.ops extensions and their corresponding ONNX symbolic functions.

Many transformer engine layers rely on custom calls from the transformer_engine_torch module, making ONNX export challenging because:
1. They often accept Python objects (quantizers), which ONNX does not support.
2. They are complex, incorporating fusions and precomputing certain values for backward passes—mechanisms unnecessary for ONNX export.

For these reasons, we introduce onnx_forward methods in each layer that are simpler and 
primarily leverage torch operators with known ONNX symbolic functions. 
These methods avoid fusions and backward pass precomputations. 
The main considerations are quantization—which PyTorch does not natively support, so we need to implement onnx symbolic functions on our own.
Also we define gemm to use ONNX operator GEMM, not MatMul.

Since ONNX does not yet support quantization, operators from TensorRT are employed. 
The primary goal of ONNX export is to enable inference compatibility with TensorRT.

"""

import torch
import onnxscript
from onnxscript import opset18 as op
import transformer_engine_torch as tex
from .tensor.float8_tensor import Float8Quantizer
from .tensor.mxfp8_tensor import MXFP8Quantizer
from .constants import TE_DType
from typing import Optional, Tuple
from .export import is_in_onnx_export_mode

# ONNX GEMM for inference


def onnx_gemm(weight: torch.Tensor, inp: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """ ONNX GEMM used for inference. """
    reshaped_inp = inp.reshape(-1, inp.shape[-1])
    out = torch_onnx_gemm_inf_op(weight, reshaped_inp, bias)
    return out.reshape(inp.shape[:-1] + (-1,))

@torch.library.custom_op("tex::gemm_inf", mutates_args=[])
def torch_onnx_gemm_inf_op(weight: torch.Tensor, inp: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """ Gemm used for inference -- weight is transposed"""
    out = inp @ weight.T
    if bias is not None:
        out = out + bias
    return out

@torch_onnx_gemm_inf_op.register_fake
def _(weight, inp, bias):
    """ Fake gemm used for inference. """    
    out = inp @ weight.T
    if bias is not None:
        out = out + bias
    return out

trt_opset = onnxscript.values.Opset("trt", version=1)

def onnx_gemm_inf_symbolic(weight: onnxscript.onnx_types.TensorType, 
                          inp: onnxscript.onnx_types.TensorType, 
                          bias: onnxscript.onnx_types.TensorType) -> onnxscript.onnx_types.TensorType:
    """ Symbolic gemm used for inference. """
    return op.Gemm(inp, weight, bias, transA=0, transB=1)

# ONNX FP8 Quantization

@torch.library.custom_op("tex::fp8_quantize", mutates_args=[])
def onnx_quantize_fp8_op(
    tensor: torch.Tensor,
    scale: float) -> torch.Tensor:
    """ Quantize to Float8Tensor used for inference. """
    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=tensor.device)
    amax_tensor = torch.tensor([1], dtype=torch.float32, device=tensor.device)
    quantizer = Float8Quantizer(scale_tensor, amax_tensor,  tex.DType.kFloat8E4M3)
    return quantizer.quantize(tensor)._data

@onnx_quantize_fp8_op.register_fake
def _(tensor, *_):
    """ Fake quantize to Float8Tensor used for inference. """
    return torch.empty(tensor.shape, dtype=torch.uint8, device=tensor.device)

def onnx_quantize_fp8_symbolic(
    tensor: onnxscript.onnx_types.TensorType, 
    scale: float, 
) -> onnxscript.onnx_types.UINT8:
    """ Symbolic quantize used for inference. """
    scale_inv = op.Constant(value_float=1/scale)
    return TRT_FP8QuantizeLinear(tensor, scale_inv)

# This is a dummy function that is used to create a TRT FP8 Quantize Linear node.
@onnxscript.script(trt_opset, default_opset=op)
def TRT_FP8QuantizeLinear(
    tensor: onnxscript.onnx_types.TensorType, scale: onnxscript.onnx_types.TensorType) -> onnxscript.onnx_types.TensorType:
    """ TRT FP8 Quantize Linear used for inference. """
    return tensor

# ONNX FP8 Dequantization

@torch.library.custom_op("tex::fp8_dequantize", mutates_args=[])
def onnx_dequantize_fp8_op(tensor: torch.Tensor, 
                       scale: float,
                       fake_dtype_int: int) -> torch.Tensor:
    """ Dequantize from Float8Tensor used for inference. """
    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=tensor.device)
    quantizer = Float8Quantizer(scale_tensor, torch.zeros(1).to(tensor.device), tex.DType.kFloat8E4M3)
    torch_dtype = _get_torch_dtype(fake_dtype_int)
    quantizer_tensor = quantizer.create_tensor_from_data(tensor, fake_dtype=torch_dtype)
    return quantizer_tensor.dequantize()

@onnx_dequantize_fp8_op.register_fake
def _(tensor: torch.Tensor, 
      scale: float, 
      fake_dtype_int: int) -> torch.Tensor:
    """ Fake dequantize from Float8Tensor used for inference. """
    torch_dtype = _get_torch_dtype(fake_dtype_int)
    return torch.empty(tensor.shape, dtype=torch.float32, device=tensor.device)

def onnx_dequantize_fp8_symbolic(
    tensor: onnxscript.onnx_types.TensorType,
    scale: float,
    fake_dtype: int) -> onnxscript.onnx_types.TensorType:
    """ Symbolic dequantize from Float8Tensor used for inference. """
    scale_inv = op.Constant(value_float=1/scale)
    return TRT_FP8DequantizeLinear(tensor, scale_inv)

# This is a dummy function that is used to create a TRT FP8 Dequantize Linear node.
@onnxscript.script(trt_opset, default_opset=op)
def TRT_FP8DequantizeLinear(
    tensor: onnxscript.onnx_types.TensorType, 
    scale: onnxscript.onnx_types.TensorType
) -> onnxscript.onnx_types.TensorType:
    """ TRT FP8 Dequantize Linear from Float8Tensor used for inference. """
    return tensor

# ONNX MXFP8 Quantization

@torch.library.custom_op("tex::mxfp8_quantize", mutates_args=[])
def onnx_quantize_mxfp8_op(tensor: torch.Tensor) -> torch.Tensor:
    """ Quantize to MXFP8Tensor used for inference. """
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    quantized_tensor = quantizer(tensor)
    return quantized_tensor._data, quantized_tensor._scale_inv

@onnx_quantize_mxfp8_op.register_fake
def _(tensor: torch.Tensor):
    """ Fake quantize to MXFP8Tensor used for inference. """
    mxfp8_scale_shape = tensor.shape
    return torch.empty(
        tensor.shape, 
        dtype=torch.uint8, 
        device=tensor.device
    ), torch.empty(
        mxfp8_scale_shape, 
        dtype=torch.uint8, 
        device=tensor.device
    )

def onnx_quantize_mxfp8_symbolic(
    tensor: onnxscript.onnx_types.TensorType,
    scale: float,
) -> onnxscript.onnx_types.TensorType:
    """ Symbolic quantize to MXFP8Tensor used for inference. """
    scale_inv = op.Constant(value_float=1/scale)
    return TRT_MXFP8QuantizeLinear(tensor, scale_inv)

# This is a dummy function that is used to create a TRT MXFP8 Quantize Linear node.
@onnxscript.script(trt_opset, default_opset=op)
def TRT_MXFP8QuantizeLinear(
    tensor: onnxscript.onnx_types.TensorType
) -> Tuple[onnxscript.onnx_types.TensorType, onnxscript.onnx_types.TensorType]:
    """ TRT MXFP8 Quantize Linear used for inference. """
    return tensor, tensor

# ONNX MXFP8 Dequantization

@torch.library.custom_op("tex::mxfp8_dequantize", mutates_args=[])
def onnx_dequantize_mxfp8_op(tensor: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """ Dequantize from MXFP8Tensor used for inference. """
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    quantizer_tensor = quantizer.create_tensor_from_data(tensor, fake_dtype=torch.float32)
    return quantizer_tensor.dequantize()

@onnx_dequantize_mxfp8_op.register_fake
def _(tensor: torch.Tensor, scale_inv: torch.Tensor):
    """ Fake dequantize from MXFP8Tensor used for inference. """
    return torch.empty(tensor.shape, dtype=torch.float32, device=tensor.device)

def onnx_dequantize_mxfp8_symbolic(
    tensor: onnxscript.onnx_types.TensorType,
    scale_inv: onnxscript.onnx_types.TensorType
) -> onnxscript.onnx_types.TensorType:
    """ Symbolic dequantize from MXFP8Tensor used for inference. """
    return TRT_MXFP8DequantizeLinear(tensor, scale_inv)

# This is a dummy function that is used to create a TRT MXFP8 Dequantize Linear node.
@onnxscript.script(trt_opset, default_opset=op)
def TRT_MXFP8DequantizeLinear(
    tensor: onnxscript.onnx_types.TensorType,
    scale_inv: onnxscript.onnx_types.TensorType
) -> onnxscript.onnx_types.TensorType:
    """ TRT MXFP8 Dequantize Linear from MXFP8Tensor used for inference. """
    return tensor

# ONNX LayerNorm

@torch.library.custom_op("tex::layernorm", mutates_args=[])
def onnx_layernorm_op(inp: torch.Tensor, 
                      weight: torch.Tensor, 
                      bias: torch.Tensor, 
                      eps: float) -> torch.Tensor:
    """ ONNX LayerNorm used for inference. """
    model = tex.LayerNorm(inp.shape[1], eps=eps)
    model.weight.data = weight
    model.bias.data = bias
    return model(inp)

@onnx_layernorm_op.register_fake
def _(inp, weight, bias, eps):
    """ Fake ONNX LayerNorm used for inference. """
    return inp

def onnx_layernorm_symbolic(inp: onnxscript.onnx_types.TensorType, 
                            weight: onnxscript.onnx_types.TensorType, 
                            bias: onnxscript.onnx_types.TensorType, 
                            eps: float) -> onnxscript.onnx_types.TensorType:
    """ Symbolic ONNX LayerNorm used for inference. """
    return op.LayerNormalization(inp, weight, bias, epsilon=eps)


# This translation table should be passed to torch.onnx.export function
# using the custom_translation_table=te_translation_table option.
te_translation_table = {
    torch.ops.tex.gemm_inf.default: onnx_gemm_inf_symbolic,
    torch.ops.tex.fp8_quantize.default: onnx_quantize_fp8_symbolic,
    torch.ops.tex.fp8_dequantize.default: onnx_dequantize_fp8_symbolic,
    torch.ops.tex.layernorm.default: onnx_layernorm_symbolic
}


# utility functions

def onnx_attention_mask_func(
    attention_scores: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Get attention mask"""
    if is_in_onnx_export_mode():
        return attention_scores.masked_fill(attention_mask, -10000.0)
    else:
        attention_scores.masked_fill_(attention_mask, -10000.0)
        return attention_scores

def _get_te_dtype(id: int):
    """Get the tex.DType enum value from the given id - reverse for int(tex.DType.*)."""
    all_te_dtypes = list(tex.DType.__members__.values())
    for te_dtype in all_te_dtypes:
        if int(te_dtype) == id:
            return te_dtype
    raise ValueError(f"Unknown transformer engine dtype id: {id}")

def _get_torch_dtype(dtype_id: int):
    """ Get the torch.dtype value from the given tex.DType enum value. """
    te_dtype = _get_te_dtype(dtype_id)
    all_torch_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    for torch_dtype in all_torch_dtypes:
        if TE_DType[torch_dtype] == te_dtype:
            return torch_dtype
    raise ValueError(f"Unknown torch dtype: {te_dtype}")

def onnx_layernorm(input: torch.Tensor, 
                   layer_norm_weight: torch.Tensor, 
                   layer_norm_bias: torch.Tensor, 
                   eps: float, 
                   normalization: str, 
                   zero_centered_gamma: bool, 
                   output_dtype: torch.dtype,
                   return_layernorm_output: bool,
                   input_quantizer) -> torch.Tensor:
    """ ONNX LayerNorm used for inference. """
    ln_weight = layer_norm_weight if not zero_centered_gamma else layer_norm_weight + 1
    ln_weight = ln_weight.to(input.dtype).to(torch.float32)
    input = input.to(torch.float32)
    layer_norm_bias = (
        layer_norm_bias.to(output_dtype).to(torch.float32)
        if layer_norm_bias is not None
        else None
    )

    if normalization == "RMSNorm":
        ln_out = torch.nn.functional.rms_norm(input, input.shape[-1:], ln_weight, eps)
    else:
        ln_out = torch.nn.functional.layer_norm(input, input.shape[-1:], ln_weight, layer_norm_bias, eps)
    
    if input_quantizer is not None:
        if return_layernorm_output:
            # In case of return_layernorm_output, layernorm is not fused with fp8 cast,
            # so we cast to input_dtype and then perform cast to fp8 if needed
            ln_out = ln_out.to(output_dtype).to(torch.float32)
        ln_out_quantized = input_quantizer.onnx_quantize(ln_out)
        ln_out = input_quantizer.onnx_dequantize(ln_out_quantized)
    ln_out = ln_out.to(output_dtype)
    return ln_out