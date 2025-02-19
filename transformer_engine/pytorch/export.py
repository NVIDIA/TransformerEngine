# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Export utilities for TransformerEngine"""
from contextlib import contextmanager
import torch
import onnxscript
from onnxscript import opset18 as op
from .tensor.float8_tensor import Float8Quantizer
from .tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from .tensor.quantized_tensor import Quantizer
from .constants import TE_DType
from typing import Generator, Optional

_IN_ONNX_EXPORT_MODE = False


@contextmanager
def onnx_export(enabled: bool = False) -> Generator[None, None, None]:
    """
    Context manager for exporting to ONNX.

    .. code-block:: python

        with onnx_export(enabled=True):
            torch.onnx.export(model, dynamo=True)

    Parameters
    ----------
    enabled: bool, default = `False`
             whether or not to enable export
    """

    global _IN_ONNX_EXPORT_MODE
    onnx_export_state = _IN_ONNX_EXPORT_MODE
    try:
        _IN_ONNX_EXPORT_MODE = enabled
        yield
    finally:
        _IN_ONNX_EXPORT_MODE = onnx_export_state


def is_in_onnx_export_mode() -> bool:
    """Returns True if onnx export mode is enabled, False otherwise."""
    return _IN_ONNX_EXPORT_MODE

# ONNX GEMM for inference

@torch.library.custom_op("tex::gemm_inf", mutates_args=[])
def torch_onnx_gemm_inf_op(inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """ Gemm used for inference -- weight is transposed"""
    out = inp @ weight.T
    if bias is not None:
        out = out + bias
    return out

@torch_onnx_gemm_inf_op.register_fake
def _(inp, weight, bias):
    """ Fake gemm used for inference. """    
    out = inp @ weight.T
    if bias is not None:
        out = out + bias
    return out

_onnx_opset = onnxscript.values.Opset("tex", version=1)
@onnxscript.script(_onnx_opset, default_opset=op)
def onnx_gemm_inf_symbolic(inp, weight, bias):
    """ Symbolic gemm used for inference. """
    return op.Gemm(inp, weight, bias, transA=1, transB=0)

# ONNX Quantization

def _get_te_dtype(id: int):
    """ Get the tex.DType enum value from the given id - reverse for int(tex.DType.*). """
    all_te_dtypes = list(tex.DType.__members__.values())
    for te_dtype in all_te_dtypes:
        if int(te_dtype) == id:
            return te_dtype
    raise ValueError(f"Unknown transformer engine dtype id: {id}")

def _get_torch_dtype(te_dtype: tex.DType):
    """ Get the torch.dtype value from the given tex.DType enum value. """
    all_torch_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    for torch_dtype in all_torch_dtypes:
        if TE_DType[torch_dtype] == te_dtype:
            return torch_dtype
    raise ValueError(f"Unknown torch dtype: {te_dtype}")

def _get_quantizer_class(quantizer_id: int):
    """
    Dynamically retrieves the quantizer class based on the given quantizer_id.
    This way, when new quantizers are added, no modifications in this file are required.
    """
    for cls in Quantizer.__subclasses__():
        if id(cls) == quantizer_id:
            return cls
    raise ValueError(f"Unknown quantizer id: {quantizer_id}")

@torch.library.custom_op("tex::quantize", mutates_args=[])
def onnx_quantize_op(
    tensor: torch.Tensor, quantizer_id: int, amax: torch.Tensor, 
    scale: torch.Tensor, scale_inv: torch.Tensor, fp8_dtype: int) -> torch.Tensor:
    """ Quantize used for inference. """
    quantizer_type = _get_quantizer_class(quantizer_id)
    quantizer = quantizer_type(amax, scale, _get_te_dtype(fp8_dtype))
    return quantizer.quantize(tensor)._data

@onnx_quantize_op.register_fake
def _(tensor, *_):
    """ Fake quantize used for inference. """
    return torch.empty(tensor.shape, dtype=torch.uint8, device=tensor.device)

_onnx_opset = onnxscript.values.Opset("tex", version=1)
@onnxscript.script(_onnx_opset, default_opset=op)
def onnx_quantize_symbolic(tensor, quantizer_id: int, amax, scale, scale_inv, fp8_dtype: int):
    """ Symbolic quantize used for inference. """
    return op.TRT_FP8QuantizeLinear(tensor, scale_inv)

# ONNX Dequantization

@torch.library.custom_op("tex::dequantize", mutates_args=[])
def onnx_dequantize_op(tensor: torch.Tensor, quantizer_id: int, 
                       amax: torch.Tensor, scale: torch.Tensor, 
                       fp8_dtype: int, fake_dtype: int) -> torch.Tensor:
    """ Dequantize used for inference. """
    quantizer_type = _get_quantizer_class(quantizer_id)
    quantizer = quantizer_type(amax, scale, _get_te_dtype(fp8_dtype))
    te_dtype = _get_te_dtype(fake_dtype)
    torch_dtype = _get_torch_dtype(te_dtype)
    quantizer_tensor = quantizer.create_tensor_from_data(tensor, fake_dtype=torch_dtype)
    return quantizer_tensor.dequantize()

@onnx_dequantize_op.register_fake
def _(tensor: torch.Tensor, quantizer_id: int, 
      amax: torch.Tensor, scale: torch.Tensor, 
      fp8_dtype: int, fake_dtype: int) -> torch.Tensor:
    """ Fake dequantize used for inference. """
    te_dtype = _get_te_dtype(fake_dtype)
    torch_dtype = _get_torch_dtype(te_dtype)
    return torch.empty(tensor.shape, dtype=torch_dtype, device=tensor.device)

@onnxscript.script(_onnx_opset, default_opset=op)
def onnx_dequantize_symbolic(tensor, quantizer_id: int, amax, scale, fp8_dtype: int, fake_dtype: int):
    """ Symbolic dequantize used for inference. """
    out = op.TRT_FP8DequantizeLinear(tensor, scale)
    return out


# This translation table should be passed to torch.onnx.export function
# using the custom_translation_table=te_translation_table option
te_translation_table = {
    torch.ops.tex.gemm_inf.default: onnx_gemm_inf_symbolic,
    torch.ops.tex.quantize.default: onnx_quantize_symbolic,
    torch.ops.tex.dequantize.default: onnx_dequantize_symbolic
}
