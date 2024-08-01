# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for activation extensions"""
from typing import Union
import torch
import transformer_engine_torch as tex


__all__ = ["gelu", "relu", "reglu", "geglu", "swiglu", "qgelu", "srelu"]


def gelu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """GeLU with FP8 output"""
    empty_tensor = torch.Tensor()
    if fp8_meta_tensor is not None:
        scale = fp8_meta_tensor.scale
        amax_history = fp8_meta_tensor.amax_history
        scale_inv = fp8_meta_tensor.scale_inv
    else:
        scale = empty_tensor
        amax_history = empty_tensor
        scale_inv = empty_tensor

    return torch.ops.tex_ts.gelu_ts(
        inp,
        scale,
        amax_history,
        scale_inv,
        fp8_tensor,
        otype,
    )


def relu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """ReLU with FP8 output"""
    empty_tensor = torch.Tensor()
    if fp8_meta_tensor is not None:
        scale = fp8_meta_tensor.scale
        amax_history = fp8_meta_tensor.amax_history
        scale_inv = fp8_meta_tensor.scale_inv
    else:
        scale = empty_tensor
        amax_history = empty_tensor
        scale_inv = empty_tensor
    return torch.ops.tex_ts.relu_ts(
        inp,
        scale,
        amax_history,
        scale_inv,
        fp8_tensor,
        otype,
    )


def geglu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """GeGLU with FP8 output"""
    empty_tensor = torch.Tensor()
    if fp8_meta_tensor is not None:
        scale = fp8_meta_tensor.scale
        amax_history = fp8_meta_tensor.amax_history
        scale_inv = fp8_meta_tensor.scale_inv
    else:
        scale = empty_tensor
        amax_history = empty_tensor
        scale_inv = empty_tensor
    return torch.ops.tex_ts.geglu_ts(
        inp,
        scale,
        amax_history,
        scale_inv,
        fp8_tensor,
        otype,
    )


def reglu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """ReGLU with FP8 output"""
    empty_tensor = torch.Tensor()
    if fp8_meta_tensor is not None:
        scale = fp8_meta_tensor.scale
        amax_history = fp8_meta_tensor.amax_history
        scale_inv = fp8_meta_tensor.scale_inv
    else:
        scale = empty_tensor
        amax_history = empty_tensor
        scale_inv = empty_tensor
    return torch.ops.tex_ts.reglu_ts(
        inp,
        scale,
        amax_history,
        scale_inv,
        fp8_tensor,
        otype,
    )


def swiglu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """SwiGLU with FP8 output"""
    empty_tensor = torch.Tensor()
    if fp8_meta_tensor is not None:
        scale = fp8_meta_tensor.scale
        amax_history = fp8_meta_tensor.amax_history
        scale_inv = fp8_meta_tensor.scale_inv
    else:
        scale = empty_tensor
        amax_history = empty_tensor
        scale_inv = empty_tensor
    return torch.ops.tex_ts.swiglu_ts(
        inp,
        scale,
        amax_history,
        scale_inv,
        fp8_tensor,
        otype,
    )


def qgelu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """QuickGELU with FP8 output"""
    empty_tensor = torch.Tensor()
    if fp8_meta_tensor is not None:
        scale = fp8_meta_tensor.scale
        amax_history = fp8_meta_tensor.amax_history
        scale_inv = fp8_meta_tensor.scale_inv
    else:
        scale = empty_tensor
        amax_history = empty_tensor
        scale_inv = empty_tensor
    return torch.ops.tex_ts.qgelu_ts(
        inp,
        scale,
        amax_history,
        scale_inv,
        fp8_tensor,
        otype,
    )


def srelu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """ReLU with FP8 output"""
    empty_tensor = torch.Tensor()
    if fp8_meta_tensor is not None:
        scale = fp8_meta_tensor.scale
        amax_history = fp8_meta_tensor.amax_history
        scale_inv = fp8_meta_tensor.scale_inv
    else:
        scale = empty_tensor
        amax_history = empty_tensor
        scale_inv = empty_tensor
    return torch.ops.tex_ts.srelu_ts(
        inp,
        scale,
        amax_history,
        scale_inv,
        fp8_tensor,
        otype,
    )
