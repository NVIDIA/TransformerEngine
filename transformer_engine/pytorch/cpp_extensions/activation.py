# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for activation extensions"""
from typing import Union
import torch
import transformer_engine_extensions as tex


__all__ = []


def register_activation(name: str):
    def f(
        inp: torch.Tensor,
        fp8_meta_tensor: tex.FP8TensorMeta,
        fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
        otype: tex.DType,
    ) -> torch.Tensor:
        empty_tensor = torch.Tensor()
        if fp8_meta_tensor is not None:
            scale = fp8_meta_tensor.scale
            amax_history = fp8_meta_tensor.amax_history
            scale_inv = fp8_meta_tensor.scale_inv
        else:
            scale = empty_tensor
            amax_history = empty_tensor
            scale_inv = empty_tensor

        return getattr(torch.ops.tex_ts, f"{name.lower()}_ts")(
            inp,
            scale,
            amax_history,
            scale_inv,
            fp8_tensor,
            otype,
        )

    f.__doc__ = f"""{name} with FP8 output"""
    __all__.append(name.lower())
    return f


gelu = register_activation("GeLU")
relu = register_activation("ReLU")
geglu = register_activation("GeGLU")
reglu = register_activation("ReGLU")
swiglu = register_activation("SwiGLU")
