# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for cast extensions"""
from typing import Optional, Union
import torch
import transformer_engine_torch as tex


__all__ = ["cast_to_fp8", "cast_from_fp8"]


def cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Cast input to FP8"""

    if out is not None:
        if inp.nelement() > 0:
            torch.ops.tex_ts.cast_to_fp8_noalloc_ts(
                inp,
                fp8_meta_tensor.scale,
                out,
                fp8_meta_tensor.amax_history,
                fp8_meta_tensor.scale_inv,
                fp8_tensor,
                otype,
            )
        return None

    return torch.ops.tex_ts.cast_to_fp8_ts(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
    )


def cast_from_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    itype: tex.DType,
    otype: tex.DType,
) -> torch.Tensor:
    """Cast input from FP8"""
    return torch.ops.tex_ts.cast_from_fp8_ts(
        inp,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        itype,
        otype,
    )
