# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for transpose extensions"""
from typing import List, Optional, Tuple, Union
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType


__all__ = [
    "fp8_cast_transpose_fused",
    "fp8_cast_transpose_bgrad_fused",
    "fp8_cast_transpose_bgrad_dgelu_fused",
    "fp8_multi_cast_transpose_fused",
    "fp8_transpose_bgrad_fused",
]


def fp8_cast_transpose_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    cast_out: Optional[torch.Tensor] = None,
    transpose_out: Optional[torch.Tensor] = None,
    noop_flag: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """Cast + Transpose with FP8 output"""

    return_outputs = False
    if transpose_out is None:
        transpose_out = torch.empty(inp.shape[1], inp.shape[0], device="cuda", dtype=torch.uint8)
        return_outputs = True
    if cast_out is None:
        cast_out = torch.empty_like(inp, dtype=torch.uint8)
        return_outputs = True

    if noop_flag is None:
        noop_flag = torch.Tensor()

    if inp.nelement() > 0:
        tex.fused_cast_transpose_noop(
            inp,
            noop_flag,
            fp8_meta_tensor.scale,
            fp8_meta_tensor.amax_history,
            fp8_meta_tensor.scale_inv,
            cast_out,
            transpose_out,
            otype,
            scale_offset=int(fp8_tensor),
            amax_offset=int(fp8_tensor),
            scale_inv_offset=int(fp8_tensor),
        )

    if return_outputs:
        return cast_out, transpose_out
    return None


def fp8_cast_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD with FP8 output"""
    return tex.fused_cast_transpose_bgrad(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        otype,
        scale_offset=int(fp8_tensor),
        amax_offset=int(fp8_tensor),
        scale_inv_offset=int(fp8_tensor),
    )


def fp8_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    grad_bias_type: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose + BGRAD with FP8 output"""
    return tex.fused_fp8_transpose_bgrad(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        otype,
        TE_DType[grad_bias_type],
        scale_offset=int(fp8_tensor),
        amax_offset=int(fp8_tensor),
        scale_inv_offset=int(fp8_tensor),
    )


def fp8_cast_transpose_bgrad_dgelu_fused(
    grad_output: torch.Tensor,
    gelu_input: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD + DGELU with FP8 output"""
    return tex.fused_cast_transpose_bgrad_dgelu(
        grad_output,
        gelu_input,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        otype,
        scale_offset=int(fp8_tensor),
        amax_offset=int(fp8_tensor),
        scale_inv_offset=int(fp8_tensor),
    )


def fp8_multi_cast_transpose_fused(
    input_list: List[torch.Tensor],
    fp8_meta_tensor: tex.FP8TensorMeta,
    scale_indices: List[int],
    amax_indices: List[int],
    scale_inv_indices: List[int],
    otype: tex.DType,
    cast_output_list: Optional[List[torch.Tensor]] = None,
    transposed_output_list: Optional[List[torch.Tensor]] = None,
) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], None]:
    """Cast + Transpose with FP8 output"""

    return_outputs = False
    if transposed_output_list is None:
        transposed_output_list = [
            torch.empty(inp.shape[1], inp.shape[0], device="cuda", dtype=torch.uint8)
            for inp in input_list
        ]
        return_outputs = True
    if cast_output_list is None:
        cast_output_list = [
            torch.empty_like(inp, dtype=torch.uint8) for inp in input_list
        ]
        return_outputs = True

    tex.fused_multi_cast_transpose(
        input_list,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        scale_indices,
        amax_indices,
        scale_inv_indices,
        cast_output_list,
        transposed_output_list,
        otype,
    )

    if return_outputs:
        return cast_output_list, transposed_output_list
    return None
