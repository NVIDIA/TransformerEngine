# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import List, Optional, Tuple

import torch

import transformer_engine_torch as tex

from ..fp8 import FP8GlobalStateManager
from ..jit import no_torch_dynamo
from ..tensor.float8_blockwise_tensor import Float8BlockwiseQTensor

__all__ = ["Fp8Padding"]


class _Fp8Padding(torch.autograd.Function):
    """functional FP8 padding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        # Make sure input dimensions are compatible
        in_features = inp.shape[-1]

        # Allocate cast and transpose output tensor
        total_row = sum(padded_m_splits)
        out = torch.empty([total_row, in_features], dtype=inp.dtype, device=inp.device)

        tex.fused_multi_row_padding(inp.view(-1, in_features), out, m_splits, padded_m_splits)

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring

        grad_input = None
        if ctx.requires_dgrad:
            grad_output = grad_output.contiguous()

            grad_output_mats = torch.split(
                grad_output.view(-1, grad_output.shape[-1]), ctx.padded_m_splits
            )
            grad_input = torch.cat(
                [
                    grad_output_mat[: ctx.m_splits[i]]
                    for i, grad_output_mat in enumerate(grad_output_mats)
                ],
                dim=0,
            )

        return (grad_input, None, None, None)

class _Fp8Padding_FromFp8(torch.autograd.Function):
    """
    functional FP8 padding from fp8 input
    
    forward:
        fp8 input -> padded fp8 output
    backward:
        padded bf16 input -> bf16 output
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        # Make sure input dimensions are compatible

        assert isinstance(inp, Float8BlockwiseQTensor), "Fp8Padding_FromFp8 only support Float8BlockwiseQTensor now"
        
        in_features = inp._rowwise_data.shape[-1]
        in_scale_features = inp._rowwise_scale_inv.shape[0]
        assert inp._rowwise_data.shape[0] == inp._rowwise_scale_inv.shape[1], 'The number of rows of hidden and scale should be the same'

        # Allocate cast and transpose output tensor
        total_row = sum(padded_m_splits)

        out_data = torch.empty([total_row, in_features], dtype=inp._rowwise_data.dtype, device=inp.device)
        out_scale_inv = torch.empty([total_row, in_scale_features], dtype=inp._rowwise_scale_inv.dtype, device=inp.device)

        rowwise_data = inp._rowwise_data.view(-1, in_features)
        rowwise_scale_inv = inp._rowwise_scale_inv.T.view(-1, in_scale_features).contiguous()
        
        tex.fused_multi_row_padding(rowwise_data, out_data, m_splits, padded_m_splits)
        tex.fused_multi_row_padding(rowwise_scale_inv, out_scale_inv, m_splits, padded_m_splits)
        
        padded_tensor = Float8BlockwiseQTensor(
            shape=out_data.shape,
            dtype=inp.dtype,
            rowwise_data=out_data,
            rowwise_scale_inv=out_scale_inv.T.contiguous(),
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=inp._fp8_dtype,
            quantizer=None,
            is_2D_scaled=False,
            requires_grad=inp.requires_grad,
        )

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return padded_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring

        grad_input = None
        if ctx.requires_dgrad:
            grad_output = grad_output.contiguous()

            grad_output_mats = torch.split(
                grad_output.view(-1, grad_output.shape[-1]), ctx.padded_m_splits
            )
            grad_input = torch.cat(
                [
                    grad_output_mat[: ctx.m_splits[i]]
                    for i, grad_output_mat in enumerate(grad_output_mats)
                ],
                dim=0,
            )

        return (grad_input, None, None, None)

class Fp8Padding(torch.nn.Module):
    """
    Apply the padding for Grouped GEMM input.

    Parameters
    ----------
    num_gemms: int
               number of GEMMs to be performed simutaneously.
    align_size: int, optional
                the alignment size for the input tensor. If not provided, the alignment size  will
                be determined by the FP8 recipe, 32 for MXFP8 and 16 for others.
    """

    def __init__(
        self,
        num_gemms: int,
        align_size: Optional[int] = None,
        fp8_input: bool = False,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms
        if align_size is None:
            self.align_size = 32 if FP8GlobalStateManager.get_fp8_recipe().mxfp8() else 16
        else:
            self.align_size = align_size
        self.fp8_input = fp8_input

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Apply the padding to the input.

        Parameters
        ----------
        inp : torch.Tensor
                Input tensor.
        m_splits : List[int]
                    List of integers representing the split of the input tensor.
        """

        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."

        # FP8 padding calculate
        padded_m_splits = [
            (m + self.align_size - 1) // self.align_size * self.align_size for m in m_splits
        ]
        # no padding needed
        if m_splits == padded_m_splits:
            return inp, m_splits

        if self.fp8_input:
            if torch.is_grad_enabled():
                fn = _Fp8Padding_FromFp8.apply
                args = []
            else:
                fn = _Fp8Padding_FromFp8.forward
                args = [None]
        else :
            if torch.is_grad_enabled():
                fn = _Fp8Padding.apply
                args = []
            else:
                fn = _Fp8Padding.forward
                args = [None]

        args += (
            inp,
            m_splits,
            padded_m_splits,
            torch.is_grad_enabled(),
        )
        out = fn(*args)

        return out, padded_m_splits
