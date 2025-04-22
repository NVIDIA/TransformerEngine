# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import List, Optional

import torch

import transformer_engine_torch as tex

from ..fp8 import FP8GlobalStateManager
from ..jit import no_torch_dynamo
from ..tensor.float8_blockwise_tensor import Float8BlockwiseQTensor

__all__ = ["Fp8Unpadding"]


class _Fp8Unpadding(torch.autograd.Function):
    """functional FP8 unpadding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        inputmats = torch.split(inp.view(-1, inp.shape[-1]), padded_m_splits)
        out_ret = torch.cat(
            [grad_output_mat[: m_splits[i]] for i, grad_output_mat in enumerate(inputmats)], dim=0
        )

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out_ret

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring
        grad_input = None
        if ctx.requires_dgrad:
            grad_output = grad_output.contiguous()

            in_features = grad_output.shape[-1]

            # Allocate cast and transpose output tensor
            total_row = sum(ctx.padded_m_splits)
            grad_input = torch.empty(
                [total_row, in_features], dtype=grad_output.dtype, device=grad_output.device
            )
            # FP8 pad input for forward, FP8 input transpose for backward wgrad
            tex.fused_multi_row_padding(
                grad_output.view(-1, in_features), grad_input, ctx.m_splits, ctx.padded_m_splits
            )

        return (grad_input, None, None, None)


class _Fp8Unpadding_Fp8Grad(torch.autograd.Function):
    """
    functional FP8 unpadding

    forward:
        padded bf16 input -> bf16 output
    backward:
        fp8 input -> padded fp8 output
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
        inputmats = torch.split(inp.view(-1, inp.shape[-1]), padded_m_splits)
        out_ret = torch.cat(
            [grad_output_mat[: m_splits[i]] for i, grad_output_mat in enumerate(inputmats)], dim=0
        )

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out_ret

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring
        grad_input = None
        if ctx.requires_dgrad:
            grad_output = grad_output.contiguous()

            assert isinstance(
                grad_output, Float8BlockwiseQTensor
            ), "Fp8Unpadding_Fp8Grad only support Float8BlockwiseQTensor now"

            in_features = grad_output._rowwise_data.shape[-1]
            in_scale_features = grad_output._rowwise_scale_inv.shape[0]
            assert (
                grad_output._rowwise_data.shape[0] == grad_output._rowwise_scale_inv.shape[1]
            ), "The number of rows of hidden and scale should be the same"

            # Allocate cast and transpose output tensor
            total_row = sum(ctx.padded_m_splits)
            grad_input_data = torch.empty(
                [total_row, in_features],
                dtype=grad_output._rowwise_data.dtype,
                device=grad_output.device,
            )
            grad_input_scale = torch.empty(
                [total_row, in_scale_features],
                dtype=grad_output._rowwise_scale_inv.dtype,
                device=grad_output.device,
            )

            rowwise_data = grad_output._rowwise_data.view(-1, in_features)
            rowwise_scale_inv = grad_output._rowwise_scale_inv.T.view(
                -1, in_scale_features
            ).contiguous()

            tex.fused_multi_row_padding(
                rowwise_data, grad_input_data, ctx.m_splits, ctx.padded_m_splits
            )
            tex.fused_multi_row_padding(
                rowwise_scale_inv, grad_input_scale, ctx.m_splits, ctx.padded_m_splits
            )

            # FP8 pad input for forward, FP8 input transpose for backward wgrad
            grad_input = Float8BlockwiseQTensor(
                shape=grad_input_data.shape,
                dtype=grad_output.dtype,
                rowwise_data=grad_input_data,
                rowwise_scale_inv=grad_input_scale.T.contiguous(),
                columnwise_data=None,
                columnwise_scale_inv=None,
                fp8_dtype=grad_output._fp8_dtype,
                quantizer=None,
                is_2D_scaled=False,
                requires_grad=grad_output.requires_grad,
            )

        return (grad_input, None, None, None)


class Fp8Unpadding(torch.nn.Module):
    """
    Apply the unpadding for Grouped GEMM input.

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
        fp8_grad: bool = False,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms
        if align_size is None:
            self.align_size = 32 if FP8GlobalStateManager.get_fp8_recipe().mxfp8() else 16
        else:
            self.align_size = align_size
        self.fp8_grad = fp8_grad

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
    ) -> torch.Tensor:
        """
        Apply the unpadding to the input.

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
            return inp

        if self.fp8_grad:
            if torch.is_grad_enabled():
                fn = _Fp8Unpadding_Fp8Grad.apply
                args = []
            else:
                fn = _Fp8Unpadding_Fp8Grad.forward
                args = [None]
        else:
            if torch.is_grad_enabled():
                fn = _Fp8Unpadding.apply
                args = []
            else:
                fn = _Fp8Unpadding.forward
                args = [None]

        args += (
            inp,
            m_splits,
            padded_m_splits,
            torch.is_grad_enabled(),
        )
        out = fn(*args)

        return out
