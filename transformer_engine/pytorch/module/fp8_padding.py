# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import List, Optional, Tuple

import torch

import transformer_engine_torch as tex

from ..quantization import FP8GlobalStateManager, get_align_size_for_quantization
from ..jit import no_torch_dynamo


__all__ = ["Fp8Padding"]


class _Fp8Padding(torch.autograd.Function):
    """functional FP8 padding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        non_tensor_args: Tuple,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Reduce number of arguments to autograd function in order
        # to reduce CPU overhead due to pytorch arg checking.
        (m_splits, padded_m_splits, is_grad_enabled) = non_tensor_args

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

            in_features = grad_output.shape[-1]

            # Allocate cast and transpose output tensor
            total_row = sum(ctx.m_splits)
            grad_input = torch.empty(
                [total_row, in_features], dtype=grad_output.dtype, device=grad_output.device
            )

            tex.fused_multi_row_unpadding(
                grad_output.view(-1, in_features), grad_input, ctx.padded_m_splits, ctx.m_splits
            )

        return grad_input, None


class Fp8Padding(torch.nn.Module):
    """
    Apply the padding for Grouped GEMM input.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simultaneously.
    align_size : int, optional
                 the alignment size for the input tensor. If not provided, the alignment size will
                 be determined by the FP8/FP4 recipe (32 for MXFP8/NVFP4 and 16 for others) in the first
                 forward pass.
    """

    def __init__(
        self,
        num_gemms: int,
        align_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms
        self.align_size = align_size

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
        if self.align_size is None:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            self.align_size = get_align_size_for_quantization(recipe)

        # FP8 padding calculate
        padded_m_splits = [
            (m + self.align_size - 1) // self.align_size * self.align_size for m in m_splits
        ]
        # no padding needed
        if m_splits == padded_m_splits:
            return inp, m_splits

        is_grad_enabled = torch.is_grad_enabled()

        if is_grad_enabled:
            fn = _Fp8Padding.apply
            autograd_ctx = []
        else:
            fn = _Fp8Padding.forward
            autograd_ctx = [None]

        non_tensor_args = (
            m_splits,
            padded_m_splits,
            is_grad_enabled,
        )
        out = fn(*autograd_ctx, inp, non_tensor_args)

        return out, padded_m_splits
