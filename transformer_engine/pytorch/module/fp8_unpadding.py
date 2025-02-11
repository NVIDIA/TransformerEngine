# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import List

import torch

import transformer_engine_torch as tex

from ..jit import no_torch_dynamo


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


class Fp8Unpadding(torch.nn.Module):
    """
    Apply the unpadding for Grouped GEMM input.

    Parameters
    ----------
    num_gemms: int
               number of GEMMs to be performed simutaneously.
    """

    def __init__(
        self,
        num_gemms,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms

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
        padded_m_splits = [(m + 15) // 16 * 16 for m in m_splits]

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
