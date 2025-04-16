# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import Union, List

import torch

import transformer_engine_torch as tex

from ..jit import no_torch_dynamo


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


class Fp8Padding(torch.nn.Module):
    """
    Apply the padding for Grouped GEMM input.

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
    ) -> Union[torch.Tensor, List[int]]:
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
        padded_m_splits = [(m + 15) // 16 * 16 for m in m_splits]

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
