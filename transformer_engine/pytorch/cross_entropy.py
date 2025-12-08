# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross Entropy Loss API"""

from typing import Optional
import warnings

import torch

import transformer_engine.pytorch.triton.cross_entropy as triton_cross_entropy

__all__ = [
    "parallel_cross_entropy",
]


class CrossEntropyFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Cross Entropy loss. The input tensor can be in BF16/FP32, the
    loss and gradient calculation happens in FP32 only. The returned loss is always in FP32, the input gradients are upcasted
    to the dataype of the input.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
        is_cg_capturable=False,
    ):
        """
        The forward pass of the Cross Entropy loss. If dist_process_group is passed for distributed loss calculation, the input to each
        distributed rank should be (*,V/world_size). Note that each of the ranks should get equal shards along the V dimension.

        Parameters:
        ctx : The context object.
        inp (tensor): The input tensor of shape (B, SQ, V) or (SQ, B, V) where B is batch size, SQ is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (B,SQ) or (SQ, B) where each value is in [0, V-1].
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduce_loss (bool): If true, returns the averaged loss across the B*SQ dimension.
        dist_process_group (torch.dist.ProcessGroup): The distributed process group the loss computation is split across, None if on 1 device.
        ignore_idx (int): The index for which loss and gradients are made to zero

        Returns:
        tensor: The computed loss.
        """
        loss, inp = triton_cross_entropy.cross_entropy_forward(
            inp,
            target,
            label_smoothing,
            reduce_loss,
            dist_process_group,
            ignore_idx,
        )

        ctx.save_for_backward(inp.detach())
        ctx.is_cg_capturable = is_cg_capturable
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass of the Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.

        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        (inp,) = ctx.saved_tensors
        inp = triton_cross_entropy.cross_entropy_backward(inp, grad_output, ctx.is_cg_capturable)
        return (
            inp,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def parallel_cross_entropy(
    inp: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    reduce_loss: bool = False,
    dist_process_group: Optional[torch.distributed.ProcessGroup] = None,
    ignore_idx: int = -100,
    is_cg_capturable: bool = False,
    *,
    _input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cross Entropy loss with optional distributed reduction.

    The input tensor can be in BF16/FP32, the loss and gradient calculation happens in
    FP32 only. The returned loss is always in FP32, the input gradients are upcasted
    to the datatype of the input.

    If ``dist_process_group`` is passed for distributed loss calculation, the input to each
    distributed rank should be ``(*, V/world_size)``. Note that each of the ranks should
    get equal shards along the V dimension.

    Parameters
    ----------
    inp : torch.Tensor
        The input tensor of shape ``(B, SQ, V)`` or ``(SQ, B, V)`` where B is batch size,
        SQ is sequence length, V is vocab size.
    target : torch.Tensor
        The target tensor of shape ``(B, SQ)`` or ``(SQ, B)`` where each value is in ``[0, V-1]``.
    label_smoothing : float, default = 0.0
        The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    reduce_loss : bool, default = False
        If True, returns the averaged loss across the B*SQ dimension.
    dist_process_group : torch.distributed.ProcessGroup, default = None
        The distributed process group the loss computation is split across, None if on 1 device.
    ignore_idx : int, default = -100
        The index for which loss and gradients are made to zero.
    is_cg_capturable : bool, default = False
        Whether the operation is CUDA graph capturable.

    Returns
    -------
    torch.Tensor
        The computed loss.
    """
    # Handle backward compatibility with _input parameter
    if _input is not None:
        warnings.warn(
            "The '_input' parameter is deprecated. Please use 'inp' instead.",
            FutureWarning,
        )
        inp = _input

    return CrossEntropyFunction.apply(
        inp,
        target,
        label_smoothing,
        reduce_loss,
        dist_process_group,
        ignore_idx,
        is_cg_capturable,
    )
