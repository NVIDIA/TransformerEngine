# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross Entropy Loss API"""

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
        _input,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
    ):
        """
        The forward pass of the Cross Entropy loss. If dist_process_group is passed for distributed loss calculation, the input to each
        distributed rank should be (*,V/world_size). Note that each of the ranks should get equal shards along the V dimension.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor of shape (B, SQ, V) or (SQ, B, V) where B is batch size, SQ is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (B,SQ) or (SQ, B) where each value is in [0, V-1].
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduce_loss (bool): If true, returns the averaged loss across the B*SQ dimension.
        dist_process_group (torch.dist.ProcessGroup): The distributed process group the loss computation is split across, None if on 1 device.
        ignore_idx (int): The index for which loss and gradients are made to zero

        Returns:
        tensor: The computed loss.
        """
        loss, _input = triton_cross_entropy.cross_entropy_forward(
            _input, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx
        )

        ctx.save_for_backward(_input.detach())
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
        (_input,) = ctx.saved_tensors
        _input = triton_cross_entropy.cross_entropy_backward(_input, grad_output)
        return (
            _input,
            None,
            None,
            None,
            None,
        )


parallel_cross_entropy = CrossEntropyFunction.apply
