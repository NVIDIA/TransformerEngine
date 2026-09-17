# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross Entropy Loss API"""

from typing import Optional
import warnings

import torch

import transformer_engine.pytorch.triton.cross_entropy as triton_cross_entropy
from transformer_engine.pytorch.jit import no_torch_dynamo

__all__ = ["parallel_cross_entropy"]


class CrossEntropyFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Cross Entropy loss. The input
    tensor can be in BF16/FP32, and loss and gradient calculations happen in FP32. The
    returned loss is always in FP32.
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
        overwrite_input=False,
    ):
        """Compute the loss and save the input and softmax statistics for backward."""

        (
            loss,
            saved_input,
            stats,
            target,
            n_non_ignore,
            rank,
            world_size,
        ) = triton_cross_entropy.cross_entropy_forward(
            inp,
            target,
            label_smoothing,
            reduce_loss,
            dist_process_group,
            ignore_idx,
            overwrite_input,
        )
        tensors_to_save = (saved_input.detach(), stats, target)
        if reduce_loss:
            tensors_to_save += (n_non_ignore,)
        ctx.save_for_backward(*tensors_to_save)
        ctx.label_smoothing = label_smoothing
        ctx.reduce_loss = reduce_loss
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.ignore_idx = ignore_idx
        ctx.overwrite_input = overwrite_input
        ctx.did_backward = False
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Reconstruct the input gradient from the tensors saved during forward."""

        if ctx.did_backward:
            raise RuntimeError(
                "parallel_cross_entropy does not support repeated backward passes "
                "because backward reuses its saved input buffer"
            )
        ctx.did_backward = True
        saved_input, stats, target, *optional_tensors = ctx.saved_tensors
        n_non_ignore = optional_tensors[0] if ctx.reduce_loss else None
        grad_input = triton_cross_entropy.cross_entropy_backward(
            saved_input,
            stats,
            target,
            n_non_ignore,
            grad_output,
            ctx.label_smoothing,
            ctx.reduce_loss,
            ctx.rank,
            ctx.world_size,
            ctx.ignore_idx,
        )
        if ctx.overwrite_input:
            torch.autograd.graph.increment_version(saved_input)
        return grad_input, None, None, None, None, None, None


def _validate_inputs(
    inp: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    overwrite_input: bool,
) -> None:
    """Validate assumptions made by the cross entropy Triton kernels."""

    if inp.ndim != 3:
        raise ValueError(f"inp must be a 3D tensor, but got shape {tuple(inp.shape)}")
    if any(size == 0 for size in inp.shape):
        raise ValueError(f"inp dimensions must be non-zero, but got shape {tuple(inp.shape)}")
    if inp.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"inp must have BF16 or FP32 dtype, but got {inp.dtype}")
    if not inp.is_cuda:
        raise ValueError("inp must be a CUDA tensor")
    if target.device != inp.device:
        raise ValueError("target must be on the same CUDA device as inp")
    if target.dtype != torch.int64:
        raise TypeError(f"target must have torch.int64 dtype, but got {target.dtype}")
    if target.numel() != inp.shape[0] * inp.shape[1]:
        raise ValueError(
            "Each input row needs one target token ID: "
            f"expected {inp.shape[0] * inp.shape[1]}, got {target.numel()}"
        )
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1], but got {label_smoothing}")
    if overwrite_input and not inp.is_contiguous():
        raise ValueError("overwrite_input=True requires a contiguous input")


@no_torch_dynamo()
def _parallel_cross_entropy_overwrite_input(
    inp: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_process_group: Optional[torch.distributed.ProcessGroup],
    ignore_idx: int,
) -> torch.Tensor:
    """Run destructive cross entropy outside Torch Dynamo's compiled graph."""

    return CrossEntropyFunction.apply(
        inp,
        target,
        label_smoothing,
        reduce_loss,
        dist_process_group,
        ignore_idx,
        True,
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
    overwrite_input: bool = False,
    _input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross entropy loss with optional distributed reduction.

    Loss and derivative calculations use FP32 arithmetic for BF16 and FP32
    inputs. The returned loss is FP32.

    By default, ``inp`` is preserved. With ``overwrite_input=True``, ``inp``
    must be contiguous and its storage is overwritten during backward. Callers
    must not read or reuse it after starting backward. Only one backward pass
    is supported for each loss result.

    Parameters
    ----------
    inp : torch.Tensor
        Input logits with shape ``(B, SQ, V)`` or ``(SQ, B, V)``.
    target : torch.Tensor
        Target token IDs with one value for each input row.
    label_smoothing : float, default = 0.0
        Amount of label smoothing.
    reduce_loss : bool, default = False
        Return the mean loss over non-ignored targets when True.
    dist_process_group : torch.distributed.ProcessGroup, default = None
        Tensor-parallel process group, or None for a single device. Every rank
        must provide an equally sized vocabulary shard, and shards must be
        ordered by process-group rank.
    ignore_idx : int, default = -100
        Target value for ignored rows.
    is_cg_capturable : bool, default = False
        Deprecated and unused. The operation is always CUDA graph capturable.
    overwrite_input : bool, default = False
        Allow ``inp`` to be overwritten during backward. The input must be
        contiguous and cannot be reused afterward. This mode is incompatible with
        ``torch.compile`` and will result in a graph break if used in that context.

    Returns
    -------
    torch.Tensor
        The computed loss.
    """

    if _input is not None:
        warnings.warn(
            "The '_input' parameter is deprecated. Please use 'inp' instead.",
            FutureWarning,
        )
        inp = _input

    if is_cg_capturable:
        warnings.warn(
            "The 'is_cg_capturable' parameter is deprecated and has no effect. "
            "The operation is always CUDA graph capturable.",
            FutureWarning,
        )

    _validate_inputs(inp, target, label_smoothing, overwrite_input)
    if overwrite_input:
        return _parallel_cross_entropy_overwrite_input(
            inp,
            target,
            label_smoothing,
            reduce_loss,
            dist_process_group,
            ignore_idx,
        )
    return CrossEntropyFunction.apply(
        inp,
        target,
        label_smoothing,
        reduce_loss,
        dist_process_group,
        ignore_idx,
        overwrite_input,
    )
