# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross Entropy Loss API"""

from typing import Optional
import warnings

import torch

import transformer_engine.pytorch.triton.cross_entropy as triton_cross_entropy

__all__ = ["parallel_cross_entropy"]


class CrossEntropyFunction(torch.autograd.Function):
    """Cross entropy autograd function that recomputes its derivative."""

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
        overwrite_input=False,
    ):
        loss, logits, stats, target, n_non_ignore = triton_cross_entropy.cross_entropy_forward(
            inp,
            target,
            label_smoothing,
            reduce_loss,
            dist_process_group,
            ignore_idx,
            overwrite_input,
        )
        ctx.save_for_backward(logits.detach(), stats, target, n_non_ignore)
        ctx.label_smoothing = label_smoothing
        ctx.reduce_loss = reduce_loss
        ctx.dist_process_group = dist_process_group
        ctx.ignore_idx = ignore_idx
        ctx.is_cg_capturable = is_cg_capturable
        ctx.did_backward = False
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.did_backward:
            raise RuntimeError(
                "parallel_cross_entropy does not support repeated backward passes "
                "because backward reuses its saved logits buffer"
            )
        ctx.did_backward = True
        logits, stats, target, n_non_ignore = ctx.saved_tensors
        grad_input = triton_cross_entropy.cross_entropy_backward(
            logits,
            stats,
            target,
            n_non_ignore,
            grad_output,
            ctx.label_smoothing,
            ctx.reduce_loss,
            ctx.dist_process_group,
            ctx.ignore_idx,
            ctx.is_cg_capturable,
        )
        return grad_input, None, None, None, None, None, None, None


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
    if overwrite_input and (not inp.is_contiguous() or torch._debug_has_internal_overlap(inp) != 0):
        raise ValueError(
            "overwrite_input=True requires a contiguous input with no internal overlap"
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
    inputs. Instead of saving a full FP32 derivative, this function saves an
    input-typed logits buffer and per-row FP32 softmax maximum/denominator
    statistics. The returned loss is FP32.

    By default, the saved logits are a private contiguous copy and ``inp`` is
    preserved. This safe mode necessarily has both the caller's input and the
    copy live during forward. With ``overwrite_input=True``, the original
    contiguous, non-overlapping input is used as the saved buffer and is
    overwritten with its gradient during backward. Callers must not read or
    otherwise reuse that input after starting backward.

    The saved logits buffer is consumed by backward, so repeated backward
    passes on the same result are not supported.

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
        Tensor-parallel process group, or None for a single device.
    ignore_idx : int, default = -100
        Target value for ignored rows.
    is_cg_capturable : bool, default = False
        Whether the operation is CUDA graph capturable.
    overwrite_input : bool, default = False
        Reuse and overwrite ``inp`` rather than allocating a private logits
        copy. The input must be contiguous and have no internal overlap.

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

    _validate_inputs(inp, target, label_smoothing, overwrite_input)
    return CrossEntropyFunction.apply(
        inp,
        target,
        label_smoothing,
        reduce_loss,
        dist_process_group,
        ignore_idx,
        is_cg_capturable,
        overwrite_input,
    )
