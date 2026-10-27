# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for Cross Entropy Triton kernels."""

from typing import Optional, Union
import torch
import torch.distributed as dist

import triton

from transformer_engine.common.triton.cross_entropy import (
    cross_entropy_forward_kernel,
    cross_entropy_tp_pre_kernel,
    cross_entropy_tp_post_kernel,
    cross_entropy_backward_kernel,
    cross_entropy_backward_mxfp8_kernel,
)

# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2

# Launch configuration of the single-rank forward kernel and of the backward kernel, which run
# one program per row. Small blocks with few warps keep more rows in flight per SM: one
# 1024-thread program per row reaches well under half of the HBM bandwidth at a 129k
# vocabulary. Only the order in which the softmax statistics are accumulated changes. The
# forward uses one block size with and without the argmax so that the loss does not depend on
# return_argmax.
_FWD_BLOCK_SIZE = 4096
_FWD_NUM_WARPS = 4
_BWD_BLOCK_SIZE = 4096
_BWD_NUM_WARPS = 8
# Tile width and warps of the backward that writes an MXFP8 gradient (32-row tiles).
_MXFP8_BWD_BLOCK_COLS = 256
_MXFP8_BWD_NUM_WARPS = 4


def cross_entropy_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_process_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
    overwrite_input: bool,
    compute_argmax: bool = False,
):
    """Forward implementation that saves the input and compact softmax statistics.

    With ``compute_argmax``, also returns the index of each row's first maximum in the
    full vocabulary (otherwise None).
    """

    B, SQ, V = _input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    target = target.contiguous()
    saved_input = (
        _input
        if overwrite_input
        else torch.empty(_input.shape, dtype=_input.dtype, device=_input.device)
    )
    loss_1d = torch.empty(n_rows, dtype=torch.float32, device=_input.device)
    stats = torch.empty((n_rows, 2), dtype=torch.float32, device=_input.device)
    n_non_ignore = torch.zeros(1, dtype=torch.int64, device=_input.device) if reduce_loss else None
    argmax = (
        torch.empty(n_rows, dtype=torch.int64, device=_input.device) if compute_argmax else None
    )

    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if world_size == 1:
        cross_entropy_forward_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride_0=_input.stride(0),
            X_stride_1=_input.stride(1),
            X_stride_2=_input.stride(2),
            saved_input_ptr=saved_input,
            Y_ptr=target,
            loss_ptr=loss_1d,
            stats_ptr=stats,
            argmax_ptr=argmax if argmax is not None else stats,
            n_non_ignore=n_non_ignore if n_non_ignore is not None else stats,
            n_cols=V,
            n_rows_1=SQ,
            ignore_idx=ignore_idx,
            label_smoothing=label_smoothing,
            COUNT_NON_IGNORE=reduce_loss,
            COPY_INPUT=not overwrite_input,
            COMPUTE_ARGMAX=compute_argmax,
            BLOCK_SIZE=min(_FWD_BLOCK_SIZE, BLOCK_SIZE),
            num_warps=_FWD_NUM_WARPS,
        )
    else:
        # Per row and rank: max, sum of exponentials, target logit, logit sum, and with
        # compute_argmax the local argmax (as FP32 bits) and its value (NaN if the shard has one)
        local_data_size = 6 if compute_argmax else 4
        local_data = torch.empty(
            (n_rows, local_data_size), dtype=torch.float32, device=_input.device
        )
        cross_entropy_tp_pre_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride_0=_input.stride(0),
            X_stride_1=_input.stride(1),
            X_stride_2=_input.stride(2),
            saved_input_ptr=saved_input,
            Y_ptr=target,
            local_data_ptr=local_data,
            n_non_ignore=n_non_ignore if n_non_ignore is not None else stats,
            rank=rank,
            n_cols=V,
            n_rows_1=SQ,
            ignore_idx=ignore_idx,
            COUNT_NON_IGNORE=reduce_loss,
            COMPUTE_X_SUM=label_smoothing > 0,
            COPY_INPUT=not overwrite_input,
            COMPUTE_ARGMAX=compute_argmax,
            LOCAL_DATA_SIZE=local_data_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
        gathered_data = torch.empty(
            (world_size * n_rows, local_data_size), dtype=torch.float32, device=_input.device
        )
        dist.all_gather_into_tensor(gathered_data, local_data, group=dist_process_group)
        cross_entropy_tp_post_kernel[(n_rows,)](
            gathered_data_ptr=gathered_data,
            Y_ptr=target,
            loss_ptr=loss_1d,
            stats_ptr=stats,
            argmax_ptr=argmax if argmax is not None else stats,
            world_size=world_size,
            n_rows=n_rows,
            n_cols=V,
            ignore_idx=ignore_idx,
            label_smoothing=label_smoothing,
            COMPUTE_ARGMAX=compute_argmax,
            LOCAL_DATA_SIZE=local_data_size,
            num_warps=1,
        )

    loss = loss_1d.reshape(B, SQ)
    if reduce_loss:
        n_non_ignore.clamp_(min=1)
        loss = loss_1d.sum() / n_non_ignore
    if argmax is not None:
        argmax = argmax.reshape(B, SQ)

    return loss, saved_input, stats, target, n_non_ignore, rank, world_size, argmax


def _mxfp8_backward_block_cols(
    saved_input: torch.Tensor, world_size: int, grad_quantizer
) -> Optional[int]:
    """Tile width of the fused MXFP8 backward, or None if it cannot write this gradient."""
    # pylint: disable=import-outside-toplevel
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex

    if not isinstance(grad_quantizer, MXFP8Quantizer):
        return None
    if grad_quantizer.dtype not in (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2):
        return None
    # 2D blocks and GEMM-swizzled scales are left to the quantizer.
    if grad_quantizer.with_2d_quantization or grad_quantizer.optimize_for_gemm:
        return None
    if not (grad_quantizer.rowwise_usage or grad_quantizer.columnwise_usage):
        return None
    if world_size != 1 or saved_input.dtype != torch.bfloat16 or not saved_input.is_contiguous():
        return None
    B, SQ, V = saved_input.shape
    # Unpadded scale-inverse layouts: 128 rows x 4 blocks of the row-wise scales and
    # 4 x 128 columns of the column-wise scales.
    if (B * SQ) % 128 != 0 or V % 128 != 0:
        return None
    return _MXFP8_BWD_BLOCK_COLS if V % _MXFP8_BWD_BLOCK_COLS == 0 else 128


def cross_entropy_backward(
    saved_input: torch.Tensor,
    stats: torch.Tensor,
    target: torch.Tensor,
    n_non_ignore: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    rank: int,
    world_size: int,
    ignore_idx: int,
    grad_quantizer=None,
):
    """Reconstruct the derivative in FP32.

    Without ``grad_quantizer``, the derivative overwrites the saved input buffer, which is
    returned. With it, the derivative is returned quantized by ``grad_quantizer``: a BF16 input
    with an MXFP8 quantizer is written as MXFP8 directly by one kernel, and anything else is
    computed in the saved input buffer and then quantized.
    """

    B, SQ, V = saved_input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    grad_output = grad_output.contiguous()

    if grad_quantizer is not None:
        block_cols = _mxfp8_backward_block_cols(saved_input, world_size, grad_quantizer)
        if block_cols is not None:
            import transformer_engine_torch as tex  # pylint: disable=import-outside-toplevel

            grad_input = grad_quantizer.make_empty(
                saved_input.shape, dtype=saved_input.dtype, device=saved_input.device
            )
            rowwise, columnwise = (
                grad_quantizer.rowwise_usage,
                grad_quantizer.columnwise_usage,
            )
            rowwise_scale = grad_input._rowwise_scale_inv if rowwise else None
            columnwise_scale = grad_input._columnwise_scale_inv if columnwise else None
            cross_entropy_backward_mxfp8_kernel[(n_rows // 32, V // block_cols)](
                saved_input_ptr=saved_input,
                Y_ptr=target,
                stats_ptr=stats,
                n_non_ignore_ptr=n_non_ignore if n_non_ignore is not None else stats,
                grad_output_ptr=grad_output,
                grad_output_stride=0 if reduce_loss else 1,
                rowwise_data_ptr=grad_input._rowwise_data if rowwise else saved_input,
                rowwise_scale_ptr=rowwise_scale if rowwise else saved_input,
                rowwise_scale_stride=rowwise_scale.shape[-1] if rowwise else 0,
                columnwise_data_ptr=(grad_input._columnwise_data if columnwise else saved_input),
                columnwise_scale_ptr=columnwise_scale if columnwise else saved_input,
                columnwise_scale_stride=columnwise_scale.shape[-1] if columnwise else 0,
                n_cols=V,
                ignore_idx=ignore_idx,
                reduce_loss=reduce_loss,
                label_smoothing=label_smoothing,
                ROWWISE=rowwise,
                COLUMNWISE=columnwise,
                FP8_E5M2=grad_quantizer.dtype == tex.DType.kFloat8E5M2,
                BLOCK_COLS=block_cols,
                num_warps=_MXFP8_BWD_NUM_WARPS,
            )
            return grad_input

    cross_entropy_backward_kernel[(n_rows,)](
        saved_input_ptr=saved_input,
        Y_ptr=target,
        stats_ptr=stats,
        n_non_ignore_ptr=n_non_ignore if n_non_ignore is not None else stats,
        grad_output_ptr=grad_output,
        grad_output_stride=0 if reduce_loss else 1,
        rank=rank,
        world_size=world_size,
        n_cols=V,
        ignore_idx=ignore_idx,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=min(_BWD_BLOCK_SIZE, BLOCK_SIZE),
        num_warps=_BWD_NUM_WARPS,
    )
    if grad_quantizer is not None:
        return grad_quantizer(saved_input)
    return saved_input
