# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions and scaling_aware_fp8_transpose Triton kernels."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scaling_aware_fp8_transpose_kernel(
    # input pointers
    rowwise_data_ptrs,
    rowwise_scale_inv_ptrs,
    columnwise_data_ptrs,
    columnwise_scale_inv_ptrs,
    rowwise_scale_inv_t_ptrs,
    rows_ptr,
    # sizes
    cols,
    rsi_cols,
    # strides
    stride_rowwise_data_r,
    stride_rsi_r,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    pid_group_index = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_col = tl.program_id(2)

    rows = tl.load(rows_ptr + pid_group_index)
    nbrows = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid_row >= nbrows:
        return

    row_base = tl.load(rowwise_data_ptrs + pid_group_index).to(
        tl.pointer_type(tl.uint8)
    )
    rsi_base = tl.load(rowwise_scale_inv_ptrs + pid_group_index).to(
        tl.pointer_type(tl.float32)
    )
    col_base = tl.load(columnwise_data_ptrs + pid_group_index).to(
        tl.pointer_type(tl.uint8)
    )
    csi_base = tl.load(columnwise_scale_inv_ptrs + pid_group_index).to(
        tl.pointer_type(tl.float32)
    )

    r_start = pid_row * BLOCK_SIZE
    c_start = pid_col * BLOCK_SIZE
    r_offsets = r_start + tl.arange(0, BLOCK_SIZE)
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE)
    valid_r = r_offsets < rows
    valid_c = c_offsets < cols
    data = tl.load(
        row_base + (r_offsets[:, None] * stride_rowwise_data_r + c_offsets[None, :]),
        mask=valid_r[:, None] & valid_c[None, :],
        other=0,
    )

    rsi_c_offsets = pid_col + tl.arange(0, 1)
    valid_rsi_c = rsi_c_offsets < rsi_cols
    si = tl.load(
        rsi_base + r_offsets[:, None] * stride_rsi_r + rsi_c_offsets[None, :],
        mask=valid_r[:, None] & valid_rsi_c[None, :],
        other=0.0,
    )

    # Write rowwise_scale_inv.T
    rst_base = tl.load(rowwise_scale_inv_t_ptrs + pid_group_index).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(
        rst_base + (rsi_c_offsets[:, None] * rows + r_offsets[None, :]),
        si.T,
        mask=valid_rsi_c[:, None] & valid_r[None, :],
    )

    # For the current block-row (128 rows), take the per-channel max of rowwise_scale_inv
    # This max value becomes the columnwise scaling factor for this block
    target_si = tl.max(si, axis=0)
    tl.store(csi_base + (pid_row * cols + c_offsets), target_si, mask=valid_c)

    # FP8 decode/encode
    sign = (data >> 7) & 1
    exp = (data >> 3) & 0xF
    mant = data & 0x7
    # log2_t = tl.log2(target_si)
    # log2_si = tl.log2(si + 1e-30)
    # kf = log2_t - log2_si
    # k = tl.cast(tl.floor(kf + 0.5), tl.int32)
    bits_target = tl.cast(target_si, tl.uint32, bitcast=True)
    bits_si = tl.cast(si, tl.uint32, bitcast=True)
    exp_t = ((bits_target & 0x7F800000) >> 23) - 127
    exp_s = ((bits_si & 0x7F800000) >> 23) - 127
    k_approx = exp_t[None, :] - exp_s
    k = tl.cast(k_approx, tl.int32)
    exp_new = exp - k
    exp_new = tl.where(exp_new < 1, 0, exp_new)
    new_data = (sign << 7) | (exp_new << 3) | mant
    new_data = tl.where(exp == 0, 0, new_data)

    # write columnwise_data (uint8) to [K,M] (c, r)
    tl.store(
        col_base + (c_offsets[:, None] * rows + r_offsets[None, :]),
        new_data.T,
        mask=valid_c[:, None] & valid_r[None, :],
    )


def blockwise_scaling_aware_fp8_transpose(
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    m_splits: list[int],
    block_size: int = 128,
):
    """
    Scaling-aware FP8 transpose that converts row-wise quantized FP8 tensors to a
    column-wise layout in the FP8 domain.

    The input is split along the M dimension according to ``m_splits``. For each split,
    the kernel transposes FP8 data from shape ``[m_i, cols]`` to ``[cols, m_i]`` while
    producing column-wise scaling factors at block-row granularity. The operation is
    performed without dequantizing to higher precision types.

    Parameters
    ----------
    rowwise_data : torch.Tensor
        Row-wise FP8-encoded data stored as ``uint8`` with shape
        ``[sum(m_splits), cols]``.

    rowwise_scale_inv : torch.Tensor
        Row-wise scaling factors associated with ``rowwise_data`` with shape
        ``[sum(m_splits), rsi_cols]``.

    m_splits : list[int]
        Sizes of splits along the M dimension. Each entry ``m_i`` defines the number of
        rows in one group.

    block_size : int, optional
        Tile size for the blockwise transpose and scaling-aware conversion.

    Returns
    -------
    rowwise_data_list : list[torch.Tensor]
        List of input views split by ``m_splits``, each with shape ``[m_i, cols]`` and
        dtype matching ``rowwise_data``.

    rowwise_scale_inv_t_list : list[torch.Tensor]
        List of transposed row-wise inverse scaling tensors, each with shape
        ``[nbcols, m_i]``, where ``nbcols = ceil(cols / block_size)`` and dtype matching
        ``rowwise_scale_inv``.

    columnwise_data_list : list[torch.Tensor]
        List of column-wise FP8-encoded output tensors, each with shape ``[cols, m_i]``
        and dtype matching ``rowwise_data`` (raw FP8 bits in ``uint8``).

    columnwise_scale_inv_list : list[torch.Tensor]
        List of column-wise inverse scaling tensors at block-row granularity, each with
        shape ``[nbrows_i, cols]``, where ``nbrows_i = ceil(m_i / block_size)`` and dtype
        matching ``rowwise_scale_inv``.

    """
    assert len(m_splits) > 0, "m_splits can not be zero"
    device = rowwise_data.device
    data_dtype = rowwise_data.dtype
    scale_dtype = rowwise_scale_inv.dtype

    cols = rowwise_data.shape[1]
    rsi_cols = rowwise_scale_inv.shape[1]
    # Number of block-rows (along the M dimension) for each tensor,
    # since each Mi differs, we must take the maximum among them
    nbrows_list = [(m + block_size - 1) // block_size for m in m_splits]
    nbcols = (cols + block_size - 1) // block_size

    rowwise_data_list = list(torch.split(rowwise_data, m_splits, dim=0))
    rowwise_scale_inv_list = list(torch.split(rowwise_scale_inv, m_splits, dim=0))
    rowwise_scale_inv_t_list = [
        torch.empty((nbcols, m), dtype=scale_dtype, device=device) for m in m_splits
    ]
    columnwise_data_list = [
        torch.empty((cols, m), dtype=data_dtype, device=device) for m in m_splits
    ]
    columnwise_scale_inv_list = [
        torch.empty((nb, cols), dtype=scale_dtype, device=device) for nb in nbrows_list
    ]

    rowwise_data_ptrs = torch.as_tensor([t.data_ptr() for t in rowwise_data_list]).to(
        device=device, non_blocking=True
    )
    rowwise_scale_inv_ptrs = torch.as_tensor(
        [t.data_ptr() for t in rowwise_scale_inv_list]
    ).to(device=device, non_blocking=True)
    rowwise_scale_inv_t_ptrs = torch.as_tensor(
        [t.data_ptr() for t in rowwise_scale_inv_t_list]
    ).to(device=device, non_blocking=True)
    columnwise_data_ptrs = torch.as_tensor(
        [t.data_ptr() for t in columnwise_data_list]
    ).to(device=device, non_blocking=True)
    columnwise_scale_inv_ptrs = torch.as_tensor(
        [t.data_ptr() for t in columnwise_scale_inv_list]
    ).to(device=device, non_blocking=True)

    rows_t = torch.as_tensor(m_splits, dtype=torch.int32).to(
        device=device, non_blocking=True
    )

    grid = (len(m_splits), max(nbrows_list), nbcols)
    _scaling_aware_fp8_transpose_kernel[grid](
        rowwise_data_ptrs,
        rowwise_scale_inv_ptrs,
        columnwise_data_ptrs,
        columnwise_scale_inv_ptrs,
        rowwise_scale_inv_t_ptrs,
        rows_t,
        cols,
        rsi_cols,
        rowwise_data.stride(0),
        rowwise_scale_inv.stride(0),
        BLOCK_SIZE=block_size,
    )

    return (
        rowwise_data_list,
        rowwise_scale_inv_t_list,
        columnwise_data_list,
        columnwise_scale_inv_list,
    )
