# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped (multi-tensor) NVFP4 per-token quantize Python wrapper.

Dispatches through ``tex.nvfp4_per_token_group_quantize_bulk`` -- the bulk
C++ binding owns allocation, view-slicing, and the composite K1+K2 kernel
dispatch. Requires bf16 input with K and every split_sections[i] a multiple
of 128; up to 64 splits.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_per_token import (
    RefNVFP4TensorPerToken,
    _PER_TOKEN_TILE,
)


def _validate_per_token_group_input(
    x_concat: torch.Tensor, split_sections: Sequence[int]
) -> tuple[int, int]:
    """Enforce the per-token grouped kernel's hard constraints. Returns
    ``(sum_M, K)``.
    """
    if x_concat.ndim != 2:
        raise ValueError(f"nvfp4_per_token_group_quantize expects a 2D input, got {x_concat.ndim}D")
    if not x_concat.is_contiguous():
        raise ValueError("x_concat must be contiguous (row-major)")
    if x_concat.dtype != torch.bfloat16:
        raise ValueError(f"Per-token grouped kernel is bf16-only; got dtype {x_concat.dtype}.")
    sum_M, K = x_concat.shape
    if K % _PER_TOKEN_TILE != 0:
        raise ValueError(f"Per-token grouped kernel requires K % {_PER_TOKEN_TILE} == 0; got K={K}")
    if len(split_sections) == 0:
        raise ValueError("split_sections must not be empty")
    if len(split_sections) > 64:
        raise ValueError(
            f"num_tensors must be <= 64 (kernel arg-struct cap); got {len(split_sections)}"
        )
    acc = 0
    for i, M_i in enumerate(split_sections):
        if M_i <= 0:
            raise ValueError(f"split_sections[{i}] must be > 0, got {M_i}")
        if M_i % _PER_TOKEN_TILE != 0:
            raise ValueError(f"split_sections[{i}] = {M_i} must be a multiple of {_PER_TOKEN_TILE}")
        acc += M_i
    if acc != sum_M:
        raise ValueError(f"sum(split_sections) = {acc} must equal input.size(0) = {sum_M}")
    return sum_M, K


# Default RHT sign-flip mask seed; matches the single-tensor wrapper.
_RHT_MASK_DEFAULT: int = 0xACE1


def nvfp4_per_token_group_quantize(
    x_concat: torch.Tensor,
    split_sections: Sequence[int],
    *,
    rowwise: bool = True,
    columnwise: bool = False,
    with_rht: bool = False,
    random_sign_mask_t: int = _RHT_MASK_DEFAULT,
) -> List[RefNVFP4TensorPerToken]:
    """Grouped NVFP4 per-token cast; returns N RefNVFP4TensorPerToken splits.

    Args:
        x_concat: (sum_M, K) bf16, row-major contiguous.
        split_sections: per-split row counts (each a multiple of 128).
        rowwise / columnwise: which directions to emit.
        with_rht: True -> apply a 16-pt col-wise RHT in BOTH K1 and K2;
            downstream GEMM must consume RHT-rotated weights to stay
            unbiased. Rowwise never sees RHT.
        random_sign_mask_t: low 16 bits = sign pattern shared by K1+K2.

    Raises ``ValueError`` on shape / dtype / split-size violations.
    """
    import transformer_engine_torch as tex  # type: ignore

    if not (rowwise or columnwise):
        raise ValueError("At least one of rowwise / columnwise must be True.")

    _validate_per_token_group_input(x_concat, split_sections)
    split_sections_list = [int(M_i) for M_i in split_sections]
    N = len(split_sections_list)

    # Bulk C++ call returns per-split views; s_dec_* already in fp8_e4m3fn dtype.
    (
        q_row_list,
        s_dec_row_list,
        row_amax_list,
        q_col_list,
        s_dec_col_list,
        col_amax_list,
    ) = tex.nvfp4_per_token_group_quantize_bulk(
        x_concat,
        split_sections_list,
        rowwise,
        columnwise,
        with_rht=bool(with_rht),
        random_sign_mask_t=int(random_sign_mask_t) & 0xFFFF,
    )

    outs: List[RefNVFP4TensorPerToken] = []
    for i in range(N):
        out = RefNVFP4TensorPerToken()
        if rowwise:
            out.data = q_row_list[i]
            out.scale = s_dec_row_list[i]
            out.row_amax = row_amax_list[i]
        if columnwise:
            out.columnwise_data = q_col_list[i]
            out.columnwise_scale = s_dec_col_list[i]
            out.col_amax = col_amax_list[i]
        outs.append(out)
    return outs
