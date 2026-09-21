# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Batch-invariant BF16 GEMM forward path.

The general :func:`general_gemm` path delegates kernel selection to cuBLASLt, whose
heuristic depends on the full problem shape. That makes a given input row's output
depend on how many *other* rows are present in the batch. This module provides an
opt-in forward path with a reduction order that only depends on the tile
coordinates, so a row's result is bitwise stable across batch composition.

Supported: BF16, ``Y = X @ W.T``, contiguous 2-D operands, no bias.
Unsupported combinations raise instead of silently falling back.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = ["batch_invariant_gemm", "is_supported"]

# Tile geometry. These are fixed on purpose: making them depend on M would
# reintroduce the batch dependence this path exists to remove.
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64


@triton.jit
def _bi_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[m, n] = sum_k A[m, k] * B[n, k], accumulated in a single BLOCK_K loop."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Fixed-trip sequential reduction: the order is a function of K and BLOCK_K
    # only, so it cannot vary with M or with the tile's position in the batch.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def is_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Whether ``batch_invariant_gemm`` can run this configuration."""
    try:
        _check(a, b, out)
    except ValueError:
        return False
    return True


def _check(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]) -> None:
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(
            f"batch_invariant_gemm supports bfloat16 inputs only, got {a.dtype} and {b.dtype}."
        )
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"batch_invariant_gemm requires 2-D operands, got {a.dim()}-D and {b.dim()}-D."
        )
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("batch_invariant_gemm requires contiguous operands.")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"K mismatch: A has {a.shape[1]} columns, B has {b.shape[1]}."
        )
    if out is not None and (out.dim() != 2 or not out.is_contiguous()):
        raise ValueError("batch_invariant_gemm requires a contiguous 2-D out tensor.")


def batch_invariant_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``Y = A @ B.T`` with a batch-composition-independent reduction order.

    Parameters
    ----------
    a : torch.Tensor
        BF16 activation, shape ``[M, K]``, contiguous. Row ``m`` of ``a`` always
        produces the same output bits for a given ``b``, independent of ``M`` and of
        the row's position.
    b : torch.Tensor
        BF16 weight, shape ``[N, K]``, contiguous.
    out : torch.Tensor, optional
        BF16 destination of shape ``[M, N]``. A fresh tensor is allocated when absent.

    Returns
    -------
    torch.Tensor
        The ``[M, N]`` result. Same layout contract as :func:`general_gemm`'s output.
    """
    _check(a, b, out)
    m, k = a.shape
    n = b.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    elif out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}.")
    if m == 0 or n == 0:
        return out

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    _bi_gemm_kernel[grid](
        a, b, out,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out
