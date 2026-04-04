# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import itertools
import os

import triton
import triton.language as tl


def projection_config():
    block_m = [32, 64, 128]
    block_k = [32, 64, 128]
    warps = [2, 4]
    stages = [1, 2, 3, 4]

    configs = []
    for m, k, w, s in itertools.product(block_m, block_k, warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_K": k}, num_warps=w, num_stages=s)
        )
    if os.environ.get("TRITON_SKIP_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=projection_config(),
    key=["M", "K"],
)
@triton.jit
def _mhc_projection_fwd_fused(
    x_ptr,  # (M, K)
    phi_ptr,  # (N, K)
    h_ptr,  # (M, 32)
    r_ptr,  # (M,)
    M,
    N,
    K,
    stride_xm,
    stride_xk: tl.constexpr,
    stride_phin,
    stride_phik: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_r: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    out_dtype: tl.constexpr,
    precision: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Kernel for computing the matmul Y = X x W and r = sqrt(((X * X).mean(dim=1) + eps) / K) in a fused manner.
    X has shape (M, K), W has shape (N, K) which represent a col-major (K, N) and Y has shape (M, N)
    r has shape (M,)
    Note: W is column-major so we can have coalesced memory access when loading W.

    Each block computes a row of X and the full W, since N is very small here (2*n + n^2, where n is the width of Hyper-Connection and is at most 4)
    We don't need to use grouped ordering because the output MxN has small width which is the same as one block's width
    """

    pid_m = tl.program_id(axis=0)

    tl.assume(pid_m >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk == 1)
    tl.assume(stride_phin == K)
    tl.assume(stride_phik == 1)
    tl.assume(stride_hm == 32)
    tl.assume(stride_hn == 1)
    tl.assume(stride_r == 1)

    tl.assume(BLOCK_SIZE_M % 32 == 0)
    tl.assume(BLOCK_SIZE_K % 32 == 0)
    tl.assume(BLOCK_SIZE_N == 32)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_full = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M

    h_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    r_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offs < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        phi_ptrs = phi_ptr + offs_n_full[:, None] * stride_phin + k_offs[None, :] * stride_phik
        phi = tl.load(
            phi_ptrs, mask=(offs_n_full[:, None] < N) & mask_k[None, :], other=0.0
        )  # (BLOCK_SIZE_N, BLOCK_SIZE_K), loaded as column-major
        phi = tl.trans(phi, (1, 0))  # ( BLOCK_SIZE_K, BLOCK_SIZE_N)
        # RMSNorm denominator computation
        # r_acc += tl.sum((x * x).to(tl.float32), axis=1).to(r_acc.dtype)
        x_fp32 = x.to(tl.float32)
        r_acc += tl.sum(x_fp32 * x_fp32, axis=1)
        # Matrix multiplication
        h_acc = tl.dot(x, phi, h_acc, input_precision=precision, out_dtype=tl.float32)
    h = h_acc.to(out_dtype)

    h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_n_full[None, :] * stride_hn
    tl.store(h_ptrs, h, mask=mask_m[:, None])
    offs_rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_rm = offs_rm < M
    offs_rm %= M
    r_ptrs = r_ptr + offs_rm * stride_r
    r = tl.sqrt(r_acc / tl.cast(K, tl.float32) + eps).to(out_dtype)
    tl.store(r_ptrs, r, mask=masks_rm)


@triton.autotune(
    configs=projection_config(),
    key=["M", "K"],
)
@triton.jit
def _mhc_projection_bwd_fused(
    x_ptr,
    dx_ptr,  # (M, K)
    phi_ptr,  # (N, K)
    dh_ptr,  # (M, N)
    r_ptr,
    dr_ptr,  # (M,)
    M,
    N,
    K,
    stride_xm,
    stride_xk: tl.constexpr,
    stride_dxm,
    stride_dxk: tl.constexpr,
    stride_phin,
    stride_phik: tl.constexpr,
    stride_dphin,
    stride_dphik: tl.constexpr,
    stride_dhm: tl.constexpr,
    stride_dhn: tl.constexpr,
    stride_dr: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    precision: tl.constexpr,
):
    """
    This computes
    Each block handles (BLOCK_SIZE_M, N) of dY and (N, BLOCK_SIZE_K) of W^T, where N is covered by BLOCK_SIZE_N
    and also handles the element-wise multiplication part, and writes back (BLOCK_SIZE_M, BLOCK_SIZE_K) of dX each time
    """
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_k >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk == 1)
    tl.assume(stride_dhm == 32)
    tl.assume(stride_dhn == 1)
    tl.assume(stride_phin == K)
    tl.assume(stride_phik == 1)
    tl.assume(stride_dphin == K)
    tl.assume(stride_dphik == 1)

    tl.assume(BLOCK_SIZE_M % 32 == 0)
    tl.assume(BLOCK_SIZE_K % 32 == 0)
    tl.assume(BLOCK_SIZE_N == 32)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n_full = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_k = offs_k < K

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)

    dh_ptrs = dh_ptr + offs_m[:, None] * stride_dhm + offs_n_full[None, :] * stride_dhn
    dh = tl.load(
        dh_ptrs, mask=mask_m[:, None] & (offs_n_full[None, :] < N), other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    phi_ptrs = phi_ptr + offs_n_full[:, None] * stride_phin + offs_k[None, :] * stride_phik
    offs_r = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    r_ptrs = r_ptr + offs_r * stride_dr
    dr_ptrs = dr_ptr + offs_r * stride_dr

    phi = tl.load(
        phi_ptrs, mask=(offs_n_full[:, None] < N) & mask_k[None, :], other=0.0
    )  # (BLOCK_SIZE_N, BLOCK_SIZE_K)
    dx = tl.dot(
        dh, phi, input_precision=precision, out_dtype=tl.float32
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    dr = tl.load(dr_ptrs, mask=offs_r < M, other=0.0)  # (BLOCK_SIZE_M,)
    r = tl.load(r_ptrs, mask=offs_r < M, other=1.0)  # (BLOCK_SIZE_M,)
    r_scaled = dr / (tl.cast(K, tl.float32) * r)  # (BLOCK_SIZE_M,)
    dx += x * r_scaled[:, None]

    dx_ptrs = dx_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk
    dx = dx.to(x.dtype)
    tl.store(dx_ptrs, dx, mask=mask_m[:, None] & mask_k[None, :])


def elementwise_config():
    block_m = [128, 256, 512, 1024]
    warps = [2, 4]
    stages = [1, 2, 3, 4]

    configs = []
    for m, w, s in itertools.product(block_m, warps, stages):
        configs.append(triton.Config({"BLOCK_SIZE_M": m}, num_warps=w, num_stages=s))

    if os.environ.get("TRITON_SKIP_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=elementwise_config(),
    key=["M"],
)
@triton.jit
def _mhc_elementwise_fwd_fused(
    h_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    a_ptr,  # (3,)
    b_ptr,  # (2n + n^2)
    r_ptr,  # (M,)
    out_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    M,
    n,
    stride_hm,
    stride_hn,
    stride_a,
    stride_b,
    stride_r,
    stride_out_m,
    stride_out_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)  # 1D grid

    tl.assume(M > 0)
    tl.assume(n == 4)
    tl.assume(stride_hm == 32)
    tl.assume(stride_hn == 1)
    tl.assume(stride_out_m == 32)
    tl.assume(stride_out_n == 1)
    tl.assume(stride_a == 1)
    tl.assume(stride_b == 1)
    tl.assume(stride_r == 1)
    tl.assume(BLOCK_SIZE_N == 32)

    N = 2 * n + n * n

    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M

    # Expand a to BLOCK_SIZE_N length
    offs_a = tl.zeros_like(cols)
    offs_a = tl.where((cols >= n) & (cols < 2 * n), 1, offs_a)
    offs_a = tl.where((cols >= 2 * n) & (cols < 2 * n + n * n), 2, offs_a)
    # Pick a[0] from a for the first 4 columns, a[1] for the next 4 columns, and a[2] for the rest of the columns
    a = tl.load(
        a_ptr + offs_a * stride_a, mask=offs_a < N, other=0.0
    )  # a[2*n + n*n:] is filled with garbage
    a = tl.where(cols < N, a, 0.0)  # Mask out the garbage values in a

    b = tl.load(b_ptr + cols * stride_b, mask=cols < N, other=0.0)  # (BLOCK_SIZE_N,)
    r = tl.load(r_ptr + offs_m * stride_r, mask=mask_m, other=0.0)  # (BLOCK_SIZE_M,)

    h = tl.load(
        h_ptr + offs_m[:, None] * stride_hm + cols[None, :] * stride_hn,
        mask=mask_m[:, None],
        other=0.0,
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    h = a[None, :] * h
    h = tl.fma(
        h, 1.0 / r[:, None], b[None, :]
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N), where the first 2n columns are H_pre and H_post, and the rest are H_res
    h_sigmoid_pre = tl.sigmoid(h)
    h_sigmold_post = 2 * h_sigmoid_pre

    # Use this mask to select h[:, :2n]
    h = tl.where(cols[None, :] < n, h_sigmoid_pre, h)
    h = tl.where((cols[None, :] >= n) & (cols[None, :] < 2 * n), h_sigmold_post, h)

    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + cols[None, :] * stride_out_n,
        h,
        mask=mask_m[:, None],
    )


@triton.autotune(
    configs=elementwise_config(),
    key=["M"],
    reset_to_zero=["grad_a_ptr", "grad_b_ptr"],
)
@triton.jit
def _mhc_elementwise_bwd_fused(
    grad_out_ptr,
    out_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    grad_h_ptr,
    h_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    grad_a_ptr,
    a_ptr,  # (3,)
    grad_b_ptr,  # (2n + n^2,)
    grad_r_ptr,
    r_ptr,  # (M,)
    M,
    n,
    stride_grad_out_m,
    stride_grad_out_n,
    stride_out_m,
    stride_out_n,
    stride_grad_hm,
    stride_grad_hn,
    stride_hm,
    stride_hn,
    stride_grad_a,
    stride_a,
    stride_grad_b,
    stride_grad_r,
    stride_r,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)

    tl.assume(M > 0)
    tl.assume(n == 4)
    tl.assume(stride_grad_out_m == 32)
    tl.assume(stride_grad_out_n == 1)
    tl.assume(stride_out_m == 32)
    tl.assume(stride_out_n == 1)
    tl.assume(stride_grad_hm == 32)
    tl.assume(stride_grad_hn == 1)
    tl.assume(stride_hm == 32)
    tl.assume(stride_hn == 1)
    tl.assume(stride_grad_a == 1)
    tl.assume(stride_a == 1)
    tl.assume(stride_grad_b == 1)
    tl.assume(stride_grad_r == 1)
    tl.assume(stride_r == 1)
    tl.assume(BLOCK_SIZE_N == 32)

    N = 2 * n + n * n

    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = cols < N

    # Expand a to BLOCK_SIZE_N length
    offs_a = tl.zeros_like(cols)
    offs_a = tl.where((cols >= n) & (cols < 2 * n), 1, offs_a)
    offs_a = tl.where((cols >= 2 * n) & (cols < 2 * n + n * n), 2, offs_a)
    # Pick a[0] from a for the first 4 columns, a[1] for the next 4 columns, and a[2] for the rest of the columns
    a = tl.load(
        a_ptr + offs_a * stride_a, mask=offs_a < 3, other=0.0
    )  # a[2*n + n*n:] is filled with garbage
    a = tl.where(cols < N, a, 0.0)  # Mask out the garbage values in a

    r_offsets = offs_m
    r_mask = mask_m
    r = tl.load(r_ptr + r_offsets * stride_r, mask=r_mask, other=1.0)  # (BLOCK_SIZE_M,)

    grad_out = tl.load(
        grad_out_ptr + offs_m[:, None] * stride_grad_out_m + cols[None, :] * stride_grad_out_n,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    out = tl.load(
        out_ptr + offs_m[:, None] * stride_out_m + cols[None, :] * stride_out_n,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    h = tl.load(
        h_ptr + offs_m[:, None] * stride_hm + cols[None, :] * stride_hn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Gradiient of H before H_pre and H_post go through sigmoid
    grad_out_out = grad_out * out
    grad_h_pre = grad_out_out * (1 - out)
    grad_h_post = grad_out_out * 0.5 * (2 - out)
    grad_h = grad_out
    grad_h = tl.where(cols[None, :] < n, grad_h_pre, grad_h)
    grad_h = tl.where((cols[None, :] >= n) & (cols[None, :] < 2 * n), grad_h_post, grad_h)

    grad_a = tl.sum(h * grad_h / r[:, None], axis=0).to(a.dtype)
    # Write grad_a[0:4].sum to grad_a_ptr[0], grad_a[4:8].sum to grad_a_ptr[1], and grad_a[8:24].sum to grad_a_ptr[2]
    tl.atomic_add(grad_a_ptr, tl.where(cols[None, :] < n, grad_a, 0.0).sum())
    tl.atomic_add(
        grad_a_ptr + stride_grad_a,
        tl.where((cols[None, :] >= n) & (cols[None, :] < 2 * n), grad_a, 0.0).sum(),
    )
    tl.atomic_add(
        grad_a_ptr + 2 * stride_grad_a,
        tl.where((cols[None, :] >= 2 * n) & (cols[None, :] < 2 * n + n * n), grad_a, 0.0).sum(),
    )

    grad_b = tl.sum(grad_h, axis=0).to(a.dtype)
    tl.atomic_add(grad_b_ptr + cols * stride_grad_b, grad_b, mask=cols < N)

    grad_r = (tl.sum((-grad_h * h * a[None, :]), axis=1) / (r * r)).to(r.dtype)
    tl.store(grad_r_ptr + r_offsets * stride_grad_r, grad_r, mask=r_mask)

    grad_h = a[None, :] * grad_h / r[:, None]
    tl.store(
        grad_h_ptr + offs_m[:, None] * stride_grad_hm + cols[None, :] * stride_grad_hn,
        grad_h,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def sinkhorn_config():
    block = [128, 256, 512, 1024]
    warps = [2, 4, 8]
    stages = [1, 2, 3, 4]
    configs = []
    for b, w, s in itertools.product(block, warps, stages):
        configs.append(triton.Config({"BLOCK_SIZE": b}, num_warps=w, num_stages=s))
    if os.environ.get("TRITON_SKIP_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=sinkhorn_config(),
    key=["M"],
)
@triton.jit
def _mhc_sinkhorn_knopp_fwd_fused_recompute(
    x_ptr,  # (M, n*n)
    output_ptr,
    stride_xm,
    stride_xn,
    stride_out_m,
    stride_out_n,
    M,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    iters,
):
    """
    Fused Sinkhorn-Knopp algorithm to convert a matrix into a doubly stochastic matrix.
    Calculated in log space for numerical stability.

    :param X: a tensor of shape (B, T, n, n), input
    :param output_ptr: a tensor of shape (B, T, n, n), output
    :param hist_f_ptr: a tensor of shape (iters+1, B, T, n), to store f history
    :param hist_g_ptr: a tensor of shape (iters+1, B, T, n), to store g history
    :param B: batch size
    :param T: sequence length
    :param BLOCK_SIZE: size of the blocks to process
    :param iters: number of Sinkhorn iterations
    """
    pid = tl.program_id(0)  # 1D grid

    tl.static_assert(BLOCK_SIZE % (n * n) == 0, "BLOCK_SIZE must be divisible by n*n")
    tl.assume(M > 0 and iters > 0)
    tl.assume(n == 4)

    BATCH_SIZE: tl.constexpr = BLOCK_SIZE // (n * n)  # Assume there's no remainder for simplicity

    offs_batch = pid * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    offs_nn = tl.arange(0, n * n)
    mask_batch = offs_batch < M

    x_ptrs = x_ptr + offs_batch[:, None] * stride_xm + offs_nn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    x = tl.reshape(x, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)

    log_mu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    log_nu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    f = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    g = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    for _ in range(iters):
        # Update f: logsumexp over the column dimension (1)
        f = x + g[:, None, :]  # Broadcast g to (BATCH_SIZE, n, n)
        f_max = tl.max(f, axis=2)
        f = tl.log(tl.sum(tl.exp(f - f_max[:, :, None]), axis=2))  # logsumexp over columns
        f = log_mu - f - f_max

        # Update g: logsumexp over the row dimension (2)
        g = x + f[:, :, None]  # Broadcast f to (BATCH_SIZE, n, n)
        g_max = tl.max(g, axis=1)
        g = tl.log(tl.sum(tl.exp(g - g_max[:, None, :]), axis=1))  # logsumexp over rows
        g = log_nu - g - g_max

    log_P = f[:, :, None] + x + g[:, None, :]
    log_P = tl.reshape(
        log_P,
        (
            BATCH_SIZE,
            n * n,
        ),
    )
    P = tl.exp(log_P)

    output_ptrs = output_ptr + offs_batch[:, None] * stride_out_m + offs_nn[None, :] * stride_out_n
    tl.store(output_ptrs, P, mask=mask_batch[:, None])


@triton.autotune(
    configs=sinkhorn_config(),
    key=["M"],
)
@triton.jit
def _mhc_sinkhorn_knopp_bwd_fused_recompute(
    grad_out_ptr,
    output_ptr,
    grad_x_ptr,
    x_ptr,
    hist_f_ptr,
    hist_g_ptr,
    stride_grad_out_m,
    stride_grad_out_n,
    stride_out_m,
    stride_out_n,
    stride_grad_xm,
    stride_grad_xn,
    stride_xm,
    stride_xn,
    M,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    iters,
):
    """
    Backward pass for the fused Sinkhorn-Knopp algorithm with intermediate values recomputed.

    :param grad_out_ptr: pointer to the gradient of the output
    :param grad_x_ptr: pointer to the gradient of the input
    :param x_ptr: pointer to the input tensor
    :param hist_f_ptr: pointer to the tensor storing f history, (iters+1, B, T, n)
    :param hist_g_ptr: pointer to the tensor storing g history, (iters+1, B, T, n)
    :param B: batch size
    :param T: sequence length
    :param n: size of the submatrix (n x n)
    :param BLOCK_SIZE: size of the blocks to process
    :param iters: number of iterations
    """
    pid = tl.program_id(0)  # 1D grid

    tl.static_assert(BLOCK_SIZE % (n * n) == 0, "BLOCK_SIZE must be divisible by n*n")
    tl.assume(M > 0 and iters > 0)
    tl.assume(n == 4)

    BATCH_SIZE: tl.constexpr = BLOCK_SIZE // (n * n)  # Assume there's no remainder for simplicity

    offs_batch = pid * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    offs_nn = tl.arange(0, n * n)
    offs_n_hist = tl.arange(0, n)
    mask_batch = offs_batch < M

    x_ptrs = x_ptr + offs_batch[:, None] * stride_xm + offs_nn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    x = tl.reshape(x, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)
    exp_x = tl.exp(x)

    P_ptrs = output_ptr + offs_batch[:, None] * stride_out_m + offs_nn[None, :] * stride_out_n
    P = tl.load(P_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    P = tl.reshape(P, (BATCH_SIZE, n, n))

    grad_out_ptrs = (
        grad_out_ptr
        + offs_batch[:, None] * stride_grad_out_m
        + offs_nn[None, :] * stride_grad_out_n
    )
    grad_out = tl.load(grad_out_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    grad_out = tl.reshape(grad_out, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)

    BTn = M * n

    # Recompute the full history of f and g
    log_mu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    log_nu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    f = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    g = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    f_hist_ptrs = hist_f_ptr + offs_batch[:, None] * n + offs_n_hist[None, :]
    g_hist_ptrs = hist_g_ptr + offs_batch[:, None] * n + offs_n_hist[None, :]
    tl.store(f_hist_ptrs, f, mask=mask_batch[:, None])
    tl.store(g_hist_ptrs, g, mask=mask_batch[:, None])

    for iter_idx in range(iters):
        # Update f: logsumexp over the column dimension (1)
        f = x + g[:, None, :]  # Broadcast g to (BATCH_SIZE, n, n)
        f_max = tl.max(f, axis=2)
        f = tl.log(tl.sum(tl.exp(f - f_max[:, :, None]), axis=2))  # logsumexp over columns
        f = log_mu - f - f_max

        f_hist_ptrs = (
            hist_f_ptr + (iter_idx + 1) * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(f_hist_ptrs, f, mask=mask_batch[:, None])

        # Update g: logsumexp over the row dimension (2)
        g = x + f[:, :, None]  # Broadcast f to (BATCH_SIZE, n, n)
        g_max = tl.max(g, axis=1)
        g = tl.log(tl.sum(tl.exp(g - g_max[:, None, :]), axis=1))  # logsumexp over rows
        g = log_nu - g - g_max

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx + 1) * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(g_hist_ptrs, g, mask=mask_batch[:, None])

    # Backward pass
    grad_log_P = grad_out * P  # (BATCH_SIZE, n, n)
    zeros = tl.zeros_like(grad_log_P)
    grad_g = tl.sum(grad_log_P, axis=1)  # (BATCH_SIZE, n)
    grad_x = grad_log_P

    g_hist_ptrs = hist_g_ptr + iters * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
    g = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
    g = tl.reshape(g, (BATCH_SIZE, n))

    for iter_idx in range(iters, 0, -1):
        f_hist_ptrs = hist_f_ptr + iter_idx * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
        f = tl.load(f_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        f = tl.reshape(f, (BATCH_SIZE, n))

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx - 1) * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        g_next = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        g_next = tl.reshape(g_next, (BATCH_SIZE, n))

        term_g = -grad_g[:, None, :] * tl.exp(f[:, :, None] + g[:, None, :]) * exp_x
        grad_f = tl.sum(term_g + grad_log_P, axis=2)  # (BATCH_SIZE, n)
        # Only the last iteration's f will contribute to gradients with both grad_g1 and grad_log_P
        grad_log_P = zeros  # Zero out grad_log_P for next iterations

        g = g_next

        term_f = -grad_f[:, :, None] * tl.exp(f[:, :, None] + g[:, None, :]) * exp_x
        grad_g = tl.sum(term_f, axis=1)  # (BATCH_SIZE, n)

        grad_x += term_f + term_g

    grad_x_ptrs = (
        grad_x_ptr + offs_batch[:, None] * stride_grad_xm + offs_nn[None, :] * stride_grad_xn
    )
    tl.store(
        grad_x_ptrs,
        tl.reshape(
            grad_x,
            (
                BATCH_SIZE,
                n * n,
            ),
        ),
        mask=mask_batch[:, None],
    )


@triton.autotune(
    configs=sinkhorn_config(),
    key=["M"],
)
@triton.jit
def _mhc_sinkhorn_knopp_fwd_fused(
    x_ptr,  # (M, n*n)
    output_ptr,
    hist_f_ptr,
    hist_g_ptr,  # Assume this is contiguous and laid out as (iters+1, M, n)
    stride_xm,
    stride_xn,
    stride_out_m,
    stride_out_n,
    M,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    iters,
):
    """
    Fused Sinkhorn-Knopp algorithm to convert a matrix into a doubly stochastic matrix.
    Calculated in log space for numerical stability.

    :param X: a tensor of shape (B, T, n, n), input
    :param output_ptr: a tensor of shape (B, T, n, n), output
    :param hist_f_ptr: a tensor of shape (iters+1, B, T, n), to store f history
    :param hist_g_ptr: a tensor of shape (iters+1, B, T, n), to store g history
    :param B: batch size
    :param T: sequence length
    :param BLOCK_SIZE: size of the blocks to process
    :param iters: number of Sinkhorn iterations
    """
    pid = tl.program_id(0)  # 1D grid

    tl.static_assert(BLOCK_SIZE % (n * n) == 0, "BLOCK_SIZE must be divisible by n*n")
    tl.assume(M > 0 and iters > 0)
    tl.assume(n == 4)

    BATCH_SIZE: tl.constexpr = BLOCK_SIZE // (n * n)  # Assume there's no remainder for simplicity

    offs_batch = pid * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    offs_nn = tl.arange(0, n * n)
    offs_n_hist = tl.arange(0, n)
    mask_batch = offs_batch < M

    x_ptrs = x_ptr + offs_batch[:, None] * stride_xm + offs_nn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    x = tl.reshape(x, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)

    log_mu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    log_nu = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    f = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)
    g = tl.zeros((BATCH_SIZE, n), dtype=x.dtype)  # (BATCH_SIZE, n)

    BTn = M * n

    # Store the initial f and g to history
    f_hist_ptrs = hist_f_ptr + offs_batch[:, None] * n + offs_n_hist[None, :]
    g_hist_ptrs = hist_g_ptr + offs_batch[:, None] * n + offs_n_hist[None, :]
    tl.store(f_hist_ptrs, f, mask=mask_batch[:, None])
    tl.store(g_hist_ptrs, g, mask=mask_batch[:, None])

    for iter_idx in range(iters):
        # Update f: logsumexp over the column dimension (1)
        f = x + g[:, None, :]  # Broadcast g to (BATCH_SIZE, n, n)
        f_max = tl.max(f, axis=2)
        f = tl.log(tl.sum(tl.exp(f - f_max[:, :, None]), axis=2))  # logsumexp over columns
        f = log_mu - f - f_max

        f_hist_ptrs = (
            hist_f_ptr + (iter_idx + 1) * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(f_hist_ptrs, f, mask=mask_batch[:, None])

        # Update g: logsumexp over the row dimension (2)
        g = x + f[:, :, None]  # Broadcast f to (BATCH_SIZE, n, n)
        g_max = tl.max(g, axis=1)
        g = tl.log(tl.sum(tl.exp(g - g_max[:, None, :]), axis=1))  # logsumexp over rows
        g = log_nu - g - g_max

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx + 1) * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(g_hist_ptrs, g, mask=mask_batch[:, None])

    log_P = f[:, :, None] + x + g[:, None, :]
    log_P = tl.reshape(
        log_P,
        (
            BATCH_SIZE,
            n * n,
        ),
    )
    P = tl.exp(log_P)

    output_ptrs = output_ptr + offs_batch[:, None] * stride_out_m + offs_nn[None, :] * stride_out_n
    tl.store(output_ptrs, P, mask=mask_batch[:, None])


@triton.autotune(
    configs=sinkhorn_config(),
    key=["M"],
)
@triton.jit
def _mhc_sinkhorn_knopp_bwd_fused(
    grad_out_ptr,
    output_ptr,
    grad_x_ptr,
    x_ptr,
    hist_f_ptr,
    hist_g_ptr,  # Assume this is contiguous and laid out as (iters+1, M, n)
    stride_grad_out_m,
    stride_grad_out_n,
    stride_out_m,
    stride_out_n,
    stride_grad_xm,
    stride_grad_xn,
    stride_xm,
    stride_xn,
    M,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    iters,
):
    """
    Backward pass for the fused Sinkhorn-Knopp algorithm with intermediate values recomputed.

    :param grad_out_ptr: pointer to the gradient of the output
    :param grad_x_ptr: pointer to the gradient of the input
    :param x_ptr: pointer to the input tensor
    :param hist_f_ptr: pointer to the tensor storing f history, (iters+1, B, T, n)
    :param hist_g_ptr: pointer to the tensor storing g history, (iters+1, B, T, n)
    :param B: batch size
    :param T: sequence length
    :param n: size of the submatrix (n x n)
    :param BLOCK_SIZE: size of the blocks to process
    :param iters: number of iterations
    """
    pid = tl.program_id(0)  # 1D grid

    tl.static_assert(BLOCK_SIZE % (n * n) == 0, "BLOCK_SIZE must be divisible by n*n")
    tl.assume(M > 0 and iters > 0)
    tl.assume(n == 4)

    BATCH_SIZE: tl.constexpr = BLOCK_SIZE // (n * n)  # Assume there's no remainder for simplicity

    offs_batch = pid * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    offs_nn = tl.arange(0, n * n)
    offs_n_hist = tl.arange(0, n)
    mask_batch = offs_batch < M

    x_ptrs = x_ptr + offs_batch[:, None] * stride_xm + offs_nn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    x = tl.reshape(x, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)
    exp_x = tl.exp(x)

    P_ptrs = output_ptr + offs_batch[:, None] * stride_out_m + offs_nn[None, :] * stride_out_n
    P = tl.load(P_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    P = tl.reshape(P, (BATCH_SIZE, n, n))

    grad_out_ptrs = (
        grad_out_ptr
        + offs_batch[:, None] * stride_grad_out_m
        + offs_nn[None, :] * stride_grad_out_n
    )
    grad_out = tl.load(grad_out_ptrs, mask=mask_batch[:, None], other=0.0)  # (BATCH_SIZE, n*n)
    grad_out = tl.reshape(grad_out, (BATCH_SIZE, n, n))  # (BATCH_SIZE, n, n)

    BTn = M * n

    # Backward pass
    grad_log_P = grad_out * P  # (BATCH_SIZE, n, n)
    zeros = tl.zeros_like(grad_log_P)
    grad_g = tl.sum(grad_log_P, axis=1)  # (BATCH_SIZE, n)
    grad_x = grad_log_P

    g_hist_ptrs = hist_g_ptr + iters * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
    g = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
    g = tl.reshape(g, (BATCH_SIZE, n))

    for iter_idx in range(iters, 0, -1):
        f_hist_ptrs = hist_f_ptr + iter_idx * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
        f = tl.load(f_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        f = tl.reshape(f, (BATCH_SIZE, n))

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx - 1) * BTn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        g_next = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        g_next = tl.reshape(g_next, (BATCH_SIZE, n))

        term_g = -grad_g[:, None, :] * tl.exp(f[:, :, None] + g[:, None, :]) * exp_x
        grad_f = tl.sum(term_g + grad_log_P, axis=2)  # (BATCH_SIZE, n)
        # Only the last iteration's f will contribute to gradients with both grad_g1 and grad_log_P
        grad_log_P = zeros  # Zero out grad_log_P for next iterations

        g = g_next

        term_f = -grad_f[:, :, None] * tl.exp(f[:, :, None] + g[:, None, :]) * exp_x
        grad_g = tl.sum(term_f, axis=1)  # (BATCH_SIZE, n)

        grad_x += term_f + term_g

    grad_x_ptrs = (
        grad_x_ptr + offs_batch[:, None] * stride_grad_xm + offs_nn[None, :] * stride_grad_xn
    )
    tl.store(
        grad_x_ptrs,
        tl.reshape(
            grad_x,
            (
                BATCH_SIZE,
                n * n,
            ),
        ),
        mask=mask_batch[:, None],
    )


def pre_config():
    block_m = [32, 64]
    block_c = [32, 64]
    warps = [2, 4]
    stages = [1, 2, 3, 4]

    configs = []
    for m, c, w, s in itertools.product(block_m, block_c, warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_C": c}, num_warps=w, num_stages=s)
        )
    if os.environ.get("TRITON_SKIP_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=pre_config(),
    key=["M", "C"],
)
@triton.jit
def _mhc_pre_fwd(
    x_ptr,  # # (M, n, C)
    H_pre_ptr,  # (M, n)
    output_ptr,  # (M, C)
    M,
    C,
    n: tl.constexpr,
    stride_xm,
    stride_xn,
    stride_xc,
    stride_H_pre_m,
    stride_H_pre_n,
    stride_output_m,
    stride_output_c,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Each block handles BLOCK_SIZE_M rows and BLOCK_SIZE_C columns of the output
    It reads (BLOCK_SIZE_M * n) rows, BLOCK_SIZE_C columns from x, and BLOCK_SIZE_M rows, full n columns from H_pre (which is the first n columns of H), and computes
    (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) -> (BLOCK_SIZE_M, BLOCK_SIZE_C)
    However instead of matmul, we will use the equivalent operation:
    ((BLOCK_SIZE_M, 1, n).T * (BLOCK_SIZE_M, n, BLOCK_SIZE_C)).sum(dim=-2) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) -> (BLOCK_SIZE_M, BLOCK_SIZE_C)
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_xm > 0 and stride_xn > 0 and stride_xc == 1)
    tl.assume(stride_H_pre_m == n and stride_H_pre_n == 1)
    tl.assume(stride_output_m > 0 and stride_output_c == 1)

    tl.assume(BLOCK_SIZE_M % 32 == 0)
    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_n = tl.arange(0, n)
    mask_m = offs_m < M
    mask_c = offs_c < C

    x_ptrs = (
        x_ptr
        + offs_m[:, None, None] * stride_xm
        + offs_n[None, :, None] * stride_xn
        + offs_c[None, None, :] * stride_xc
    )
    x = tl.load(
        x_ptrs, mask=mask_m[:, None, None] & mask_c[None, None, :], other=0.0
    )  # (BLOCK_SIZE_M, n, BLOCK_SIZE_C)

    H_pre_ptrs = H_pre_ptr + offs_m[:, None] * stride_H_pre_m + offs_n[None, :] * stride_H_pre_n
    H_pre = tl.load(H_pre_ptrs, mask=mask_m[:, None], other=0.0)  # (BLOCK_SIZE_M, n)

    H_pre = tl.reshape(
        H_pre, (BLOCK_SIZE_M, n, 1)
    )  # (BLOCK_SIZE_M, n, 1), which is the same as (BLOCK_SIZE_M, 1, n) after transpose

    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)

    H_pre = H_pre.reshape(BLOCK_SIZE_M, 2, 2)
    H_pre01, H_pre23 = tl.split(H_pre)
    H_pre0, H_pre1 = tl.split(H_pre01)
    H_pre2, H_pre3 = tl.split(H_pre23)  # (BLOCK_SIZE_M, 1)
    xT = tl.trans(x, (0, 2, 1))
    xT = tl.reshape(xT, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    xT01, xT23 = tl.split(xT)
    xT0, xT1 = tl.split(xT01)
    xT2, xT3 = tl.split(xT23)  # (BLOCK_SIZE_M, BLOCK_SIZE_C)

    out_acc = tl.fma(H_pre0[:, None], xT0, out_acc)
    out_acc = tl.fma(H_pre1[:, None], xT1, out_acc)
    out_acc = tl.fma(H_pre2[:, None], xT2, out_acc)
    out_acc = tl.fma(H_pre3[:, None], xT3, out_acc)

    out = out_acc.to(x.dtype)

    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_c[None, :] * stride_output_c
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_c[None, :])


@triton.autotune(configs=pre_config(), key=["M", "C"], reset_to_zero=["grad_H_pre_ptr"])
@triton.jit
def _mhc_pre_bwd(
    grad_output_ptr,  # (M, C)
    H_pre_ptr,
    grad_H_pre_ptr,  # H_pre_ptr points to a slice of the parent tensor H, and grad_H_pre_ptr points to a contiguous tensor (M, n)
    x_ptr,
    grad_x_ptr,  # # (M, n, C)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_c,
    stride_H_pre_m,
    stride_H_pre_n,
    stride_grad_H_pre_m,
    stride_grad_H_pre_n,
    stride_xm,
    stride_xn,
    stride_xc,
    stride_grad_xm,
    stride_grad_xn,
    stride_grad_xc,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Each block handles BLOCK_SIZE_M rows and BLOCK_SIZE_C columns of the output
    It reads
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) from grad_output
    - (BLOCK_SIZE_M, n) from H_pre
    - (BLOCK_SIZE_M, n, BLOCK_SIZE_C) from x
     and computes
    - (BLOCK_SIZE_M, n) of grad_H_pre
    - (BLOCK_SIZE_M, n, BLOCK_SIZE_C) of grad_x

    Forward:
        out = H_pre @ x: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) -> (BLOCK_SIZE_M, BLOCK_SIZE_C)
    Backward:
        grad_H_pre = grad_output @ x.T: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n)
                   = (grad_output.T * x).sum(dim=2)
        grad_x = H_pre.T @ grad_output: (BLOCK_SIZE_M, n, 1) @ (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
               = H_pre.T * grad_output since the inner dimension is 1, they will be automatically broadcasted
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_xm > 0 and stride_xn > 0 and stride_xc == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xn > 0 and stride_grad_xc == 1)
    tl.assume(stride_H_pre_m == n and stride_H_pre_n == 1)
    tl.assume(stride_grad_H_pre_m == n and stride_grad_H_pre_n == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_c == 1)

    tl.assume(BLOCK_SIZE_M % 32 == 0)
    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_n = tl.arange(0, n)
    mask_m = offs_m < M
    mask_c = offs_c < C

    grad_output_ptrs = (
        grad_output_ptr
        + offs_m[:, None] * stride_grad_output_m
        + offs_c[None, :] * stride_grad_output_c
    )
    grad_output = tl.load(
        grad_output_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C)

    H_pre_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_pre = tl.load(
        H_pre_ptr + H_pre_offs, mask=H_pre_offs < M * n, other=0.0
    )  # (BLOCK_SIZE_M * n)
    H_pre = tl.reshape(H_pre, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    x_ptrs = (
        x_ptr
        + offs_m[:, None, None] * stride_xm
        + offs_n[None, :, None] * stride_xn
        + offs_c[None, None, :] * stride_xc
    )
    x = tl.load(
        x_ptrs, mask=mask_m[:, None, None] & mask_c[None, None, :], other=0.0
    )  # (BLOCK_SIZE_M, n, BLOCK_SIZE_C)

    H_pre = tl.reshape(
        H_pre, (BLOCK_SIZE_M, n, 1)
    )  # (BLOCK_SIZE_M, n, 1), which is the same as (BLOCK_SIZE_M, 1, n) after transpose

    grad_x_ptrs = (
        grad_x_ptr
        + offs_m[:, None, None] * stride_grad_xm
        + offs_n[None, :, None] * stride_grad_xn
        + offs_c[None, None, :] * stride_grad_xc
    )
    grad_x = H_pre * tl.reshape(grad_output, (BLOCK_SIZE_M, 1, BLOCK_SIZE_C))
    tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None, None] & mask_c[None, None, :])

    grad_H_pre = tl.sum(
        (tl.reshape(grad_output, (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)) * x).to(tl.float32), axis=2
    )  # (BLOCK_SIZE_M, n)
    grad_H_pre = tl.reshape(grad_H_pre, (BLOCK_SIZE_M * n,))  # (BLOCK_SIZE_M * n)
    offs_grad_H_pre = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_pre_ptrs = grad_H_pre_ptr + offs_grad_H_pre * stride_grad_H_pre_n
    tl.atomic_add(grad_H_pre_ptrs, grad_H_pre.to(tl.float32), mask=offs_grad_H_pre < M * n)


def post_res_config():
    block_m = [32, 64]
    block_c = [32, 64]
    warps = [2, 4]
    stages = [1, 2, 3, 4]

    configs = []
    for m, c, w, s in itertools.product(block_m, block_c, warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_C": c}, num_warps=w, num_stages=s)
        )
    if os.environ.get("TRITON_SKIP_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=post_res_config(),
    key=["M", "C"],
)
@triton.jit
def _mhc_post_res_fwd(
    f_ptr,  # (M, C)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, n, C)
    H_res_ptr,  # (M, n, n)
    output_ptr,  # # (M, n, C)
    M,
    C,
    n: tl.constexpr,
    stride_fm,
    stride_fc,
    stride_H_post_m,
    stride_H_post_n,
    stride_xm,
    stride_xn,
    stride_xc,
    stride_H_res_m,
    stride_H_res_n1,
    stride_H_res_n2,
    stride_output_m,
    stride_output_n,
    stride_output_c,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Each block handles BLOCK_SIZE_M, ln rows and BLOCK_SIZE_C columns of the output
    It reads
    - (BLOCK_SIZE_M, n, BLOCK_SIZE_C) of x, which is the skip connection's input
    - (BLOCK_SIZE_M, n*n) of H_res, which is applied for the transformation of the skip connection
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of f, which is the output of the attention / FFN module
    - (BLOCK_SIZE_M, n) of H_post, which is applied for the transformation of the attention / FFN output
    and writes
    - (BLOCK_SIZE_M, n, BLOCK_SIZE_C) of the output, which is the post-residual output merged with the skip connection

    output = H_post.T @ f: (BLOCK_SIZE_M, n, 1) @ (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)  = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
           + H_res @ x: (BLOCK_SIZE_M, n, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_xm > 0 and stride_xn > 0 and stride_xc == 1)
    tl.assume(stride_H_post_m == n and stride_H_post_n == 1)
    tl.assume(stride_H_res_m == n * n and stride_H_res_n1 == n and stride_H_res_n2 == 1)
    tl.assume(stride_output_m > 0 and stride_output_n > 0 and stride_output_c == 1)

    tl.assume(BLOCK_SIZE_M % 32 == 0)
    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_n = tl.arange(0, n)
    mask_m = offs_m < M
    mask_c = offs_c < C

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

    H_post_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(H_post_ptr + H_post_offs, mask=H_post_offs < M * n, other=0.0)
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0)
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)

    x_ptrs = (
        x_ptr
        + offs_m[:, None, None] * stride_xm
        + offs_n[None, :, None] * stride_xn
        + offs_c[None, None, :] * stride_xc
    )
    x = tl.load(x_ptrs, mask=mask_m[:, None, None] & mask_c[None, None, :], other=0.0)

    # Manifold connection path: manifold_out = H_res @ x:
    # (BLOCK_SIZE_M, n, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # Since n=4 it's more efficient to manually unroll the matmul instead of tl.dot
    # H_res @ x = H_res[:, :, 0] @ x[:, 0, :]
    #           + H_res[:, :, 1] @ x[:, 1, :]
    #           + H_res[:, :, 2] @ x[:, 2, :]
    #           + H_res[:, :, 3] @ x[:, 3, :]
    manifold_out_acc = tl.zeros((BLOCK_SIZE_M, n, BLOCK_SIZE_C), dtype=tl.float32)

    xT = tl.trans(x, (0, 2, 1))  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    xT = tl.reshape(xT, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(xT)  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    x0, x1 = tl.split(x01)  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    x2, x3 = tl.split(x23)  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, 2, 2))
    H_res01, H_res23 = tl.split(H_res)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    manifold_out_acc = tl.fma(H_res0[:, :, None], x0[:, None, :], manifold_out_acc)
    manifold_out_acc = tl.fma(H_res1[:, :, None], x1[:, None, :], manifold_out_acc)
    manifold_out_acc = tl.fma(H_res2[:, :, None], x2[:, None, :], manifold_out_acc)
    manifold_out_acc = tl.fma(H_res3[:, :, None], x3[:, None, :], manifold_out_acc)

    manifold_out = manifold_out_acc.to(x.dtype)

    # Residual connection path: res_out = H_post.T @ f
    # (BLOCK_SIZE_M, n, 1) @ (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)  = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # Due to broadcasting, it's equivalent to a multiplicaiton
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n, 1))
    f = tl.reshape(f, (BLOCK_SIZE_M, 1, BLOCK_SIZE_C))
    res_out = H_post * f

    out = manifold_out + res_out

    output_ptrs = (
        output_ptr
        + offs_m[:, None, None] * stride_output_m
        + offs_n[None, :, None] * stride_output_n
        + offs_c[None, None, :] * stride_output_c
    )
    tl.store(output_ptrs, out, mask=mask_m[:, None, None] & mask_c[None, None, :])


@triton.autotune(
    configs=post_res_config(), key=["M", "C"], reset_to_zero=["grad_H_post_ptr", "grad_H_res_ptr"]
)
@triton.jit
def _mhc_post_res_bwd(
    grad_output_ptr,  # (M, n, C)
    f_ptr,  # (M, C)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, n, C)
    H_res_ptr,  # (M, n, n)
    grad_H_post_ptr,  # (M, n)
    grad_f_ptr,  # (M, C)
    grad_H_res_ptr,  # (M, n, n)
    grad_x_ptr,  # (M, n, C)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_n,
    stride_grad_output_c,
    stride_fm,
    stride_fc,
    stride_H_post_m,
    stride_H_post_n,
    stride_xm,
    stride_xn,
    stride_xc,
    stride_H_res_m,
    stride_H_res_n1,
    stride_H_res_n2,
    stride_grad_H_post_m,
    stride_grad_H_post_n,
    stride_grad_fm,
    stride_grad_fc,
    stride_grad_H_res_m,
    stride_grad_H_res_n1,
    stride_grad_H_res_n2,
    stride_grad_xm,
    stride_grad_xn,
    stride_grad_xc,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Each block
    It reads
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of f, which is the output of the attention / FFN module
    - (BLOCK_SIZE_M, n) of H_post, which is applied for the transformation of the attention / FFN output
    - (BLOCK_SIZE_M, n, BLOCK_SIZE_C) of x, which is the skip connection's input
    - (BLOCK_SIZE_M, n*n) of H_res, which is applied for the transformation of the skip connection
    and writes
    - (BLOCK_SIZE_M, n) of grad_H_post
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of grad_f
    - (BLOCK_SIZE_M, n, n) of grad_H_res
    - (BLOCK_SIZE_M, n, BLOCK_SIZE_C) of grad_x

    Forward:
        out = H_post @ f + H_res @ x
    Backward:
        grad_H_post = grad_output @ f.T: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) = (BLOCK_SIZE_M, n, 1)
        grad_f = H_post.T @ grad_output: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)
        grad_H_res = grad_output @ x.T: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
        grad_x = H_res.T @ grad_output: (BLOCK_SIZE_M, n, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    """

    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_xm > 0 and stride_xn > 0 and stride_xc == 1)
    tl.assume(stride_H_post_m == n and stride_H_post_n == 1)
    tl.assume(stride_H_res_m == n * n and stride_H_res_n1 == n and stride_H_res_n2 == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_n > 0 and stride_grad_output_c == 1)
    tl.assume(stride_grad_fm > 0 and stride_grad_fc == 1)
    tl.assume(stride_grad_H_post_m == n and stride_grad_H_post_n == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xn > 0 and stride_grad_xc == 1)
    tl.assume(
        stride_grad_H_res_m == n * n and stride_grad_H_res_n1 == n and stride_grad_H_res_n2 == 1
    )

    tl.assume(BLOCK_SIZE_M % 32 == 0)
    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_n = tl.arange(0, n)
    mask_m = offs_m < M
    mask_c = offs_c < C

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

    H_post_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(H_post_ptr + H_post_offs, mask=H_post_offs < M * n, other=0.0)
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(
        H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0
    )  # (BLOCK_SIZE_M, n, n)
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)

    grad_out_ptrs = (
        grad_output_ptr
        + offs_m[:, None, None] * stride_grad_output_m
        + offs_n[None, :, None] * stride_grad_output_n
        + offs_c[None, None, :] * stride_grad_output_c
    )
    grad_out = tl.load(
        grad_out_ptrs, mask=mask_m[:, None, None] & mask_c[None, None, :], other=0.0
    )  # (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    grad_out_T = tl.trans(
        grad_out, (0, 2, 1)
    )  # (BLOCK_SIZE_M, n, BLOCK_SIZE_C) -> (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    grad_out_T = tl.reshape(
        grad_out_T, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    grad_out_T01, grad_out_T23 = tl.split(
        grad_out_T
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    grad_out_T0, grad_out_T1 = tl.split(
        grad_out_T01
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_out_T2, grad_out_T3 = tl.split(
        grad_out_T23
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    # grad_f = H_post.T @ grad_output: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)
    # .       = H_post.T[:, :, 0] @ grad_out[:, 0, :] (BLOCK_SIZE_M, 1, 1) @ (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)
    #        + H_post.T[:, :, 1] @ grad_out[:, 1, :]
    #        + H_post.T[:, :, 2] @ grad_out[:, 2, :]
    #        + H_post.T[:, :, 3] @ grad_out[:, 3, :]
    # where grad_out[:, i, :] = grad_out.T[:, :, i]
    H_post_T = tl.reshape(H_post, (BLOCK_SIZE_M, 2, 2))
    H_post_T01, H_post_T23 = tl.split(H_post_T)  # (BLOCK_SIZE_M, 2), (BLOCK_SIZE_M, 2)
    H_post_T0, H_post_T1 = tl.split(H_post_T01)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)
    H_post_T2, H_post_T3 = tl.split(H_post_T23)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)

    grad_f_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)
    grad_f_acc = tl.fma(H_post_T0[:, None], grad_out_T0, grad_f_acc)
    grad_f_acc = tl.fma(H_post_T1[:, None], grad_out_T1, grad_f_acc)
    grad_f_acc = tl.fma(H_post_T2[:, None], grad_out_T2, grad_f_acc)
    grad_f_acc = tl.fma(H_post_T3[:, None], grad_out_T3, grad_f_acc)
    grad_f = grad_f_acc.to(f.dtype)

    grad_f_ptrs = grad_f_ptr + offs_m[:, None] * stride_grad_fm + offs_c[None, :] * stride_grad_fc
    tl.store(grad_f_ptrs, grad_f, mask=mask_m[:, None] & mask_c[None, :])

    # grad_H_post = grad_output @ f.T
    # grad_H_post.T = f @ grad_output.T: (BLOCK_SIZE_M, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    # grad_H_post.T[:, :, i] = f @ grad_output.T[:, :, i]: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) = (BLOCK_SIZE_M, 1)
    #                        = (f * grad_output.T[:, :, i].T).sum(dim=1): (BLOCK_SIZE_M, BLOCK_SIZE_C) * (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1)
    grad_H_post0 = tl.sum((f[:, :, None] * grad_out_T0[:, :, None]).to(tl.float32), axis=1)
    grad_H_post1 = tl.sum((f[:, :, None] * grad_out_T1[:, :, None]).to(tl.float32), axis=1)
    grad_H_post2 = tl.sum((f[:, :, None] * grad_out_T2[:, :, None]).to(tl.float32), axis=1)
    grad_H_post3 = tl.sum((f[:, :, None] * grad_out_T3[:, :, None]).to(tl.float32), axis=1)

    grad_H_post01 = tl.join(grad_H_post0, grad_H_post1)
    grad_H_post23 = tl.join(grad_H_post2, grad_H_post3)
    grad_H_post = tl.reshape(
        tl.join(grad_H_post01, grad_H_post23), (BLOCK_SIZE_M * n)
    )  # (BLOCK_SIZE_M, n)

    offs_grad_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_post_ptrs = grad_H_post_ptr + offs_grad_H_post
    tl.atomic_add(grad_H_post_ptrs, grad_H_post.to(tl.float32), mask=offs_grad_H_post < M * n)

    x_ptrs = (
        x_ptr
        + offs_m[:, None, None] * stride_xm
        + offs_n[None, :, None] * stride_xn
        + offs_c[None, None, :] * stride_xc
    )
    x = tl.load(
        x_ptrs, mask=mask_m[:, None, None] & mask_c[None, None, :], other=0.0
    )  # (BLOCK_SIZE_M, n, BLOCK_SIZE_C)

    # grad_H_res = grad_output @ x.T: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
    # For n=4, tl.dot is less efficient than manually unrolling the matmul
    # grad_H_res.T = x @ grad_out.T: (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    # grad_H_res.T[:, :, i] = x[:, :, :] @ grad_out.T[:, :, i]: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) = (BLOCK_SIZE_M, n, 1)
    #                       = (x[:, :, :] * grad_out.T[:, :, i].T).sum(dim=2): ((BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)).sum(2)

    grad_H_res_T0 = tl.sum(
        (x * grad_out_T0[:, None, :]).to(tl.float32), axis=2
    )  # (BLOCK_SIZE_M, n)
    grad_H_res_T1 = tl.sum(
        (x * grad_out_T1[:, None, :]).to(tl.float32), axis=2
    )  # (BLOCK_SIZE_M, n)
    grad_H_res_T2 = tl.sum(
        (x * grad_out_T2[:, None, :]).to(tl.float32), axis=2
    )  # (BLOCK_SIZE_M, n)
    grad_H_res_T3 = tl.sum(
        (x * grad_out_T3[:, None, :]).to(tl.float32), axis=2
    )  # (BLOCK_SIZE_M, n)

    grad_H_res_T01 = tl.join(grad_H_res_T0, grad_H_res_T1)  # (BLOCK_SIZE_M, n, 2)
    grad_H_res_T23 = tl.join(grad_H_res_T2, grad_H_res_T3)  # (BLOCK_SIZE_M, n, 2)
    grad_H_res_T = tl.reshape(
        tl.join(grad_H_res_T01, grad_H_res_T23), (BLOCK_SIZE_M, n, n)
    )  # (BLOCK_SIZE_M, n, 4)
    grad_H_res = tl.trans(grad_H_res_T, (0, 2, 1))  # (BLOCK_SIZE_M, n, n)
    grad_H_res = tl.reshape(grad_H_res, (BLOCK_SIZE_M * n * n,))  # (BLOCK_SIZE_M * n * n,)

    offs_grad_H_res = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    grad_H_res_ptrs = grad_H_res_ptr + offs_grad_H_res
    tl.atomic_add(grad_H_res_ptrs, grad_H_res.to(tl.float32), mask=offs_grad_H_res < M * n * n)

    # grad_x = H_res.T @ grad_output: (BLOCK_SIZE_M, n, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # The inner dim is n=4 which is too small for triton, so we will manually unroll the matmul
    # grad_x = H_res[:, 0, :].T @ grad_out[:, 0, :]
    #        + H_res[:, 1, :].T @ grad_out[:, 1, :]
    #        + H_res[:, 2, :].T @ grad_out[:, 2, :]
    #        + H_res[:, 3, :].T @ grad_out[:, 3, :]
    # Due to broadcasting, it's equivalent to multiplying each H_res[:, i, :].T with grad_out[:, i, :]

    H_res_T = tl.trans(H_res, (0, 2, 1))  # (BLOCK_SIZE_M, n, n) -> (BLOCK_SIZE_M, n, n)
    H_res_T = tl.reshape(H_res_T, (BLOCK_SIZE_M, n, 2, 2))  # (BLOCK_SIZE_M, n, 2, 2)
    H_res_T01, H_res_T23 = tl.split(H_res_T)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res_T0, H_res_T1 = tl.split(H_res_T01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res_T2, H_res_T3 = tl.split(H_res_T23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    grad_x_acc = tl.zeros((BLOCK_SIZE_M, n, BLOCK_SIZE_C), dtype=tl.float32)
    grad_x_acc = tl.fma(H_res_T0[:, :, None], grad_out_T0[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(H_res_T1[:, :, None], grad_out_T1[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(H_res_T2[:, :, None], grad_out_T2[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(H_res_T3[:, :, None], grad_out_T3[:, None, :], grad_x_acc)

    grad_x = grad_x_acc.to(x.dtype)

    grad_x_ptrs = (
        grad_x_ptr
        + offs_m[:, None, None] * stride_grad_xm
        + offs_n[None, :, None] * stride_grad_xn
        + offs_c[None, None, :] * stride_grad_xc
    )
    tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None, None] & mask_c[None, None, :])
