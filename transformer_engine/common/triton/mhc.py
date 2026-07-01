# pylint: disable=missing-function-docstring

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""triton kernels for mHC (manifold Hyper-Connection) operations"""

import itertools
import os

import triton
import triton.language as tl


def projection_config_fwd():
    block_m = [64, 128]
    block_k = [1024]
    step_k = [32, 64]
    warps = [4]
    stages = [3, 4]

    configs = []
    for m, bk, sk, w, s in itertools.product(block_m, block_k, step_k, warps, stages):
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_M": m, "BLOCK_SIZE_K": bk, "STEP_SIZE_K": sk},
                num_warps=w,
                num_stages=s,
            )
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


def projection_config_bwd():
    block_m = [32, 128]
    block_k = [128]
    warps = [2]
    stages = [2, 3, 4]

    configs = []
    for m, bk, w, s in itertools.product(block_m, block_k, warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_K": bk}, num_warps=w, num_stages=s)
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(configs=projection_config_fwd(), key=["M", "K"], reset_to_zero=["h_ptr", "ms_ptr"])
@triton.jit
def _mhc_projection_fwd_fused(
    x_ptr,  # (M, K)
    phi_ptr,  # (N, K)
    h_ptr,  # (M, 32)
    ms_ptr,  # (M,)
    M,
    N,
    K,
    stride_xm,
    stride_xk: tl.constexpr,
    stride_phin,
    stride_phik: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_ms: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    STEP_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    precision: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_k >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk == 1)
    tl.assume(stride_phin == K)
    tl.assume(stride_phik == 1)
    tl.assume(stride_hm == 32)
    tl.assume(stride_hn == 1)
    tl.assume(stride_ms == 1)

    tl.assume(BLOCK_SIZE_M % 32 == 0)
    tl.assume(BLOCK_SIZE_K % 32 == 0)
    tl.assume(BLOCK_SIZE_N == 32)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_full = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M

    h_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    ms_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    k_base = pid_k * BLOCK_SIZE_K
    for k_start in range(0, tl.cdiv(BLOCK_SIZE_K, STEP_SIZE_K)):
        k_offs = k_base + k_start * STEP_SIZE_K + tl.arange(0, STEP_SIZE_K)
        mask_k = k_offs < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        phi_ptrs = phi_ptr + offs_n_full[:, None] * stride_phin + k_offs[None, :] * stride_phik
        phi = tl.load(
            phi_ptrs,
            mask=(offs_n_full[:, None] < N) & mask_k[None, :],
            other=0.0,
            cache_modifier=".ca",
        )  # (BLOCK_SIZE_N, BLOCK_SIZE_K)
        ms_acc += tl.sum(x * x, axis=1)
        h_acc = tl.dot(
            x, tl.trans(phi, (1, 0)), h_acc, input_precision=precision, out_dtype=tl.float32
        )

    h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_n_full[None, :] * stride_hn
    tl.atomic_add(h_ptrs, h_acc, mask=mask_m[:, None], sem="relaxed")

    offs_ms = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_ms = offs_ms < M
    offs_ms %= M
    ms_ptrs = ms_ptr + offs_ms * stride_ms
    ms = ms_acc / tl.cast(K, tl.float32)
    tl.atomic_add(ms_ptrs, ms, mask=masks_ms, sem="relaxed")


@triton.autotune(
    configs=projection_config_bwd(),
    key=["M", "K"],
)
@triton.jit
def _mhc_projection_bwd_fused(
    x_ptr,
    grad_x_ptr,  # (M, K)
    phi_ptr,  # (N, K)
    grad_h_ptr,  # (M, N)
    grad_ms_ptr,  # (M,)
    M,
    N,
    K,
    stride_xm,
    stride_xk: tl.constexpr,
    stride_grad_xm,
    stride_grad_xk: tl.constexpr,
    stride_phin,
    stride_phik: tl.constexpr,
    stride_grad_phin,
    stride_grad_phik: tl.constexpr,
    stride_grad_hm: tl.constexpr,
    stride_grad_hn: tl.constexpr,
    stride_grad_ms: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    precision: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_k >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk == 1)
    tl.assume(stride_grad_hm == 32)
    tl.assume(stride_grad_hn == 1)
    tl.assume(stride_phin == K)
    tl.assume(stride_phik == 1)
    tl.assume(stride_grad_phin == K)
    tl.assume(stride_grad_phik == 1)
    tl.assume(stride_grad_ms == 1)

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

    grad_h_ptrs = (
        grad_h_ptr + offs_m[:, None] * stride_grad_hm + offs_n_full[None, :] * stride_grad_hn
    )
    grad_h = tl.load(
        grad_h_ptrs, mask=mask_m[:, None] & (offs_n_full[None, :] < N), other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    phi_ptrs = phi_ptr + offs_n_full[:, None] * stride_phin + offs_k[None, :] * stride_phik
    offs_ms = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    grad_ms_ptrs = grad_ms_ptr + offs_ms * stride_grad_ms

    phi = tl.load(
        phi_ptrs, mask=(offs_n_full[:, None] < N) & mask_k[None, :], other=0.0
    )  # (BLOCK_SIZE_N, BLOCK_SIZE_K)
    grad_ms = tl.load(
        grad_ms_ptrs, mask=offs_ms < M, other=0.0, cache_modifier=".ca"
    )  # (BLOCK_SIZE_M,)

    grad_x = x * (grad_ms * 2 / tl.cast(K, tl.float32))[:, None]
    grad_x = tl.dot(
        grad_h, phi, acc=grad_x, input_precision=precision, out_dtype=tl.float32
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    grad_x_ptrs = grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_k[None, :] * stride_grad_xk
    grad_x = grad_x.to(x.dtype)
    tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None] & mask_k[None, :])


def scale_config():
    block_m = [128]
    warps = [4]
    stages = [1, 2, 4]

    configs = []
    for m, w, s in itertools.product(block_m, warps, stages):
        configs.append(triton.Config({"BLOCK_SIZE_M": m}, num_warps=w, num_stages=s))

    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=scale_config(),
    key=["M"],
)
@triton.jit
def _mhc_scale_fwd_fused(
    h_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    a_ptr,  # (3,)
    b_ptr,  # (2n + n^2)
    ms_ptr,  # (M,)
    out_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    M,
    n,
    stride_hm,
    stride_hn,
    stride_a,
    stride_b,
    stride_ms,
    stride_out_m,
    stride_out_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)

    tl.assume(M > 0)
    tl.assume(n == 4)
    tl.assume(stride_hm == 32)
    tl.assume(stride_hn == 1)
    tl.assume(stride_out_m == 32)
    tl.assume(stride_out_n == 1)
    tl.assume(stride_a == 1)
    tl.assume(stride_b == 1)
    tl.assume(stride_ms == 1)
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
        a_ptr + offs_a * stride_a, mask=offs_a < 3, other=0.0
    )  # a[2*n + n*n:] is filled with garbage
    a = tl.where(cols < N, a, 0.0)  # Mask out the garbage values in a

    b = tl.load(b_ptr + cols * stride_b, mask=cols < N, other=0.0)  # (BLOCK_SIZE_N,)
    ms = tl.load(ms_ptr + offs_m * stride_ms, mask=mask_m, other=0.0)  # (BLOCK_SIZE_M,)
    # In projection kernel we use split-K so we only have the accumulated ms,
    # and now we need to take sqrt on the accumulated ms to obtain the RMSNorm denominator.
    rms = tl.sqrt(ms + eps)

    h = tl.load(
        h_ptr + offs_m[:, None] * stride_hm + cols[None, :] * stride_hn,
        mask=mask_m[:, None],
        other=0.0,
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    h = a[None, :] * h
    h = tl.fma(
        h, 1.0 / rms[:, None], b[None, :]
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N), where the first 2n columns are H_pre and H_post, and the rest are H_res
    h_sigmoid_pre = tl.sigmoid(h)
    h_sigmoid_post = 2 * h_sigmoid_pre

    # Use this mask to select h[:, :2n]
    h = tl.where(cols[None, :] < n, h_sigmoid_pre, h)
    h = tl.where((cols[None, :] >= n) & (cols[None, :] < 2 * n), h_sigmoid_post, h)

    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + cols[None, :] * stride_out_n,
        h,
        mask=mask_m[:, None],
    )


@triton.autotune(
    configs=scale_config(),
    key=["M"],
    reset_to_zero=["grad_a_ptr", "grad_b_ptr"],
)
@triton.jit
def _mhc_scale_bwd_fused(
    grad_out_ptr,
    out_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    grad_h_ptr,
    h_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    grad_a_ptr,
    a_ptr,  # (3,)
    grad_b_ptr,  # (2n + n^2,)
    grad_ms_ptr,
    ms_ptr,  # (M,)
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
    stride_grad_ms,
    stride_ms,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    eps: tl.constexpr,
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
    tl.assume(stride_grad_ms == 1)
    tl.assume(stride_ms == 1)
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

    ms_offsets = offs_m
    ms_mask = mask_m
    ms = tl.load(ms_ptr + ms_offsets * stride_ms, mask=ms_mask, other=1.0)  # (BLOCK_SIZE_M,)
    rms = tl.sqrt(ms + eps)

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

    grad_a = tl.sum(h * grad_h / rms[:, None], axis=0).to(a.dtype)
    # Write grad_a[0:4].sum to grad_a_ptr[0], grad_a[4:8].sum to grad_a_ptr[1], and grad_a[8:24].sum to grad_a_ptr[2]
    tl.atomic_add(grad_a_ptr, tl.where(cols[None, :] < n, grad_a, 0.0).sum(), sem="relaxed")
    tl.atomic_add(
        grad_a_ptr + stride_grad_a,
        tl.where((cols[None, :] >= n) & (cols[None, :] < 2 * n), grad_a, 0.0).sum(),
        sem="relaxed",
    )
    tl.atomic_add(
        grad_a_ptr + 2 * stride_grad_a,
        tl.where((cols[None, :] >= 2 * n) & (cols[None, :] < 2 * n + n * n), grad_a, 0.0).sum(),
        sem="relaxed",
    )

    grad_b = tl.sum(grad_h, axis=0).to(a.dtype)
    tl.atomic_add(grad_b_ptr + cols * stride_grad_b, grad_b, mask=cols < N, sem="relaxed")

    grad_rms = (tl.sum((-grad_h * h * a[None, :]), axis=1) / (rms * rms)).to(rms.dtype)
    grad_ms = grad_rms / (2 * rms)
    tl.store(grad_ms_ptr + ms_offsets * stride_grad_ms, grad_ms, mask=ms_mask)

    grad_h = a[None, :] * grad_h / rms[:, None]
    tl.store(
        grad_h_ptr + offs_m[:, None] * stride_grad_hm + cols[None, :] * stride_grad_hn,
        grad_h,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def sinkhorn_config():
    block = [256, 1024]
    warps = [2, 8]
    stages = [2, 4]
    configs = []
    for b, w, s in itertools.product(block, warps, stages):
        configs.append(triton.Config({"BLOCK_SIZE": b}, num_warps=w, num_stages=s))
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=sinkhorn_config(),
    key=["M"],
)
@triton.jit
def _mhc_sinkhorn_fwd_fused_recompute(
    x_ptr,  # (M, n*n)
    output_ptr,  # (M, n*n)
    stride_xm,
    stride_xn,
    stride_out_m,
    stride_out_n,
    M,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    iters,
):
    pid = tl.program_id(0)

    tl.static_assert(BLOCK_SIZE % (n * n) == 0, "BLOCK_SIZE must be divisible by n*n")
    tl.assume(M > 0 and iters > 0)
    tl.assume(n == 4)

    BATCH_SIZE: tl.constexpr = BLOCK_SIZE // (n * n)

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
def _mhc_sinkhorn_bwd_fused_recompute(
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
    pid = tl.program_id(0)

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

    sbn = M * n

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
            hist_f_ptr + (iter_idx + 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(f_hist_ptrs, f, mask=mask_batch[:, None])

        # Update g: logsumexp over the row dimension (2)
        g = x + f[:, :, None]  # Broadcast f to (BATCH_SIZE, n, n)
        g_max = tl.max(g, axis=1)
        g = tl.log(tl.sum(tl.exp(g - g_max[:, None, :]), axis=1))  # logsumexp over rows
        g = log_nu - g - g_max

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx + 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(g_hist_ptrs, g, mask=mask_batch[:, None])

    # Backward pass
    grad_log_P = grad_out * P  # (BATCH_SIZE, n, n)
    zeros = tl.zeros_like(grad_log_P)
    grad_g = tl.sum(grad_log_P, axis=1)  # (BATCH_SIZE, n)
    grad_x = grad_log_P

    g_hist_ptrs = hist_g_ptr + iters * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
    g = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
    g = tl.reshape(g, (BATCH_SIZE, n))

    for iter_idx in range(iters, 0, -1):
        f_hist_ptrs = hist_f_ptr + iter_idx * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        f = tl.load(f_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        f = tl.reshape(f, (BATCH_SIZE, n))

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx - 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        g_next = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        g_next = tl.reshape(g_next, (BATCH_SIZE, n))

        term_g = -grad_g[:, None, :] * tl.exp(f[:, :, None] + x + g[:, None, :])
        grad_f = tl.sum(term_g + grad_log_P, axis=2)  # (BATCH_SIZE, n)
        # Only the last iteration's f will contribute to gradients with both grad_g1 and grad_log_P
        grad_log_P = zeros  # Zero out grad_log_P for next iterations

        g = g_next

        term_f = -grad_f[:, :, None] * tl.exp(f[:, :, None] + x + g[:, None, :])
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
def _mhc_sinkhorn_fwd_fused(
    x_ptr,  # (M, n*n)
    output_ptr,  # (M, n*n)
    hist_f_ptr,  # (iters+1, M, n)
    hist_g_ptr,  # (iters+1, M, n)
    stride_xm,
    stride_xn,
    stride_out_m,
    stride_out_n,
    M,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    iters,
):
    pid = tl.program_id(0)

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

    sbn = M * n

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
            hist_f_ptr + (iter_idx + 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        tl.store(f_hist_ptrs, f, mask=mask_batch[:, None])

        # Update g: logsumexp over the row dimension (2)
        g = x + f[:, :, None]  # Broadcast f to (BATCH_SIZE, n, n)
        g_max = tl.max(g, axis=1)
        g = tl.log(tl.sum(tl.exp(g - g_max[:, None, :]), axis=1))  # logsumexp over rows
        g = log_nu - g - g_max

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx + 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
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
def _mhc_sinkhorn_bwd_fused(
    grad_out_ptr,  # (M, n*n)
    output_ptr,  # (M, n*n)
    grad_x_ptr,  # (M, n*n)
    x_ptr,  # (M, n*n)
    hist_f_ptr,  # (iters+1, M, n)
    hist_g_ptr,  # (iters+1, M, n)
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
    pid = tl.program_id(0)

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

    sbn = M * n

    # Backward pass
    grad_log_P = grad_out * P  # (BATCH_SIZE, n, n)
    zeros = tl.zeros_like(grad_log_P)
    grad_g = tl.sum(grad_log_P, axis=1)  # (BATCH_SIZE, n)
    grad_x = grad_log_P

    g_hist_ptrs = hist_g_ptr + iters * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
    g = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
    g = tl.reshape(g, (BATCH_SIZE, n))

    for iter_idx in range(iters, 0, -1):
        f_hist_ptrs = hist_f_ptr + iter_idx * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        f = tl.load(f_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        f = tl.reshape(f, (BATCH_SIZE, n))

        g_hist_ptrs = (
            hist_g_ptr + (iter_idx - 1) * sbn + offs_batch[:, None] * n + offs_n_hist[None, :]
        )
        g_next = tl.load(g_hist_ptrs, mask=mask_batch[:, None], other=0.0)
        g_next = tl.reshape(g_next, (BATCH_SIZE, n))

        term_g = -grad_g[:, None, :] * tl.exp(f[:, :, None] + x + g[:, None, :])
        grad_f = tl.sum(term_g + grad_log_P, axis=2)  # (BATCH_SIZE, n)
        # Only the last iteration's f will contribute to gradients with both grad_g1 and grad_log_P
        grad_log_P = zeros  # Zero out grad_log_P for next iterations

        g = g_next

        term_f = -grad_f[:, :, None] * tl.exp(f[:, :, None] + x + g[:, None, :])
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


def aggregate_config():
    block_m = [1, 2, 4]
    block_c = [64, 128, 256]
    warps = [1, 2, 4]
    stages = [1, 2, 3, 4]

    configs = []
    for m, c, w, s in itertools.product(block_m, block_c, warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_C": c}, num_warps=w, num_stages=s)
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=aggregate_config(),
    key=["M", "C"],
)
@triton.jit
def _mhc_aggregate_fwd(
    x_ptr,  # # (M, C, n)
    H_pre_ptr,  # (M, n)
    output_ptr,  # (M, C)
    M,
    C,
    n: tl.constexpr,
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_c,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    output = x @ H_pre: (M, C, n) @ (M, n, 1) = (M, C, 1)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_output_m > 0 and stride_output_c == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    offs_H_pre = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_pre = tl.load(
        H_pre_ptr + offs_H_pre, mask=offs_H_pre < M * n, other=0.0, cache_modifier=".ca"
    )  # (BLOCK_SIZE_M * n)
    H_pre = H_pre.reshape(BLOCK_SIZE_M, 2, 2)
    H_pre01, H_pre23 = tl.split(H_pre)
    H_pre0, H_pre1 = tl.split(H_pre01)
    H_pre2, H_pre3 = tl.split(H_pre23)  # (BLOCK_SIZE_M, 1)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)

    x = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(x)
    x0, x1 = tl.split(x01)
    x2, x3 = tl.split(x23)  # (BLOCK_SIZE_M, BLOCK_SIZE_C)

    # x @ H_pre: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1)
    # triton doesn't support dot prod with inner dimension < 16, so we need to manually unroll the computation for n=4:
    # x @ H_pre = x[:, :, 0] * H_pre[:, 0]
    #           + x[:, :, 1] * H_pre[:, 1]
    #           + x[:, :, 2] * H_pre[:, 2]
    #           + x[:, :, 3] * H_pre[:, 3]
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)
    out_acc = tl.fma(x0, H_pre0[:, None], out_acc)
    out_acc = tl.fma(x1, H_pre1[:, None], out_acc)
    out_acc = tl.fma(x2, H_pre2[:, None], out_acc)
    out_acc = tl.fma(x3, H_pre3[:, None], out_acc)

    out = out_acc.to(x.dtype)

    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_c[None, :] * stride_output_c
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_c[None, :])


@triton.autotune(configs=aggregate_config(), key=["M", "C"], reset_to_zero=["grad_H_pre_ptr"])
@triton.jit
def _mhc_aggregate_bwd(
    grad_output_ptr,  # (M, C)
    H_pre_ptr,  # (M, n)
    grad_H_pre_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    grad_x_ptr,  # # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_c,
    stride_xm,
    stride_xCn,
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
):
    """
    Forward:
        out = x @ H_pre: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1) = (BLOCK_SIZE_M, BLOCK_SIZE_C, 1)
    Backward:
        grad_H_pre = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) = (BLOCK_SIZE_M, n, 1)
        grad_H_pre.T = grad_output.T @ x: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
            which is easier to compute since transposing grad_H_pre and grad_output is just view change
        grad_x = grad_output @ H_pre.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_c == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    grad_output_ptrs = (
        grad_output_ptr
        + offs_m[:, None] * stride_grad_output_m
        + offs_c[None, :] * stride_grad_output_c
    )
    grad_output = tl.load(
        grad_output_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)

    grad_H_pre = tl.dot(
        tl.reshape(grad_output, (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)),
        tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, n)),
        input_precision=precision,
        out_dtype=tl.float32,
    )
    grad_H_pre = tl.reshape(grad_H_pre, (BLOCK_SIZE_M * n,))  # (BLOCK_SIZE_M * n)
    offs_grad_H_pre = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_pre_ptrs = grad_H_pre_ptr + offs_grad_H_pre
    tl.atomic_add(grad_H_pre_ptrs, grad_H_pre, mask=offs_grad_H_pre < M * n, sem="relaxed")

    H_pre_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_pre = tl.load(
        H_pre_ptr + H_pre_offs, mask=H_pre_offs < M * n, other=0.0, cache_modifier=".ca"
    )  # (BLOCK_SIZE_M * n)
    H_pre = tl.reshape(H_pre, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    # grad_x = grad_output @ H_pre.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    grad_x = grad_output[:, :, None] * H_pre[:, None, :]  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    grad_x = tl.reshape(grad_x, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))

    grad_x_ptrs = grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
    tl.store(
        grad_x_ptrs,
        grad_x,
        mask=mask_m[:, None] & mask_cn[None, :],
    )


def expand_combine_config():
    block_m = [1, 2, 4]
    block_c = [128, 256]
    warps = [1, 2]
    stages = [1, 2, 3, 4]

    configs = []
    for m, c, w, s in itertools.product(block_m, block_c, warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_C": c}, num_warps=w, num_stages=s)
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=expand_combine_config(),
    key=["M", "C"],
)
@triton.jit
def _mhc_expand_combine_fwd(
    f_ptr,  # (M, C)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    output_ptr,  # # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_fm,
    stride_fc,
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_Cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    output = f @ H_post: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
           + x @ H_res: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_output_m > 0 and stride_output_Cn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

    offs_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(
        H_post_ptr + offs_H_post, mask=offs_H_post < M * n, other=0.0, cache_modifier=".ca"
    )
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    # Residual connection path: res_out = f @ H_post:
    # (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # Due to broadcasting, it's equivalent to a multiplicaiton
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    out_acc = tl.fma(f[:, :, None], H_post[:, None, :], out_acc)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(
        H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0, cache_modifier=".ca"
    )
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # Manifold connection path: manifold_out = H_res @ x:
    # (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    # triton doesn't support dot prod with inner dimension < 16, so we need to manually unroll the computation for n=4:
    # x @ H_res = x[:, :, 0] @ H_res[:, 0, :]
    #           + x[:, :, 1] @ H_res[:, 1, :]
    #           + x[:, :, 2] @ H_res[:, 2, :]
    #           + x[:, :, 3] @ H_res[:, 3, :]

    x_reshape = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(
        x_reshape
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    x0, x1 = tl.split(x01)  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    x2, x3 = tl.split(x23)  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    H_resT = tl.reshape(tl.trans(H_res, (0, 2, 1)), (BLOCK_SIZE_M, n, 2, 2))
    H_res01, H_res23 = tl.split(H_resT)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    out_acc = tl.fma(x0[:, :, None], H_res0[:, None, :], out_acc)
    out_acc = tl.fma(x1[:, :, None], H_res1[:, None, :], out_acc)
    out_acc = tl.fma(x2[:, :, None], H_res2[:, None, :], out_acc)
    out_acc = tl.fma(x3[:, :, None], H_res3[:, None, :], out_acc)

    out = out_acc.to(x.dtype)
    out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    output_ptrs = (
        output_ptr + offs_m[:, None] * stride_output_m + offs_cn[None, :] * stride_output_Cn
    )
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_cn[None, :])


@triton.autotune(
    configs=expand_combine_config(),
    key=["M", "C"],
    reset_to_zero=["grad_H_post_ptr", "grad_H_res_ptr"],
)
@triton.jit
def _mhc_expand_combine_bwd(
    grad_output_ptr,  # (M, C, n)
    f_ptr,  # (M, C)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    grad_H_post_ptr,  # (M, n)
    grad_f_ptr,  # (M, C)
    grad_H_res_ptr,  # (M, n, n)
    grad_x_ptr,  # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_Cn,
    stride_fm,
    stride_fc,
    stride_xm,
    stride_xCn,
    stride_grad_fm,
    stride_grad_fc,
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
):
    """
    Each block
    It reads
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of f, which is the output of the attention / FFN module
    - (BLOCK_SIZE_M, n) of H_post, which is applied for the transformation of the attention / FFN output
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of x, which is the skip connection's input
    - (BLOCK_SIZE_M, n*n) of H_res, which is applied for the transformation of the skip connection
    and writes
    - (BLOCK_SIZE_M, n) of grad_H_post
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of grad_f
    - (BLOCK_SIZE_M, n, n) of grad_H_res
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of grad_x

    Forward:
        out = f @ H_post + x @ H_res
    Backward:
        GEMM:
        grad_H_post = f.T @ grad_output: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
        grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
        Not GEMM:
        grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1) = (BLOCK_SIZE_M, BLOCK_SIZE_C, 1)
        grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """

    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_Cn == 1)
    tl.assume(stride_grad_fm > 0 and stride_grad_fc == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

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
        + offs_m[:, None] * stride_grad_output_m
        + offs_cn[None, :] * stride_grad_output_Cn
    )
    grad_out = tl.load(
        grad_out_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)
    grad_out = tl.reshape(
        grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # grad_H_post =  f.T @ grad_output # (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
    grad_H_post = tl.dot(
        tl.reshape(f, (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)),
        tl.reshape(grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, n)),
        input_precision=precision,
        out_dtype=tl.float32,
    )  # (BLOCK_SIZE_M, 1, n)
    grad_H_post = tl.reshape(grad_H_post, (BLOCK_SIZE_M * n,))  # (BLOCK_SIZE_M * n)
    offs_grad_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_post_ptrs = grad_H_post_ptr + offs_grad_H_post
    tl.atomic_add(grad_H_post_ptrs, grad_H_post, mask=offs_grad_H_post < M * n, sem="relaxed")

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)
    x = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
    grad_H_res = tl.dot(
        tl.trans(x, (0, 2, 1)), grad_out, input_precision=precision, out_dtype=tl.float32
    )  # (BLOCK_SIZE_M, n, n)
    grad_H_res = tl.reshape(grad_H_res, (BLOCK_SIZE_M * n * n,))  # (BLOCK_SIZE_M * n * n)
    offs_grad_H_res = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    grad_H_res_ptrs = grad_H_res_ptr + offs_grad_H_res
    tl.atomic_add(
        grad_H_res_ptrs, grad_H_res.to(tl.float32), mask=offs_grad_H_res < M * n * n, sem="relaxed"
    )

    grad_out_reshape = tl.reshape(
        grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    grad_out01, grad_out23 = tl.split(
        grad_out_reshape
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    grad_out0, grad_out1 = tl.split(
        grad_out01
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_out2, grad_out3 = tl.split(
        grad_out23
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    # grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)
    # Triton doesn't support dot prod with inner dimension < 16, so we need to hack this:
    # grad_f = grad_out[:, :, 0] @ H_post.T[:, 0, :] (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, 1)
    #        + grad_out[:, :, 1] @ H_post.T[:, 1, :]
    #        + grad_out[:, :, 2] @ H_post.T[:, 2, :]
    #        + grad_out[:, :, 3] @ H_post.T[:, 3, :]
    # where H_post.T[:, i, :] = H_post[:, :, i]
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, 2, 2))
    H_post01, H_post23 = tl.split(H_post)  # (BLOCK_SIZE_M, 2), (BLOCK_SIZE_M, 2)
    H_post0, H_post1 = tl.split(H_post01)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)
    H_post2, H_post3 = tl.split(H_post23)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)

    grad_f_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)
    # (BLOCK_SIZE_M, BLOCK_SIZE_C) * (BLOCK_SIZE_M, 1) -> (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_f_acc = tl.fma(grad_out0, H_post0[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out1, H_post1[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out2, H_post2[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out3, H_post3[:, None], grad_f_acc)
    grad_f = grad_f_acc.to(f.dtype)

    grad_f_ptrs = grad_f_ptr + offs_m[:, None] * stride_grad_fm + offs_c[None, :] * stride_grad_fc
    tl.store(grad_f_ptrs, grad_f, mask=mask_m[:, None] & mask_c[None, :])

    # grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # The inner dim is n=4 which is too small for triton, so we will manually unroll the matmul
    # grad_x = grad_out[:, :, 0] @ H_res.T[:, 0, :]
    #        + grad_out[:, :, 1] @ H_res.T[:, 1, :]
    #        + grad_out[:, :, 2] @ H_res.T[:, 2, :]
    #        + grad_out[:, :, 3] @ H_res.T[:, 3, :]
    # where H_res.T[:, i, :] = H_res[:, :, i]
    # Due to broadcasting, it's equivalent to multiplying each H_res[:, i, :].T with grad_out[:, i, :]

    H_res_reshape = tl.reshape(H_res, (BLOCK_SIZE_M, n, 2, 2))  # (BLOCK_SIZE_M, n, 2, 2)
    H_res01, H_res23 = tl.split(H_res_reshape)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    grad_x_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    grad_x_acc = tl.fma(grad_out0[:, :, None], H_res0[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out1[:, :, None], H_res1[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out2[:, :, None], H_res2[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out3[:, :, None], H_res3[:, None, :], grad_x_acc)

    grad_x = grad_x_acc.to(x.dtype)
    grad_x = tl.reshape(grad_x, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    grad_x_ptrs = grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
    tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None] & mask_cn[None, :])


@triton.autotune(
    configs=expand_combine_config(),
    key=["M", "C"],
)
@triton.jit
def _mhc_expand_combine_with_bias_fwd(
    f_ptr,  # (M, C)
    bias_ptr,  # (C,)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    output_ptr,  # # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_fm,
    stride_fc,
    stride_bias,
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_Cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    output = (f + bias[None, :, None]) @ H_post: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
           + x @ H_res: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_bias == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_output_m > 0 and stride_output_Cn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)
    bias = tl.load(bias_ptr + offs_c * stride_bias, mask=mask_c, other=0.0)  # (BLOCK_SIZE_C,)

    offs_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(
        H_post_ptr + offs_H_post, mask=offs_H_post < M * n, other=0.0, cache_modifier=".ca"
    )
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    # Residual connection path: res_out = f @ H_post + bias @ H_post:
    # (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # Due to broadcasting, it's equivalent to a multiplicaiton
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    out_acc = tl.fma(bias[None, :, None], H_post[:, None, :], out_acc)
    out_acc = tl.fma(f[:, :, None], H_post[:, None, :], out_acc)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(
        H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0, cache_modifier=".ca"
    )
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # Manifold connection path: manifold_out = H_res @ x:
    # (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    # triton doesn't support dot prod with inner dimension < 16, so we need to manually unroll the computation for n=4:
    # x @ H_res = x[:, :, 0] @ H_res[:, 0, :]
    #           + x[:, :, 1] @ H_res[:, 1, :]
    #           + x[:, :, 2] @ H_res[:, 2, :]
    #           + x[:, :, 3] @ H_res[:, 3, :]

    x_reshape = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(
        x_reshape
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    x0, x1 = tl.split(x01)  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    x2, x3 = tl.split(x23)  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    H_resT = tl.reshape(tl.trans(H_res, (0, 2, 1)), (BLOCK_SIZE_M, n, 2, 2))
    H_res01, H_res23 = tl.split(H_resT)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    out_acc = tl.fma(x0[:, :, None], H_res0[:, None, :], out_acc)
    out_acc = tl.fma(x1[:, :, None], H_res1[:, None, :], out_acc)
    out_acc = tl.fma(x2[:, :, None], H_res2[:, None, :], out_acc)
    out_acc = tl.fma(x3[:, :, None], H_res3[:, None, :], out_acc)

    out = out_acc.to(x.dtype)
    out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    output_ptrs = (
        output_ptr + offs_m[:, None] * stride_output_m + offs_cn[None, :] * stride_output_Cn
    )
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_cn[None, :])


@triton.autotune(
    configs=expand_combine_config(),
    key=["M", "C"],
    reset_to_zero=["grad_H_post_ptr", "grad_H_res_ptr", "grad_bias_ptr"],
)
@triton.jit
def _mhc_expand_combine_with_bias_bwd(
    grad_output_ptr,  # (M, C, n)
    f_ptr,  # (M, C)
    bias_ptr,  # (C,)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    grad_H_post_ptr,  # (M, n)
    grad_f_ptr,  # (M, C)
    grad_bias_ptr,  # (C,)
    grad_H_res_ptr,  # (M, n, n)
    grad_x_ptr,  # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_Cn,
    stride_fm,
    stride_fc,
    stride_bias,
    stride_xm,
    stride_xCn,
    stride_grad_fm,
    stride_grad_fc,
    stride_grad_bias,
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
):
    """
    Each block
    It reads
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of f, which is the output of the attention / FFN module
    - (BLOCK_SIZE_M, n) of H_post, which is applied for the transformation of the attention / FFN output
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of x, which is the skip connection's input
    - (BLOCK_SIZE_M, n*n) of H_res, which is applied for the transformation of the skip connection
    and writes
    - (BLOCK_SIZE_M, n) of grad_H_post
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of grad_f
    - (BLOCK_SIZE_M, n, n) of grad_H_res
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of grad_x

    Forward:
        out = f @ H_post + x @ H_res
    Backward:
        GEMM:
        grad_H_post = f.T @ grad_output: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
        grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
        Not GEMM:
        grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1) = (BLOCK_SIZE_M, BLOCK_SIZE_C, 1)
        grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """

    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_bias == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_Cn == 1)
    tl.assume(stride_grad_fm > 0 and stride_grad_fc == 1)
    tl.assume(stride_grad_bias == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

    bias = tl.load(bias_ptr + offs_c * stride_bias, mask=mask_c, other=0.0)  # (BLOCK_SIZE_C,)

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
        + offs_m[:, None] * stride_grad_output_m
        + offs_cn[None, :] * stride_grad_output_Cn
    )
    grad_out = tl.load(
        grad_out_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)
    grad_out = tl.reshape(
        grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # grad_H_post =  f.T @ grad_output # (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
    grad_H_post = tl.dot(
        tl.reshape(f, (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)),
        tl.reshape(grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, n)),
        input_precision=precision,
        out_dtype=tl.float32,
    )  # (BLOCK_SIZE_M, 1, n)
    grad_H_post = tl.dot(
        tl.broadcast_to(bias[None, None, :], (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)),
        tl.reshape(grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, n)),
        acc=grad_H_post,
        input_precision=precision,
        out_dtype=tl.float32,
    )  # (BLOCK_SIZE_M, 1, n)
    grad_H_post = tl.reshape(grad_H_post, (BLOCK_SIZE_M * n,))  # (BLOCK_SIZE_M * n)
    offs_grad_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_post_ptrs = grad_H_post_ptr + offs_grad_H_post
    tl.atomic_add(grad_H_post_ptrs, grad_H_post, mask=offs_grad_H_post < M * n, sem="relaxed")

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)
    x = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
    grad_H_res = tl.dot(
        tl.trans(x, (0, 2, 1)), grad_out, input_precision=precision, out_dtype=tl.float32
    )  # (BLOCK_SIZE_M, n, n)
    grad_H_res = tl.reshape(grad_H_res, (BLOCK_SIZE_M * n * n,))  # (BLOCK_SIZE_M * n * n)
    offs_grad_H_res = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    grad_H_res_ptrs = grad_H_res_ptr + offs_grad_H_res
    tl.atomic_add(
        grad_H_res_ptrs, grad_H_res.to(tl.float32), mask=offs_grad_H_res < M * n * n, sem="relaxed"
    )

    grad_out_reshape = tl.reshape(
        grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    grad_out01, grad_out23 = tl.split(
        grad_out_reshape
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    grad_out0, grad_out1 = tl.split(
        grad_out01
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_out2, grad_out3 = tl.split(
        grad_out23
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    # grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)
    # Triton doesn't support dot prod with inner dimension < 16, so we need to hack this:
    #        = grad_out[:, :, 0] @ H_post.T[:, 0, :] (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, 1)
    #        + grad_out[:, :, 1] @ H_post.T[:, 1, :]
    #        + grad_out[:, :, 2] @ H_post.T[:, 2, :]
    #        + grad_out[:, :, 3] @ H_post.T[:, 3, :]
    # where H_post.T[:, i, :] = H_post[:, :, i]
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, 2, 2))
    H_post01, H_post23 = tl.split(H_post)  # (BLOCK_SIZE_M, 2), (BLOCK_SIZE_M, 2)
    H_post0, H_post1 = tl.split(H_post01)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)
    H_post2, H_post3 = tl.split(H_post23)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)

    grad_f_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)
    # (BLOCK_SIZE_M, BLOCK_SIZE_C) * (BLOCK_SIZE_M, 1) -> (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_f_acc = tl.fma(grad_out0, H_post0[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out1, H_post1[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out2, H_post2[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out3, H_post3[:, None], grad_f_acc)
    grad_f = grad_f_acc.to(f.dtype)

    grad_f_ptrs = grad_f_ptr + offs_m[:, None] * stride_grad_fm + offs_c[None, :] * stride_grad_fc
    tl.store(grad_f_ptrs, grad_f, mask=mask_m[:, None] & mask_c[None, :])

    grad_bias = tl.sum(grad_f_acc, axis=0)  # (BLOCK_SIZE_C,)
    grad_bias_ptrs = grad_bias_ptr + offs_c * stride_grad_bias
    tl.atomic_add(grad_bias_ptrs, grad_bias, mask=mask_c, sem="relaxed")

    # grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # The inner dim is n=4 which is too small for triton, so we will manually unroll the matmul
    # grad_x = grad_out[:, :, 0] @ H_res.T[:, 0, :]
    #        + grad_out[:, :, 1] @ H_res.T[:, 1, :]
    #        + grad_out[:, :, 2] @ H_res.T[:, 2, :]
    #        + grad_out[:, :, 3] @ H_res.T[:, 3, :]
    # where H_res.T[:, i, :] = H_res[:, :, i]
    # Due to broadcasting, it's equivalent to multiplying each H_res[:, i, :].T with grad_out[:, i, :]

    H_res_reshape = tl.reshape(H_res, (BLOCK_SIZE_M, n, 2, 2))  # (BLOCK_SIZE_M, n, 2, 2)
    H_res01, H_res23 = tl.split(H_res_reshape)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    grad_x_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    grad_x_acc = tl.fma(grad_out0[:, :, None], H_res0[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out1[:, :, None], H_res1[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out2[:, :, None], H_res2[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out3[:, :, None], H_res3[:, None, :], grad_x_acc)

    grad_x = grad_x_acc.to(x.dtype)
    grad_x = tl.reshape(grad_x, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    grad_x_ptrs = grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
    tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None] & mask_cn[None, :])
