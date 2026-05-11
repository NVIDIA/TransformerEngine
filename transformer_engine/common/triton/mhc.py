# pylint: disable=missing-function-docstring

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""triton kernels for mHC (manifold Hyper-Connection) operations"""

import itertools
import os
import torch

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

MAX_GRID_DIM_Y = 65535  # Maximum grid dimension in Y direction for current CUDA architectures


def align_to(x, alignment):
    return ((x + alignment - 1) // alignment) * alignment


def get_device_sms():
    """
    Get the number of SMs of the current device. This is used to determine the grid size for launching Triton kernels.
    """
    device_id = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(device_id)
    sm_count = device_props.multi_processor_count
    return sm_count


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
    return configs


def projection_prune_fwd(configs, named_args, **kwargs):
    USE_SPLIT_K = named_args.get("USE_SPLIT_K", kwargs.get("USE_SPLIT_K", None))

    if USE_SPLIT_K:
        pruned_configs = configs
    else:
        K = named_args.get("K", kwargs.get("K", None))
        block_m = [8, 16, 64, 128]
        block_k = align_to(K, 32)
        step_k = [128, 256]
        warps = [2, 4, 8]
        stages = [2, 3, 4]

        pruned_configs = []
        for bm, sk, w, s in itertools.product(block_m, step_k, warps, stages):
            pruned_configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": bm,
                        "BLOCK_SIZE_K": block_k,
                        "STEP_SIZE_K": sk,
                    },
                    num_warps=w,
                    num_stages=s,
                )
            )
    # Triton will skip calling prune function if the autotune returns only one config, which breaks the determinism override here
    # So we need to apply NVTE_DISABLE_TRITON_AUTOTUNING in the pruner instead
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        pruned_configs = pruned_configs[:1]
    return pruned_configs


@triton.autotune(
    configs=projection_config_fwd(),
    key=["M", "K", "USE_TMA", "USE_SPLIT_K"],
    reset_to_zero=["h_ptr", "ms_ptr"],
    prune_configs_by={"early_config_prune": projection_prune_fwd},
)
@triton.jit
def _mhc_projection_fwd_fused(
    x_ptr,  # (M, K)
    phi_ptr,  # (N, K)
    h_ptr,  # (M, 32)
    ms_ptr,  # (M,)
    norm_weight_ptr,  # (K,)
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
    stride_norm_weight: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    STEP_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    precision: tl.constexpr,
    USE_SPLIT_K: tl.constexpr,
    USE_TMA: tl.constexpr,  # If True, load x and phi via TMA tensor descriptors (Hopper+ only). Falls back to pointer-arith tl.load otherwise.
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
    tl.assume(stride_norm_weight == 1)

    tl.assume(BLOCK_SIZE_M % 8 == 0)
    tl.assume(BLOCK_SIZE_K % 32 == 0)
    tl.assume(BLOCK_SIZE_N == 32)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_full = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M

    h_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    ms_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    if USE_TMA:
        x_desc = tl.make_tensor_descriptor(
            x_ptr,
            shape=[M, K],
            strides=[stride_xm, 1],
            block_shape=[BLOCK_SIZE_M, STEP_SIZE_K],
        )
        phi_desc = tl.make_tensor_descriptor(
            phi_ptr,
            shape=[N, K],
            strides=[stride_phin, 1],
            block_shape=[BLOCK_SIZE_N, STEP_SIZE_K],
        )

    k_base = pid_k * BLOCK_SIZE_K
    for k_start in range(0, tl.cdiv(BLOCK_SIZE_K, STEP_SIZE_K)):
        k_off = k_base + k_start * STEP_SIZE_K
        k_offs = k_off + tl.arange(0, STEP_SIZE_K)
        mask_k = k_offs < K

        if USE_TMA:
            x = tl.load_tensor_descriptor(x_desc, [pid_m * BLOCK_SIZE_M, k_off])
            phi = tl.load_tensor_descriptor(phi_desc, [0, k_off])
        else:
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

        ms_acc += tl.sum(x.to(tl.float32) * x.to(tl.float32), axis=1)

        # Currently triton has a bug where for small block size, tl.dot(x, phi.T) will use SMEM to transpose the matrix
        # instead of emit a ldmatrix instruction with `.trans` modifier, which leads bank conflicts and performance regression
        # See https://github.com/triton-lang/triton/issues/6569#issuecomment-2841739082
        h_acc = tl.dot(
            x.to(phi.dtype),
            tl.trans(phi, (1, 0)),
            h_acc,
            input_precision=precision,
            out_dtype=tl.float32,
        )

    h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_n_full[None, :] * stride_hn
    if USE_SPLIT_K:
        tl.atomic_add(h_ptrs, h_acc, mask=mask_m[:, None], sem="relaxed")
    else:
        tl.store(h_ptrs, h_acc, mask=mask_m[:, None])

    offs_ms = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_ms = offs_ms < M
    offs_ms %= M
    ms_ptrs = ms_ptr + offs_ms * stride_ms
    ms = ms_acc / tl.cast(K, tl.float32)
    if USE_SPLIT_K:
        tl.atomic_add(ms_ptrs, ms, mask=masks_ms, sem="relaxed")
    else:
        tl.store(ms_ptrs, ms, mask=masks_ms)


def projection_config_bwd_dx():
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


@triton.autotune(
    configs=projection_config_bwd_dx(),
    key=["M", "K"],
    # When FUSE_GRAD_X_ACC=True the kernel does a read-modify-write on grad_x_ptr; without
    # restore_value the autotune timing trials accumulate onto the buffer and corrupt it.
    restore_value=["grad_x_ptr"],
)
@triton.jit
def _mhc_projection_bwd_fused_dx(
    x_ptr,
    grad_x_ptr,  # (M, K)
    phi_ptr,  # (N, K)
    norm_weight_ptr,  # (K,)
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
    stride_norm_weight: tl.constexpr,
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
    FUSE_GRAD_X_ACC: tl.constexpr,
    HAS_NORM_WEIGHT: tl.constexpr,
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
    tl.assume(stride_norm_weight == 1)

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

    if HAS_NORM_WEIGHT:
        norm_weight_ptrs = norm_weight_ptr + offs_k * stride_norm_weight
        norm_weight = tl.load(norm_weight_ptrs, mask=mask_k, other=0.0, cache_modifier=".ca").to(
            phi.dtype
        )  # (BLOCK_SIZE_K,)
        phi = phi.to(tl.float32) * norm_weight.to(tl.float32)[None, :]

    grad_ms = tl.load(
        grad_ms_ptrs, mask=offs_ms < M, other=0.0, cache_modifier=".ca"
    )  # (BLOCK_SIZE_M,)

    grad_x = x * (grad_ms * 2 / tl.cast(K, tl.float32))[:, None]
    grad_x = tl.dot(
        grad_h.to(phi.dtype), phi, acc=grad_x, input_precision=precision, out_dtype=tl.float32
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    grad_x_ptrs = grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_k[None, :] * stride_grad_xk
    grad_x = grad_x.to(x.dtype)
    if FUSE_GRAD_X_ACC:  # If fused gradient accumulation is enabled, the buffer is always fp32
        grad_x_acc = tl.load(grad_x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        grad_x = grad_x.to(tl.float32) + grad_x_acc
    tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None] & mask_k[None, :])


def projection_config_bwd_dphi():
    block_m = [512, 1024, 2048]
    step_m = [32]
    block_k = [128, 256]
    warps = [2]
    stages = [2, 3, 4]

    configs = []
    for bm, sm, bk, w, s in itertools.product(block_m, step_m, block_k, warps, stages):
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_M": bm, "STEP_SIZE_M": sm, "BLOCK_SIZE_K": bk},
                num_warps=w,
                num_stages=s,
            )
        )
    return configs


def projection_prune_bwd_dphi(configs, named_args, **kwargs):
    USE_SPLIT_M = named_args.get("USE_SPLIT_M", kwargs.get("USE_SPLIT_M", None))

    if USE_SPLIT_M:
        pruned_configs = configs
    else:
        M = named_args.get("M", kwargs.get("M", None))
        block_k = [128]
        block_m = align_to(M, 128)
        step_m = [32]
        warps = [4]
        stages = [6, 7, 8]

        pruned_configs = []
        for bk, sm, w, s in itertools.product(block_k, step_m, warps, stages):
            pruned_configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": block_m,
                        "STEP_SIZE_M": sm,
                        "BLOCK_SIZE_K": bk,
                    },
                    num_warps=w,
                    num_stages=s,
                )
            )
    # Triton will skip calling prune function if the autotune returns only one config, which breaks the determinism override here
    # So we need to apply NVTE_DISABLE_TRITON_AUTOTUNING in the pruner instead
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        pruned_configs = pruned_configs[:1]
    return pruned_configs


@triton.autotune(
    configs=projection_config_bwd_dphi(),
    key=["M", "K", "USE_SPLIT_M"],
    reset_to_zero=["grad_phi_ptr", "grad_norm_weight_ptr"],
    prune_configs_by={"early_config_prune": projection_prune_bwd_dphi},
)
@triton.jit
def _mhc_projection_bwd_fused_dphi(
    x_ptr,  # (M, K)
    grad_H_ptr,  # (M, 32)
    phi_ptr,  # (N, K), N=24 in our case since n = 4
    norm_weight_ptr,  # (K,)
    grad_phi_ptr,  # (N, K), N=24 in our case since n = 4
    grad_norm_weight_ptr,  # (K,)
    M,
    N,
    K,
    stride_xm,
    stride_xk: tl.constexpr,
    stride_grad_Hm: tl.constexpr,
    stride_grad_Hn: tl.constexpr,
    stride_phin,
    stride_phik: tl.constexpr,
    stride_norm_weight: tl.constexpr,
    stride_grad_phin,
    stride_grad_phik: tl.constexpr,
    stride_grad_norm_weight: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    STEP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    precision: tl.constexpr,
    USE_SPLIT_M: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    tl.assume(pid_k >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_xk == 1)
    tl.assume(stride_grad_Hm == 32)
    tl.assume(stride_grad_Hn == 1)
    tl.assume(stride_phin == K)
    tl.assume(stride_phik == 1)
    tl.assume(stride_grad_phin == K)
    tl.assume(stride_grad_phin == stride_phin)
    tl.assume(stride_grad_phik == 1)
    tl.assume(stride_grad_norm_weight == 1)
    tl.assume(stride_norm_weight == 1)

    tl.assume(BLOCK_SIZE_M % 128 == 0)
    tl.assume(BLOCK_SIZE_K % 64 == 0)
    tl.assume(BLOCK_SIZE_N == 32)
    tl.assume(STEP_SIZE_M % 32 == 0)

    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_k = offs_k < K
    offs_n_full = tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n_full < N

    grad_psi_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    m_start = pid_m * BLOCK_SIZE_M
    m_end = tl.minimum(m_start + BLOCK_SIZE_M, M)
    for m_idx in range(0, tl.cdiv(m_end - m_start, STEP_SIZE_M)):
        offs_m = m_start + m_idx * STEP_SIZE_M + tl.arange(0, STEP_SIZE_M)
        mask_m = offs_m < M
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )  # (STEP_SIZE_M, BLOCK_SIZE_K)
        grad_H_ptrs = (
            grad_H_ptr + offs_m[:, None] * stride_grad_Hm + offs_n_full[None, :] * stride_grad_Hn
        )
        grad_H = tl.load(
            grad_H_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
        )  # (STEP_SIZE_M, BLOCK_SIZE_N)

        grad_psi_acc = tl.dot(
            tl.trans(grad_H, (1, 0)),
            x.to(grad_H.dtype),
            acc=grad_psi_acc,
            out_dtype=tl.float32,
            input_precision=precision,
        )

    phi_ptrs = phi_ptr + offs_n_full[:, None] * stride_phin + offs_k[None, :] * stride_phik
    phi = tl.load(
        phi_ptrs, mask=(offs_n_full[:, None] < N) & mask_k[None, :], other=0.0
    )  # (BLOCK_SIZE_N, BLOCK_SIZE_K)
    norm_weight_ptrs = norm_weight_ptr + offs_k * stride_norm_weight
    norm_weight = tl.load(
        norm_weight_ptrs, mask=mask_k, other=0.0, cache_modifier=".cg"
    )  # (BLOCK_SIZE_K,)
    phi = phi.to(tl.float32)
    norm_weight = norm_weight.to(tl.float32)

    # Keep grad_psi in SRAM and get grad_phi & grad_norm_weight
    grad_phi = grad_psi_acc * norm_weight[None, :].to(grad_psi_acc.dtype)  # (32, BLOCK_SIZE_K)
    grad_norm_weight = tl.sum(grad_psi_acc * phi.to(grad_psi_acc.dtype), axis=0)  # (BLOCK_SIZE_K,)

    grad_phi_ptrs = (
        grad_phi_ptr + offs_n_full[:, None] * stride_grad_phin + offs_k[None, :] * stride_grad_phik
    )
    grad_norm_weight_ptrs = grad_norm_weight_ptr + offs_k * stride_grad_norm_weight

    if USE_SPLIT_M:
        tl.atomic_add(
            grad_phi_ptrs,
            grad_phi,
            mask=(offs_n_full[:, None] < N) & mask_k[None, :],
            sem="relaxed",
        )
        tl.atomic_add(grad_norm_weight_ptrs, grad_norm_weight, mask=mask_k, sem="relaxed")
    else:
        tl.store(
            grad_phi_ptrs, grad_phi.to(phi.dtype), mask=(offs_n_full[:, None] < N) & mask_k[None, :]
        )
        tl.store(grad_norm_weight_ptrs, grad_norm_weight.to(norm_weight.dtype), mask=mask_k)


def scale_config():
    warps = [4]
    stages = [1, 2, 4]

    configs = []
    for w, s in itertools.product(warps, stages):
        configs.append(triton.Config({}, num_warps=w, num_stages=s))

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
    grad_H_ptr,
    H_ptr,  # (M, 2n + n^2), which is padded to (M, 32) in the last dimension
    grad_a_ptr,
    a_ptr,  # (3,)
    grad_b_ptr,  # (2n + n^2,)
    grad_ms_ptr,
    ms_ptr,  # (M,)
    ws_grad_a_ptr,  # Temporary workspace for a with shape (NUM_SMS, 3), or None if DETERMINISTIC is False
    ws_grad_b_ptr,  # Temporary workspace for b with shape (NUM_SMS, 32), or None if DETERMINISTIC is False
    M,
    n: tl.constexpr,
    stride_grad_out_m,
    stride_grad_out_n,
    stride_out_m,
    stride_out_n,
    stride_grad_Hm,
    stride_grad_Hn,
    stride_Hm,
    stride_Hn,
    stride_grad_a,
    stride_a,
    stride_grad_b,
    stride_grad_ms,
    stride_ms,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    eps: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
):
    pid = tl.program_id(0)

    tl.assume(M > 0)
    tl.assume(n == 4)
    tl.assume(stride_grad_out_m == 32)
    tl.assume(stride_grad_out_n == 1)
    tl.assume(stride_out_m == 32)
    tl.assume(stride_out_n == 1)
    tl.assume(stride_grad_Hm == 32)
    tl.assume(stride_grad_Hn == 1)
    tl.assume(stride_Hm == 32)
    tl.assume(stride_Hn == 1)
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
    H = tl.load(
        H_ptr + offs_m[:, None] * stride_Hm + cols[None, :] * stride_Hn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Gradiient of H before H_pre and H_post go through sigmoid
    grad_out_out = grad_out * out
    grad_H_pre = grad_out_out * (1 - out)
    grad_H_post = grad_out_out * 0.5 * (2 - out)
    grad_H = grad_out
    grad_H = tl.where(cols[None, :] < n, grad_H_pre, grad_H)
    grad_H = tl.where((cols[None, :] >= n) & (cols[None, :] < 2 * n), grad_H_post, grad_H)
    grad_H = grad_H.to(tl.float32)
    H = H.to(tl.float32)

    grad_a = tl.sum(H * grad_H / rms[:, None], axis=0)
    grad_b = tl.sum(grad_H, axis=0)

    grad_rms = (tl.sum((-grad_H * H * a[None, :]), axis=1) / (rms * rms)).to(rms.dtype)
    grad_ms = grad_rms / (2 * rms)
    tl.store(grad_ms_ptr + ms_offsets * stride_grad_ms, grad_ms, mask=ms_mask)

    grad_H = a[None, :] * grad_H / rms[:, None]
    tl.store(
        grad_H_ptr + offs_m[:, None] * stride_grad_Hm + cols[None, :] * stride_grad_Hn,
        grad_H,
        mask=mask_m[:, None] & mask_n[None, :],
    )

    if DETERMINISTIC:
        ws_grad_a_ptrs = ws_grad_a_ptr + pid * 4
        # Write grad_a[0:4].sum to grad_a_ptr[0], grad_a[4:8].sum to grad_a_ptr[1], and grad_a[8:24].sum to grad_a_ptr[2]
        tl.store(ws_grad_a_ptrs, tl.where(cols[None, :] < n, grad_a, 0.0).sum())
        tl.store(
            ws_grad_a_ptrs + 1,
            tl.where((cols[None, :] >= n) & (cols[None, :] < 2 * n), grad_a, 0.0).sum(),
        )
        tl.store(
            ws_grad_a_ptrs + 2,
            tl.where((cols[None, :] >= 2 * n) & (cols[None, :] < 2 * n + n * n), grad_a, 0.0).sum(),
        )
        ws_grad_b_ptrs = ws_grad_b_ptr + pid * 32 + cols
        tl.store(ws_grad_b_ptrs, grad_b, mask=cols < N)
    else:
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

        tl.atomic_add(grad_b_ptr + cols * stride_grad_b, grad_b, mask=cols < N, sem="relaxed")

@gluon.jit
def _mhc_sinkhorn_fwd(
    x_ptr,
    output_ptr,
    M,
    n: gl.constexpr,
    iters: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    nn: gl.constexpr = n * n

    # GMEM-coalescing layout: each thread reads/writes a contiguous 4 fp32 along the inner n.
    # block shape on axis 0 is 8 * NUM_WARPS, so the compiler tiles a (BATCH_SIZE, n, n) load
    # into BATCH_SIZE / (8 * NUM_WARPS) coalesced LDGs per thread.
    load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 4],
        threads_per_warp=[8, 4, 1],
        warps_per_cta=[NUM_WARPS, 1, 1],
        order=[2, 1, 0],
    )

    # Compute layout: each thread owns one full (n, n) inner matrix.
    # Reductions on axes 1, 2 become register-local (no shuffles).
    compute_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, n, n],
        threads_per_warp=[32, 1, 1],
        warps_per_cta=[NUM_WARPS, 1, 1],
        order=[2, 1, 0],
    )

    # 1D offset layouts derived from load_layout (for the coalesced GMEM pointers).
    layout_M:  gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, load_layout))
    layout_n1: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, load_layout))
    layout_n2: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, load_layout))

    offs_b  = pid * BATCH_SIZE + gl.arange(0, BATCH_SIZE, layout=layout_M)
    offs_n1 = gl.arange(0, n, layout=layout_n1)
    offs_n2 = gl.arange(0, n, layout=layout_n2)
    mask_b  = offs_b < M

    ptrs = (
        x_ptr
        + offs_b[:, None, None]  * nn
        + offs_n1[None, :, None] * n
        + offs_n2[None, None, :]
    )

    # GMEM -> registers (coalesced) -> convert to per-thread (n, n) layout.
    # `convert_layout` routes through SMEM automatically when the two layouts disagree.
    x = gl.load(ptrs, mask=mask_b[:, None, None], other=0.0)
    x = gl.convert_layout(x, compute_layout)

    layout_f: gl.constexpr = gl.SliceLayout(2, compute_layout)
    layout_g: gl.constexpr = gl.SliceLayout(1, compute_layout)

    f = gl.full([BATCH_SIZE, n], 0.0, gl.float32, layout=layout_f)
    g = gl.full([BATCH_SIZE, n], 0.0, gl.float32, layout=layout_g)

    for _ in range(iters):
        z = x + g[:, None, :]
        z_max = gl.max(z, axis=2)
        f = -gl.log(gl.sum(gl.exp(z - z_max[:, :, None]), axis=2)) - z_max

        z = x + f[:, :, None]
        z_max = gl.max(z, axis=1)
        g = -gl.log(gl.sum(gl.exp(z - z_max[:, None, :]), axis=1)) - z_max

    P = gl.exp(f[:, :, None] + x + g[:, None, :])

    # Convert back to the coalesced layout for a coalesced STG.
    P_out = gl.convert_layout(P, load_layout)
    out_ptrs = (
        output_ptr
        + offs_b[:, None, None]  * nn
        + offs_n1[None, :, None] * n
        + offs_n2[None, None, :]
    )
    gl.store(out_ptrs, P_out, mask=mask_b[:, None, None])

@gluon.jit
def _mhc_sinkhorn_bwd(
    grad_out_ptr,
    output_ptr,
    x_ptr,
    grad_x_ptr,
    M,
    n: gl.constexpr,
    iters: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    nn: gl.constexpr = n * n

    # GMEM-coalescing layout: each thread reads/writes a contiguous 4 fp32 along the inner n.
    # block shape on axis 0 is 8 * NUM_WARPS, so the compiler tiles a (BATCH_SIZE, n, n) load
    # into BATCH_SIZE / (8 * NUM_WARPS) coalesced LDGs per thread.
    load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 4],
        threads_per_warp=[8, 4, 1],
        warps_per_cta=[NUM_WARPS, 1, 1],
        order=[2, 1, 0],
    )

    # Compute layout: each thread owns one full (n, n) inner matrix.
    # Reductions on axes 1, 2 become register-local (no shuffles).
    compute_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, n, n],
        threads_per_warp=[32, 1, 1],
        warps_per_cta=[NUM_WARPS, 1, 1],
        order=[2, 1, 0],
    )

    # 1D offset layouts derived from load_layout (for the coalesced GMEM pointers).
    layout_M:  gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, load_layout))
    layout_n1: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, load_layout))
    layout_n2: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, load_layout))

    layout_f: gl.constexpr = gl.SliceLayout(2, compute_layout)
    layout_g: gl.constexpr = gl.SliceLayout(1, compute_layout)

    # SMEM-resident f/g history. The leading "iter" dim is buffered — .index(i) peels
    # it to give a rank-2 (BATCH_SIZE, n) view that f / g can store into and load from.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=4, per_phase=2, max_phase=8, order=[1, 0]
    )
    hist_f = gl.allocate_shared_memory(gl.float32, [iters + 1, BATCH_SIZE, n], smem_layout)
    hist_g = gl.allocate_shared_memory(gl.float32, [iters + 1, BATCH_SIZE, n], smem_layout)

    offs_b  = pid * BATCH_SIZE + gl.arange(0, BATCH_SIZE, layout=layout_M)
    offs_n1 = gl.arange(0, n, layout=layout_n1)
    offs_n2 = gl.arange(0, n, layout=layout_n2)
    mask_b  = offs_b < M

    base_offsets = (
        offs_b[:, None, None]  * nn
        + offs_n1[None, :, None] * n
        + offs_n2[None, None, :]
    )

    # GMEM -> registers (coalesced) -> convert to per-thread (n, n) layout.
    # `convert_layout` routes through SMEM automatically when the two layouts disagree.
    x_loaded = gl.load(x_ptr + base_offsets, mask=mask_b[:, None, None], other=0.0)
    x = gl.convert_layout(x_loaded, compute_layout)

    f = gl.full([BATCH_SIZE, n], 0.0, gl.float32, layout=layout_f)
    g = gl.full([BATCH_SIZE, n], 0.0, gl.float32, layout=layout_g)
    hist_f.index(0).store(f)
    hist_g.index(0).store(g)

    # Forward recompute, recording each iteration's f and g into shared memory.
    for i in range(iters):
        z = x + g[:, None, :]
        z_max = gl.max(z, axis=2)
        f = -gl.log(gl.sum(gl.exp(z - z_max[:, :, None]), axis=2)) - z_max
        hist_f.index(i + 1).store(f)

        z = x + f[:, :, None]
        z_max = gl.max(z, axis=1)
        g = -gl.log(gl.sum(gl.exp(z - z_max[:, None, :]), axis=1)) - z_max
        hist_g.index(i + 1).store(g)

    P = gl.exp(f[:, :, None] + x + g[:, None, :])

    # Backward pass starts
    grad_out_loaded = gl.load(grad_out_ptr + base_offsets, mask=mask_b[:, None, None], other=0.0)
    grad_out = gl.convert_layout(grad_out_loaded, compute_layout)

    grad_log_P = grad_out * P
    grad_g = gl.sum(grad_log_P, axis=1)
    grad_x = grad_log_P

    g = hist_g.index(iters).load(layout_g)

    for k in range(iters):
        iter_idx = iters - k  # iterates iters .. 1
        f = hist_f.index(iter_idx    ).load(layout_f)
        g_next = hist_g.index(iter_idx - 1).load(layout_g)

        term_g = -grad_g[:, None, :] * gl.exp(f[:, :, None] + x + g[:, None, :])
        if k == 0:
            grad_f = gl.sum(term_g + grad_log_P, axis=2)
        else:
            grad_f = gl.sum(term_g, axis=2)

        g = g_next
        term_f = -grad_f[:, :, None] * gl.exp(f[:, :, None] + x + g[:, None, :])
        grad_g = gl.sum(term_f, axis=1)
        grad_x = grad_x + term_f + term_g

    # Convert grad_x back to load_layout for coalesced GMEM store.
    grad_x_out = gl.convert_layout(grad_x, load_layout)
    gl.store(grad_x_ptr + base_offsets, grad_x_out, mask=mask_b[:, None, None])



def aggregate_config_fwd():
    block_m = [1, 2, 4]
    block_c = [128, 256]
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


def aggregate_prune_fwd(configs, named_args, **kwargs):
    M = named_args.get("M", kwargs.get("M", None))

    pruned_configs = list(
        filter(
            lambda config: triton.cdiv(M, config.kwargs["BLOCK_SIZE_M"]) <= MAX_GRID_DIM_Y, configs
        )
    )
    return pruned_configs


@triton.autotune(
    configs=aggregate_config_fwd(),
    key=["M", "C"],
    prune_configs_by={"early_config_prune": aggregate_prune_fwd},
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


def aggregate_config_bwd():
    block_m = [1, 2, 4]
    block_c = [64, 128, 256]
    step_c = [32, 64]
    warps = [1, 2, 4]
    stages = [1, 2, 3, 4]

    configs = []
    for bm, bc, sc, w, s in itertools.product(block_m, block_c, step_c, warps, stages):
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_C": bc, "STEP_SIZE_C": sc},
                num_warps=w,
                num_stages=s,
            )
        )
    return configs


def aggregate_prune_bwd(configs, named_args, **kwargs):
    USE_SPLIT_K = named_args.get("USE_SPLIT_K", kwargs.get("USE_SPLIT_K", None))
    M = named_args.get("M", kwargs.get("M", None))

    if USE_SPLIT_K:
        pruned_configs = configs
    else:
        C = named_args.get("C", kwargs.get("C", None))
        block_m = [4]
        block_c = align_to(C, 64)
        step_c = [64]
        warps = [1]
        stages = [2, 3, 4]

        pruned_configs = []
        for bm, sc, w, s in itertools.product(block_m, step_c, warps, stages):
            pruned_configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": bm,
                        "BLOCK_SIZE_C": block_c,
                        "STEP_SIZE_C": sc,
                    },
                    num_warps=w,
                    num_stages=s,
                )
            )

    pruned_configs = list(
        filter(
            lambda config: triton.cdiv(M, config.kwargs["BLOCK_SIZE_M"]) <= MAX_GRID_DIM_Y,
            pruned_configs,
        )
    )

    # Triton will skip calling prune function if the autotune returns only one config, which breaks the determinism override here
    # So we need to apply NVTE_DISABLE_TRITON_AUTOTUNING in the pruner instead
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        pruned_configs = pruned_configs[:1]
    return pruned_configs


@triton.autotune(
    configs=aggregate_config_bwd(),
    key=["M", "C", "USE_SPLIT_K"],
    reset_to_zero=["grad_H_pre_ptr"],
    # When FUSE_GRAD_X_ACC=True the kernel does a read-modify-write on grad_x_ptr; without
    # restore_value the autotune timing trials accumulate onto the buffer and corrupt it.
    restore_value=["grad_x_ptr"],
    prune_configs_by={"early_config_prune": aggregate_prune_bwd},
)
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
    STEP_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
    FUSE_GRAD_X_ACC: tl.constexpr,
    USE_SPLIT_K: tl.constexpr,
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
    tl.assume(STEP_SIZE_C % 32 == 0)
    tl.assume(BLOCK_SIZE_C % STEP_SIZE_C == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M

    offs_c_start = pid_c * BLOCK_SIZE_C
    offs_cn_start = pid_c * BLOCK_SIZE_C * n

    H_pre_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_pre = tl.load(
        H_pre_ptr + H_pre_offs, mask=H_pre_offs < M * n, other=0.0, cache_modifier=".ca"
    )  # (BLOCK_SIZE_M * n)
    H_pre = tl.reshape(H_pre, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    grad_H_pre_acc = tl.zeros((BLOCK_SIZE_M, 1, n), dtype=tl.float32)
    for i in tl.range(0, BLOCK_SIZE_C, STEP_SIZE_C, loop_unroll_factor=2):
        offs_c = offs_c_start + i + tl.arange(0, STEP_SIZE_C)
        offs_cn = offs_cn_start + i * n + tl.arange(0, STEP_SIZE_C * n)
        mask_c = offs_c < C
        mask_cn = offs_cn < C * n

        grad_output_ptrs = (
            grad_output_ptr
            + offs_m[:, None] * stride_grad_output_m
            + offs_c[None, :] * stride_grad_output_c
        )
        grad_output = tl.load(
            grad_output_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0
        )  # (BLOCK_SIZE_M, STEP_SIZE_C)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
        )  # (BLOCK_SIZE_M, STEP_SIZE_C * n)

        grad_H_pre_acc = tl.dot(
            tl.reshape(grad_output, (BLOCK_SIZE_M, 1, STEP_SIZE_C)),
            tl.reshape(x, (BLOCK_SIZE_M, STEP_SIZE_C, n)),
            acc=grad_H_pre_acc,
            input_precision=precision,
            out_dtype=tl.float32,
        )

        # grad_x = grad_output @ H_pre.T: (BLOCK_SIZE_M, STEP_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n) = (BLOCK_SIZE_M, STEP_SIZE_C, n)
        grad_x = grad_output[:, :, None] * H_pre[:, None, :]  # (BLOCK_SIZE_M, STEP_SIZE_C, n)
        grad_x = tl.reshape(grad_x, (BLOCK_SIZE_M, STEP_SIZE_C * n))

        grad_x_ptrs = (
            grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
        )

        if FUSE_GRAD_X_ACC:  # If fused gradient accumulation is enabled, the buffer is always fp32
            grad_x_acc = tl.load(grad_x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0)
            grad_x = grad_x.to(tl.float32) + grad_x_acc
        tl.store(
            grad_x_ptrs,
            grad_x,
            mask=mask_m[:, None] & mask_cn[None, :],
        )

    grad_H_pre = tl.reshape(grad_H_pre_acc, (BLOCK_SIZE_M * n,))  # (BLOCK_SIZE_M * n)
    offs_grad_H_pre = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_pre_ptrs = grad_H_pre_ptr + offs_grad_H_pre
    if USE_SPLIT_K:
        tl.atomic_add(grad_H_pre_ptrs, grad_H_pre, mask=offs_grad_H_pre < M * n, sem="relaxed")
    else:
        tl.store(grad_H_pre_ptrs, grad_H_pre.to(H_pre.dtype), mask=offs_grad_H_pre < M * n)


def expand_combine_config_fwd():
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


def expand_combine_prune_fwd(configs, named_args, **kwargs):
    M = named_args.get("M", kwargs.get("M", None))

    pruned_configs = list(
        filter(
            lambda config: triton.cdiv(M, config.kwargs["BLOCK_SIZE_M"]) <= MAX_GRID_DIM_Y, configs
        )
    )
    return pruned_configs


@triton.autotune(
    configs=expand_combine_config_fwd(),
    key=["M", "C"],
    prune_configs_by={"early_config_prune": expand_combine_prune_fwd},
)
@triton.jit
def _mhc_expand_combine_fwd(
    f_ptr,  # (M, C)
    bias_ptr,  # (C,), or None if HAS_BIAS is False
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    output_ptr,  # # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_fm,
    stride_fc,
    stride_bias,  # Not used if HAS_BIAS is False
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_Cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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
    if HAS_BIAS:
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
    if HAS_BIAS:
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


def expand_combine_config_bwd():
    block_m = [1, 2, 4]
    block_c = [128, 256]
    step_c = [32, 64]
    warps = [1, 2]
    stages = [1, 2, 3, 4]

    configs = []
    for m, c, sc, w, s in itertools.product(block_m, block_c, step_c, warps, stages):
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_M": m, "BLOCK_SIZE_C": c, "STEP_SIZE_C": sc},
                num_warps=w,
                num_stages=s,
            )
        )
    return configs


def expand_combine_prune_bwd(configs, named_args, **kwargs):
    USE_SPLIT_K = named_args.get("USE_SPLIT_K", kwargs.get("USE_SPLIT_K", None))
    M = named_args.get("M", kwargs.get("M", None))

    # Use Split-K only if determinism is not enforced and M is not large enough to effectively parallelize
    # sms * 8 is a empirical threshold I found via experiments on B200 for non-split-K starts to be better
    if USE_SPLIT_K:
        pruned_configs = configs
    else:
        C = named_args.get("C", kwargs.get("C", None))
        block_m = [4]
        block_c = align_to(C, 32)
        step_c = [32, 64, 128]
        warps = [1, 2]
        stages = [1, 2, 3, 4]

        pruned_configs = []
        for bm, sc, w, s in itertools.product(block_m, step_c, warps, stages):
            pruned_configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": bm,
                        "BLOCK_SIZE_C": block_c,
                        "STEP_SIZE_C": sc,
                    },
                    num_warps=w,
                    num_stages=s,
                )
            )

    pruned_configs = list(
        filter(
            lambda config: triton.cdiv(M, config.kwargs["BLOCK_SIZE_M"]) <= MAX_GRID_DIM_Y,
            pruned_configs,
        )
    )

    # Triton will skip calling prune function if the autotune returns only one config, which breaks the determinism override here
    # So we need to apply NVTE_DISABLE_TRITON_AUTOTUNING in the pruner instead
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        pruned_configs = pruned_configs[:1]
    return pruned_configs


@triton.autotune(
    configs=expand_combine_config_bwd(),
    key=["M", "C", "DETERMINISTIC", "USE_SPLIT_K"],
    reset_to_zero=["grad_H_post_ptr", "grad_H_res_ptr", "grad_bias_ptr"],
    prune_configs_by={"early_config_prune": expand_combine_prune_bwd},
)
@triton.jit
def _mhc_expand_combine_bwd(
    grad_output_ptr,  # (M, C, n)
    f_ptr,  # (M, C)
    bias_ptr,  # (C,), or None if HAS_BIAS is False
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    grad_H_post_ptr,  # (M, n)
    grad_f_ptr,  # (M, C)
    grad_bias_ptr,  # (C,), or None if HAS_BIAS is False
    grad_bias_ws_ptr,  # (M // BLOCK_SIZE_M, C), or None if HAS_BIAS is False or DETERMINISTIC is False
    grad_H_res_ptr,  # (M, n, n)
    grad_x_ptr,  # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_Cn,
    stride_fm,
    stride_fc,
    stride_bias,  # Not used if HAS_BIAS is False
    stride_xm,
    stride_xCn,
    stride_grad_fm,
    stride_grad_fc,
    stride_grad_bias,  # Not used if HAS_BIAS is False
    stride_grad_bias_ws_m,  # Not used if HAS_BIAS is False or DETERMINISTIC is False
    stride_grad_bias_ws_c,  # Not used if HAS_BIAS is False or DETERMINISTIC is False
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    STEP_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
    USE_SPLIT_K: tl.constexpr,
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
    tl.assume(stride_grad_bias_ws_m == C)
    tl.assume(stride_grad_bias_ws_c == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M

    offs_c_start = pid_c * BLOCK_SIZE_C
    offs_cn_start = pid_c * BLOCK_SIZE_C * n

    grad_H_post_acc = tl.zeros((BLOCK_SIZE_M, 1, n), dtype=tl.float32)
    grad_H_res_acc = tl.zeros((BLOCK_SIZE_M, n, n), dtype=tl.float32)

    H_post_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(H_post_ptr + H_post_offs, mask=H_post_offs < M * n, other=0.0)
    H_post_reshape = tl.reshape(H_post, (BLOCK_SIZE_M, 2, 2))
    H_post01, H_post23 = tl.split(H_post_reshape)  # (BLOCK_SIZE_M, 2), (BLOCK_SIZE_M, 2)
    H_post0, H_post1 = tl.split(H_post01)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)
    H_post2, H_post3 = tl.split(H_post23)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(
        H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0
    )  # (BLOCK_SIZE_M, n, n)
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)
    H_res_reshape = tl.reshape(H_res, (BLOCK_SIZE_M, n, 2, 2))  # (BLOCK_SIZE_M, n, 2, 2)
    H_res01, H_res23 = tl.split(H_res_reshape)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    for i in tl.range(0, BLOCK_SIZE_C, STEP_SIZE_C, loop_unroll_factor=2):
        offs_c = offs_c_start + i + tl.arange(0, STEP_SIZE_C)
        offs_cn = offs_cn_start + i * n + tl.arange(0, STEP_SIZE_C * n)
        mask_c = offs_c < C
        mask_cn = offs_cn < C * n

        f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
        f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + offs_c * stride_bias, mask=mask_c, other=0.0
            )  # (STEP_SIZE_C,)

        grad_out_ptrs = (
            grad_output_ptr
            + offs_m[:, None] * stride_grad_output_m
            + offs_cn[None, :] * stride_grad_output_Cn
        )
        grad_out = tl.load(
            grad_out_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
        )  # (BLOCK_SIZE_M, STEP_SIZE_C * n)
        grad_out = tl.reshape(
            grad_out, (BLOCK_SIZE_M, STEP_SIZE_C, n)
        )  # (BLOCK_SIZE_M, STEP_SIZE_C, n)

        # grad_H_post =  f.T @ grad_output # (BLOCK_SIZE_M, 1, STEP_SIZE_C) @ (BLOCK_SIZE_M, STEP_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
        grad_H_post_acc = tl.dot(
            tl.reshape(f, (BLOCK_SIZE_M, 1, STEP_SIZE_C)),
            tl.reshape(grad_out, (BLOCK_SIZE_M, STEP_SIZE_C, n)),
            acc=grad_H_post_acc,
            input_precision=precision,
            out_dtype=tl.float32,
        )  # (BLOCK_SIZE_M, 1, n)
        if HAS_BIAS:
            grad_H_post_acc = tl.dot(
                tl.broadcast_to(bias[None, None, :], (BLOCK_SIZE_M, 1, STEP_SIZE_C)),
                tl.reshape(grad_out, (BLOCK_SIZE_M, STEP_SIZE_C, n)),
                acc=grad_H_post_acc,
                input_precision=precision,
                out_dtype=tl.float32,
            )  # (BLOCK_SIZE_M, 1, n)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
        )  # (BLOCK_SIZE_M, STEP_SIZE_C*n)
        x = tl.reshape(x, (BLOCK_SIZE_M, STEP_SIZE_C, n))  # (BLOCK_SIZE_M, STEP_SIZE_C, n)

        # grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, STEP_SIZE_C) @ (BLOCK_SIZE_M, STEP_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
        grad_H_res_acc = tl.dot(
            tl.trans(x, (0, 2, 1)),
            grad_out,
            acc=grad_H_res_acc,
            input_precision=precision,
            out_dtype=tl.float32,
        )  # (BLOCK_SIZE_M, n, n)

        grad_out_reshape = tl.reshape(
            grad_out, (BLOCK_SIZE_M, STEP_SIZE_C, 2, 2)
        )  # (BLOCK_SIZE_M, STEP_SIZE_C, 2, 2)
        grad_out01, grad_out23 = tl.split(
            grad_out_reshape
        )  # (BLOCK_SIZE_M, STEP_SIZE_C, 2), (BLOCK_SIZE_M, STEP_SIZE_C, 2)
        grad_out0, grad_out1 = tl.split(
            grad_out01
        )  # (BLOCK_SIZE_M, STEP_SIZE_C), (BLOCK_SIZE_M, STEP_SIZE_C)
        grad_out2, grad_out3 = tl.split(
            grad_out23
        )  # (BLOCK_SIZE_M, STEP_SIZE_C), (BLOCK_SIZE_M, STEP_SIZE_C)

        # grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, STEP_SIZE_C) = (BLOCK_SIZE_M, 1, STEP_SIZE_C)
        # Triton doesn't support dot prod with inner dimension < 16, so we need to hack this:
        #        = grad_out[:, :, 0] @ H_post.T[:, 0, :] (BLOCK_SIZE_M, STEP_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, 1)
        #        + grad_out[:, :, 1] @ H_post.T[:, 1, :]
        #        + grad_out[:, :, 2] @ H_post.T[:, 2, :]
        #        + grad_out[:, :, 3] @ H_post.T[:, 3, :]
        # where H_post.T[:, i, :] = H_post[:, :, i]

        grad_f_acc = tl.zeros((BLOCK_SIZE_M, STEP_SIZE_C), dtype=tl.float32)
        # (BLOCK_SIZE_M, STEP_SIZE_C) * (BLOCK_SIZE_M, 1) -> (BLOCK_SIZE_M, STEP_SIZE_C)
        grad_f_acc = tl.fma(grad_out0, H_post0[:, None], grad_f_acc)
        grad_f_acc = tl.fma(grad_out1, H_post1[:, None], grad_f_acc)
        grad_f_acc = tl.fma(grad_out2, H_post2[:, None], grad_f_acc)
        grad_f_acc = tl.fma(grad_out3, H_post3[:, None], grad_f_acc)
        grad_f = grad_f_acc.to(f.dtype)

        grad_f_ptrs = (
            grad_f_ptr + offs_m[:, None] * stride_grad_fm + offs_c[None, :] * stride_grad_fc
        )
        tl.store(grad_f_ptrs, grad_f, mask=mask_m[:, None] & mask_c[None, :])

        if HAS_BIAS:
            grad_bias = tl.sum(grad_f_acc, axis=0)  # (STEP_SIZE_C,)
            # This is reduction over M dimension, so it has nothing to do with whether we use split-C. It only depends on determinism or not.
            if DETERMINISTIC:
                grad_bias_ws_ptrs = (
                    grad_bias_ws_ptr
                    + pid_m * stride_grad_bias_ws_m
                    + offs_c * stride_grad_bias_ws_c
                )
                tl.store(grad_bias_ws_ptrs, grad_bias, mask=mask_c)
            else:
                grad_bias_ptrs = grad_bias_ptr + offs_c * stride_grad_bias
                tl.atomic_add(grad_bias_ptrs, grad_bias, mask=mask_c, sem="relaxed")

        # grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, STEP_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, n, STEP_SIZE_C)
        # The inner dim is n=4 which is too small for triton, so we will manually unroll the matmul
        # grad_x = grad_out[:, :, 0] @ H_res.T[:, 0, :]
        #        + grad_out[:, :, 1] @ H_res.T[:, 1, :]
        #        + grad_out[:, :, 2] @ H_res.T[:, 2, :]
        #        + grad_out[:, :, 3] @ H_res.T[:, 3, :]
        # where H_res.T[:, i, :] = H_res[:, :, i]
        # Due to broadcasting, it's equivalent to multiplying each H_res[:, i, :].T with grad_out[:, i, :]

        grad_x_acc = tl.zeros((BLOCK_SIZE_M, STEP_SIZE_C, n), dtype=tl.float32)
        grad_x_acc = tl.fma(grad_out0[:, :, None], H_res0[:, None, :], grad_x_acc)
        grad_x_acc = tl.fma(grad_out1[:, :, None], H_res1[:, None, :], grad_x_acc)
        grad_x_acc = tl.fma(grad_out2[:, :, None], H_res2[:, None, :], grad_x_acc)
        grad_x_acc = tl.fma(grad_out3[:, :, None], H_res3[:, None, :], grad_x_acc)

        grad_x = grad_x_acc.to(x.dtype)
        grad_x = tl.reshape(
            grad_x, (BLOCK_SIZE_M, STEP_SIZE_C * n)
        )  # (BLOCK_SIZE_M, STEP_SIZE_C*n)

        grad_x_ptrs = (
            grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
        )
        tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None] & mask_cn[None, :])

    grad_H_post = tl.reshape(grad_H_post_acc, (BLOCK_SIZE_M * n,))  # (BLOCK_SIZE_M * n)
    offs_grad_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    grad_H_post_ptrs = grad_H_post_ptr + offs_grad_H_post

    grad_H_res = tl.reshape(grad_H_res_acc, (BLOCK_SIZE_M * n * n,))  # (BLOCK_SIZE_M * n * n)
    offs_grad_H_res = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    grad_H_res_ptrs = grad_H_res_ptr + offs_grad_H_res

    if USE_SPLIT_K:
        tl.atomic_add(grad_H_post_ptrs, grad_H_post, mask=offs_grad_H_post < M * n, sem="relaxed")
        tl.atomic_add(
            grad_H_res_ptrs,
            grad_H_res.to(tl.float32),
            mask=offs_grad_H_res < M * n * n,
            sem="relaxed",
        )
    else:
        tl.store(grad_H_post_ptrs, grad_H_post.to(H_post.dtype), mask=offs_grad_H_post < M * n)
        tl.store(grad_H_res_ptrs, grad_H_res.to(H_res.dtype), mask=offs_grad_H_res < M * n * n)
