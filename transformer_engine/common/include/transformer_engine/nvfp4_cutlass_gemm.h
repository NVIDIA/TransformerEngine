/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nvfp4_cutlass_gemm.h
 *  \brief CUTLASS NVFP4 GEMM kernels: scalar (alpha, beta) and per-row*per-col
 *         fused EVT variants. BF16 output matches the cublasLt NVFP4 path. */

#ifndef TRANSFORMER_ENGINE_NVFP4_CUTLASS_GEMM_H_
#define TRANSFORMER_ENGINE_NVFP4_CUTLASS_GEMM_H_

#include <cuda_runtime_api.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/*! \brief D = alpha * (A @ B^T) + beta * C. A row-major (M,K), B col-major
 *  (K,N), D/C row-major (M,N). A/B FP4-e2m1 packed; SFs FP8-e4m3 in CUTLASS
 *  Sm1xxBlkScaledConfig layout; D BF16. M, N, K must be multiples of 256. */
void nvte_nvfp4_cutlass_gemm(const NVTETensor a_data, const NVTETensor b_data,
                             const NVTETensor a_sf, const NVTETensor b_sf, NVTETensor d,
                             float alpha, float beta, cudaStream_t stream);

/*! \brief D[i,j] = alpha_a[i] * alpha_b[j] * (A @ B^T)[i,j] (per-row * per-col
 *  rescale fused into the EVT epilogue). alpha_a/b are FP32 (M,)/(N,).
 *
 *  D may be BF16 (plain overwrite) or FP32. When D is FP32, accumulate=true
 *  computes D += ... in place (used for wgrad fused into a FP32 main_grad);
 *  accumulate=false overwrites. accumulate=true requires a FP32 D. */
void nvte_nvfp4_cutlass_per_token_gemm(const NVTETensor a_data, const NVTETensor b_data,
                                       const NVTETensor a_sf, const NVTETensor b_sf,
                                       const NVTETensor alpha_a, const NVTETensor alpha_b,
                                       NVTETensor d, bool accumulate, cudaStream_t stream);

/*! \brief Grouped (MoE) variant of nvte_nvfp4_cutlass_per_token_gemm.
 *
 *  Computes, for every group g in [0, num_groups):
 *      D_g[i,j] = bf16(alpha_a_g[i] * alpha_b_g[j] * (A_g @ B_g^T)[i,j])
 *  with a single CUTLASS ptr-array grouped NVFP4 launch (no per-expert loop).
 *
 *  All array parameters are host arrays of length num_groups; the underlying
 *  data they reference must live on device. Per group:
 *    - a_data[g] : FP4-e2m1 packed, logical (M_g, K)
 *    - b_data[g] : FP4-e2m1 packed, logical (N_g, K)
 *    - a_sf[g]   : FP8-e4m3 1x16 inner SF for A, ALREADY in CUTLASS
 *                  Sm1xxBlkScaledConfig swizzled layout
 *    - b_sf[g]   : FP8-e4m3 1x16 inner SF for B, ALREADY swizzled
 *    - alpha_a[g]: FP32 per-row outer scale, length M_g
 *    - alpha_b[g]: FP32 per-col outer scale, length N_g
 *    - d[g]      : BF16 (overwrite) or FP32 output, logical (M_g, N_g)
 *    - bias      : optional (may be NULL). When non-NULL, bias[g] is an FP32
 *                  (N_g,) vector fused into the epilogue: D_g += bias[g] (added
 *                  in FP32 before the BF16 cast). Forward-only; requires BF16
 *                  outputs (mutually exclusive with accumulate).
 *
 *  Each group must satisfy M_g % 128 == 0, N_g % 128 == 0, K % 128 == 0
 *  (same 1-CTA MmaTile = (128,128,256) constraint as the dense per-token
 *  kernel). Groups with M_g == 0 must be filtered out by the caller.
 *
 *  When d is FP32, accumulate=true computes d[g] += ... in place (wgrad fused
 *  into FP32 main_grad); accumulate=false overwrites. accumulate requires FP32
 *  outputs. The output dtype must be uniform across groups. */
void nvte_nvfp4_cutlass_grouped_per_token_gemm(
    int num_groups, const NVTETensor *a_data, const NVTETensor *b_data, const NVTETensor *a_sf,
    const NVTETensor *b_sf, const NVTETensor *alpha_a, const NVTETensor *alpha_b, NVTETensor *d,
    const NVTETensor *bias, bool accumulate, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TRANSFORMER_ENGINE_NVFP4_CUTLASS_GEMM_H_
