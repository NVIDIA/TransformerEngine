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

/*! \brief D[i,j] = bf16(alpha_a[i] * alpha_b[j] * (A @ B^T)[i,j]). Per-row *
 *  per-col rescale fused into the EVT epilogue (replaces the trailing
 *  nvte_nvfp4_per_token_post_scale kernel). alpha_a/b are FP32 (M,)/(N,). */
void nvte_nvfp4_cutlass_per_token_gemm(const NVTETensor a_data, const NVTETensor b_data,
                                       const NVTETensor a_sf, const NVTETensor b_sf,
                                       const NVTETensor alpha_a, const NVTETensor alpha_b,
                                       NVTETensor d, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TRANSFORMER_ENGINE_NVFP4_CUTLASS_GEMM_H_
