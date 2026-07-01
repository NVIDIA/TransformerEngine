/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

/**
 * @file nvfp4_cutlass_grouped_gemm.cuh
 * @brief Single-launch CUTLASS grouped (MoE) GEMM for *per-tensor* NVFP4 on
 *        SM100 (Blackwell). Replaces the multi-stream cuBLASLt loop in the
 *        production NVFP4 grouped path with one CUTLASS ptr-array launch.
 *
 * Compared to the per-token grouped kernel, the per-tensor second-level scale
 * collapses to a single fp32 scalar per group
 *   alpha[g] = amax_A[g] * amax_B[g] / (fp4_max^2 * fp8_max^2),
 * applied through the epilogue's per-group alpha_ptr_array (default
 * LinearCombination), so no vector row/col broadcast EVT is needed.
 *
 * The launcher takes raw device-pointer vectors so the TE-tensor / scale /
 * layout extraction (and the cuBLAS->CUTLASS A/B swap) lives in the dispatcher
 * in cublaslt_gemm.cu, reusing CanonicalizeGemmInput for parity with cuBLASLt.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <vector>

namespace transformer_engine {
namespace nvfp4_cutlass {

// Single-launch per-tensor NVFP4 grouped GEMM. CUTLASS computes, per group,
//   D[g] = out_dtype(alpha[g] * (A[g] @ B[g]^T) + beta * C[g])
// with A[g] row-major (M,K) FP4, B[g] col-major (N,K) FP4, D[g] row-major
// (M,N). a_sf/b_sf are the swizzled e4m3 1x16 block-scale buffers for A/B.
// alpha_ptrs[g] points to a single device fp32 holding the per-group global
// second-level scale. M, N, K must be multiples of 128.
//
// Output / accumulate / bias modes:
//   * fp32_output == false -> BF16 output, overwrite (fprop / dgrad).
//   * fp32_output == true  -> FP32 output (wgrad). With accumulate == true,
//     beta = 1 and C == D, i.e. D (== main_grad) is read-modify-written
//     in-place (Megatron wgrad fusion). accumulate requires fp32_output.
//   * bias_ptrs non-empty -> fused per-group bias (fprop): each bias_ptrs[g]
//     points to group g's length-N bias vector (D-element dtype), added in the
//     epilogue (D[g] = alpha[g]*acc + bias[g]). Requires BF16 output and
//     overwrite (no accumulate). Pass an empty vector when there is no bias.
void run_grouped_per_tensor_gemm(
    const std::vector<const void *> &a_data, const std::vector<const void *> &b_data,
    const std::vector<const void *> &a_sf, const std::vector<const void *> &b_sf,
    const std::vector<const float *> &alpha_ptrs, const std::vector<void *> &d_ptrs,
    const std::vector<const void *> &bias_ptrs, const std::vector<int> &Ms,
    const std::vector<int> &Ns, const std::vector<int> &Ks, bool fp32_output, bool accumulate,
    cudaStream_t stream);

}  // namespace nvfp4_cutlass
}  // namespace transformer_engine
