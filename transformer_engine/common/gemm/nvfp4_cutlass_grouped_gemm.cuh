/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

/**
 * @file nvfp4_cutlass_grouped_gemm.cuh
 * @brief Single-launch CUTLASS grouped (MoE) GEMM for *per-tensor* NVFP4 on
 *        SM100 (Blackwell). Env-selectable (NVTE_NVFP4_CUTLASS_GROUPED_GEMM)
 *        backend for the grouped-tensor grouped-GEMM path, replacing the
 *        cuBLAS single-launch kernel with one CUTLASS ptr-array launch.
 *
 * Compared to the per-token grouped kernel, the per-tensor second-level scale
 * collapses to a single fp32 scalar per group
 *   alpha[g] = amax_A[g] * amax_B[g] / (fp4_max^2 * fp8_max^2),
 * applied through the epilogue's per-group alpha_ptr_array (default
 * LinearCombination), so no vector row/col broadcast EVT is needed.
 *
 * The launcher consumes device-side pointer / dim arrays (produced by the
 * shared GroupedTensor setup kernel in cublaslt_grouped_gemm.cu) and performs
 * the cuBLAS->CUTLASS A/B swap internally, so the whole launch is CUDA-graph
 * capturable with no host<->device sync.
 */

#pragma once

#include <cuda_runtime_api.h>

namespace transformer_engine {
namespace nvfp4_cutlass {

// Graph-safe device-array entry for the grouped-tensor (cublasLt grouped) API.
//
// Every per-group array already lives on device: they are the pointer / dim
// arrays that the shared GroupedTensor setup kernel (launch_grouped_gemm_setup)
// writes into GroupedGemmSetupWorkspace, derived on-device from the (possibly
// dynamic) GroupedTensor offsets. This launcher therefore performs NO host<->
// device sync and NO host-side pointer arithmetic, so it is CUDA-graph
// capturable.
//
// Inputs use the cuBLAS operand convention (A/B/D as passed to execute_grouped_
// gemm). The cuBLAS->CUTLASS A/B swap and the M/N/K derivation happen internally:
//   CUTLASS A := B_ptrs (sfa := b_scale_inv_ptrs), CUTLASS B := A_ptrs
//   (sfb := a_scale_inv_ptrs); M = d_cols, N = d_rows, K = a_trans ? a_rows :
//   a_cols. alpha_ptrs holds the per-group global (second-level) NVFP4 scale.
//
// Accumulate / C source:
//   * beta_ptrs == nullptr -> overwrite: D = alpha[g] * (A@B). No C is read
//     (ptr_C = nullptr), matching the proven-safe LinearCombination overwrite
//     path (C_ptrs is ignored, so uninitialized D is never loaded).
//   * beta_ptrs != nullptr -> per-group beta: D = alpha[g]*(A@B) + beta[g]*C[g]
//     with C := C_ptrs (== D for in-place wgrad accumulation). beta[g] lives on
//     device; the epilogue's Sm90ScalarBroadcastPtrArray consumes it per group,
//     so this stays graph-safe. Requires fp32 output.
// Fused bias is not supported here (the grouped-tensor path applies bias via a
// separate nvte_grouped_bias_add); callers must exclude the fused-bias case.
void run_grouped_per_tensor_gemm_grouped_tensor(
    void **A_ptrs, void **B_ptrs, void **a_scale_inv_ptrs, void **b_scale_inv_ptrs,
    float **alpha_ptrs, void **C_ptrs, void **D_ptrs, float **beta_ptrs, const int *a_rows,
    const int *a_cols, const int *d_rows, const int *d_cols, bool a_trans, int num_groups,
    bool fp32_output, cudaStream_t stream);

}  // namespace nvfp4_cutlass
}  // namespace transformer_engine
