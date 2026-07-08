/*************************************************************************
* Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
*
* See LICENSE for license information.
************************************************************************/

// cuBLAS(Lt) grouped GEMM backend. The backend-agnostic setup kernels, operand
// selection, dispatch and public nvte_grouped_* entry points live in
// grouped_gemm.cu; shared types are in grouped_gemm_common.h. This file only
// implements execute_grouped_gemm() and its cuBLASLt-specific helpers.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <transformer_engine/transformer_engine.h>

#include <cstddef>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/handle_manager.h"
#include "../util/logging.h"
#include "./grouped_gemm_common.h"

using namespace transformer_engine::grouped_gemm;

namespace {

inline void CreateCublasHandle(cublasLtHandle_t *handle) {
  NVTE_CHECK_CUBLAS(cublasLtCreate(handle));
}

}  // namespace

#if CUBLAS_VERSION >= CUBLAS_GROUPED_GEMM_VERSION

namespace {

inline void init_matrix_layouts(
    cublasLtMatrixLayoutOpaque_t &descA, cublasLtMatrixLayoutOpaque_t &descB,
    cublasLtMatrixLayoutOpaque_t &descC, cublasLtMatrixLayoutOpaque_t &descD,
    const GroupedGemmSetupWorkspace &ws, const GroupedOperandSelection &A_sel,
    const GroupedOperandSelection &B_sel, transformer_engine::DType d_dtype, size_t num_tensors) {
  const cudaDataType_t A_type = get_cuda_dtype(A_sel.dtype);
  const cudaDataType_t B_type = get_cuda_dtype(B_sel.dtype);
  const cudaDataType_t D_type = get_cuda_dtype(d_dtype);

  // Storage dimensions computed by kernel, leading dimension = rows
  NVTE_CHECK_CUBLAS(cublasLtGroupedMatrixLayoutInit(&descA, A_type, num_tensors, ws.a_rows,
                                                    ws.a_cols, ws.a_rows));
  NVTE_CHECK_CUBLAS(cublasLtGroupedMatrixLayoutInit(&descB, B_type, num_tensors, ws.b_rows,
                                                    ws.b_cols, ws.b_rows));
  NVTE_CHECK_CUBLAS(cublasLtGroupedMatrixLayoutInit(&descC, D_type, num_tensors, ws.d_rows,
                                                    ws.d_cols, ws.d_rows));
  NVTE_CHECK_CUBLAS(cublasLtGroupedMatrixLayoutInit(&descD, D_type, num_tensors, ws.d_rows,
                                                    ws.d_cols, ws.d_rows));
}

inline void init_matmul_desc(cublasLtMatmulDescOpaque_t &matmulDesc, cublasOperation_t op_A,
                             cublasOperation_t op_B, bool use_fp8, bool use_split_accumulator,
                             bool use_per_group_alpha_beta) {
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescInit(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_A,
                                                   sizeof(op_A)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_B,
                                                   sizeof(op_B)));

  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                   &pointer_mode, sizeof(pointer_mode)));

  if (use_per_group_alpha_beta) {
    int64_t alphabeta_batch_stride = 1;
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                     CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE,
                                                     &alphabeta_batch_stride, sizeof(int64_t)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                     CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE,
                                                     &alphabeta_batch_stride, sizeof(int64_t)));
  }

  // Fast accumulation is only supported for FP8 (mirrors non-grouped GEMM logic).
  int8_t fastAccuMode = use_split_accumulator ? 0 : static_cast<int8_t>(use_fp8);
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                   &fastAccuMode, sizeof(fastAccuMode)));
}

// Configures cuBLAS for MXFP8 grouped GEMM: sets VEC32_UE8M0 scale mode and scale pointers
// for both A and B.
inline void set_mxfp8_scale_pointers(cublasLtMatmulDescOpaque_t &matmulDesc,
                                     void **a_scale_inv_ptrs, void **b_scale_inv_ptrs) {
#if CUBLAS_VERSION >= CUBLAS_MXFP8_GROUPED_GEMM_VERSION
  NVTE_CHECK(transformer_engine::cuda::cublas_version() >= CUBLAS_MXFP8_GROUPED_GEMM_VERSION,
             "MXFP8 grouped GEMM requires cuBLAS ", CUBLAS_MXFP8_GROUPED_GEMM_VERSION,
             "+, but run-time cuBLAS version is ", transformer_engine::cuda::cublas_version());
  const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                   &scale_mode, sizeof(scale_mode)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                   &scale_mode, sizeof(scale_mode)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                   &a_scale_inv_ptrs, sizeof(a_scale_inv_ptrs)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                   &b_scale_inv_ptrs, sizeof(b_scale_inv_ptrs)));
#else
  NVTE_CHECK(false, "MXFP8 grouped GEMM requires cuBLAS ", CUBLAS_MXFP8_GROUPED_GEMM_VERSION,
             "+, but compile-time cuBLAS version is ", CUBLAS_VERSION);
#endif  // CUBLAS_VERSION >= CUBLAS_MXFP8_GROUPED_GEMM_VERSION
}

// Configures cuBLAS for NVFP4 grouped GEMM: sets VEC16_UE4M3 scale mode and scale pointers
// for both A and B. Requires cuBLAS 13.4+.
inline void set_nvfp4_scale_pointers(cublasLtMatmulDescOpaque_t &matmulDesc,
                                     void **a_scale_inv_ptrs, void **b_scale_inv_ptrs) {
#if CUBLAS_VERSION >= CUBLAS_NVFP4_GROUPED_GEMM_VERSION
  NVTE_CHECK(transformer_engine::cuda::cublas_version() >= CUBLAS_NVFP4_GROUPED_GEMM_VERSION,
             "NVFP4 grouped GEMM requires cuBLAS 13.4+, but run-time cuBLAS version is ",
             transformer_engine::cuda::cublas_version());
  const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                   &scale_mode, sizeof(scale_mode)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                   &scale_mode, sizeof(scale_mode)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                   &a_scale_inv_ptrs, sizeof(a_scale_inv_ptrs)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                   &b_scale_inv_ptrs, sizeof(b_scale_inv_ptrs)));
#else
  NVTE_CHECK(false,
             "NVFP4 grouped GEMM requires cuBLAS 13.4+, but compile-time "
             "cuBLAS version is ",
             CUBLAS_VERSION);
#endif  // CUBLAS_VERSION >= CUBLAS_NVFP4_GROUPED_GEMM_VERSION
}

// Configures cuBLAS for FP8 block-scaling grouped GEMM: sets VEC128_32F or BLK128x128_32F
// scale mode and scale pointers for A and B. Requires cuBLAS 13.4+.
inline void set_fp8_block_scaling_scale_pointers(cublasLtMatmulDescOpaque_t &matmulDesc,
                                                 void **a_scale_inv_ptrs, void **b_scale_inv_ptrs,
                                                 NVTEScalingMode a_scaling_mode,
                                                 NVTEScalingMode b_scaling_mode) {
#if CUBLAS_VERSION >= CUBLAS_FP8_BLOCK_GROUPED_GEMM_VERSION
  NVTE_CHECK(
      transformer_engine::cuda::cublas_version() >= CUBLAS_FP8_BLOCK_GROUPED_GEMM_VERSION,
      "FP8 block scaling grouped GEMM requires cuBLAS 13.4+, but run-time cuBLAS version is ",
      transformer_engine::cuda::cublas_version());

  NVTE_CHECK(!(a_scaling_mode == NVTE_BLOCK_SCALING_2D && b_scaling_mode == NVTE_BLOCK_SCALING_2D),
             "Only 1D by 1D, 1D by 2D, and 2D by 1D block scaling grouped GEMM is supported, "
             "but got 2D by 2D");

  const cublasLtMatmulMatrixScale_t scale_mode_a =
      a_scaling_mode == NVTE_BLOCK_SCALING_1D ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
                                              : CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
  const cublasLtMatmulMatrixScale_t scale_mode_b =
      b_scaling_mode == NVTE_BLOCK_SCALING_1D ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
                                              : CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                   &scale_mode_a, sizeof(scale_mode_a)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                   &scale_mode_b, sizeof(scale_mode_b)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                   &a_scale_inv_ptrs, sizeof(a_scale_inv_ptrs)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                   &b_scale_inv_ptrs, sizeof(b_scale_inv_ptrs)));
#else
  NVTE_CHECK(false,
             "FP8 block scaling grouped GEMM requires cuBLAS 13.4+, but compile-time "
             "cuBLAS version is ",
             CUBLAS_VERSION);
#endif  // CUBLAS_VERSION >= CUBLAS_FP8_BLOCK_GROUPED_GEMM_VERSION
}

// Configures cuBLAS for tensor-scaling FP8 grouped GEMM: sets PER_BATCH_SCALAR_32F scale mode
// and scale pointers for A and B. Both operands are guaranteed FP8 by the caller.
inline void set_fp8_scale_pointers(cublasLtMatmulDescOpaque_t &matmulDesc, void **a_scale_inv_ptrs,
                                   void **b_scale_inv_ptrs) {
  const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                   &scale_mode, sizeof(scale_mode)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                   &a_scale_inv_ptrs, sizeof(a_scale_inv_ptrs)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                   &scale_mode, sizeof(scale_mode)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                   &b_scale_inv_ptrs, sizeof(b_scale_inv_ptrs)));
}

inline cublasLtMatmulAlgo_t select_grouped_gemm_algo(cublasLtHandle_t handle,
                                                     cublasLtMatmulDescOpaque_t &matmulDesc,
                                                     cublasLtMatrixLayoutOpaque_t &descA,
                                                     cublasLtMatrixLayoutOpaque_t &descB,
                                                     cublasLtMatrixLayoutOpaque_t &descC,
                                                     cublasLtMatrixLayoutOpaque_t &descD,
                                                     int64_t avg_m, int64_t avg_n, int64_t avg_k) {
  cublasLtMatmulPreferenceOpaque_t preference;
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceInit(&preference));
  NVTE_CHECK_CUBLAS(
      cublasLtMatmulPreferenceSetAttribute(&preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &kGroupedGemmCublasWorkspaceSize, sizeof(size_t)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      &preference, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, &avg_m, sizeof(int64_t)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      &preference, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, &avg_n, sizeof(int64_t)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      &preference, CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, &avg_k, sizeof(int64_t)));

  cublasLtMatmulHeuristicResult_t heuristicResult;
  int returnedResults = 0;
  auto status = cublasLtMatmulAlgoGetHeuristic(handle, &matmulDesc, &descA, &descB, &descC, &descD,
                                               &preference, 1, &heuristicResult, &returnedResults);
  NVTE_CHECK(status != CUBLAS_STATUS_NOT_SUPPORTED,
             "Unable to find suitable cuBLAS grouped GEMM algorithm");
  NVTE_CHECK_CUBLAS(status);
  NVTE_CHECK(returnedResults > 0, "No suitable algorithm found for grouped GEMM");
  return heuristicResult.algo;
}

}  // namespace

namespace transformer_engine {
namespace grouped_gemm {

void execute_grouped_gemm(const GroupedGemmSetupWorkspace &setup_workspace,
                          const GroupedOperandSelection &A_sel,
                          const GroupedOperandSelection &B_sel, transformer_engine::DType d_dtype,
                          size_t num_tensors, const GroupedGemmConfig &config,
                          void *cublas_workspace_ptr, cudaStream_t stream) {
  using cublasHandleManager =
      transformer_engine::detail::HandleManager<cublasLtHandle_t, CreateCublasHandle>;
  cublasLtHandle_t handle = cublasHandleManager::Instance().GetHandle();

  cublasOperation_t op_A = A_sel.trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_B = B_sel.trans ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasLtMatrixLayoutOpaque_t descA, descB, descC, descD;
  init_matrix_layouts(descA, descB, descC, descD, setup_workspace, A_sel, B_sel, d_dtype,
                      num_tensors);

  cublasLtMatmulDescOpaque_t matmulDesc;
  init_matmul_desc(matmulDesc, op_A, op_B, config.use_fp8, config.use_split_accumulator,
                   config.use_per_group_alpha_beta);
  if (transformer_engine::is_mxfp_scaling(A_sel.scaling_mode)) {
    set_mxfp8_scale_pointers(matmulDesc, setup_workspace.a_scale_inv_ptrs,
                             setup_workspace.b_scale_inv_ptrs);
  } else if (transformer_engine::is_nvfp_scaling(A_sel.scaling_mode)) {
    set_nvfp4_scale_pointers(matmulDesc, setup_workspace.a_scale_inv_ptrs,
                             setup_workspace.b_scale_inv_ptrs);
  } else if (transformer_engine::is_fp8_block_scaling(A_sel.scaling_mode)) {
    set_fp8_block_scaling_scale_pointers(matmulDesc, setup_workspace.a_scale_inv_ptrs,
                                         setup_workspace.b_scale_inv_ptrs, A_sel.scaling_mode,
                                         B_sel.scaling_mode);
  } else if (config.use_fp8) {
    const int sm = transformer_engine::cuda::sm_arch(transformer_engine::cuda::current_device());
    if (sm < 100) {
      NVTE_CHECK(transformer_engine::cuda::cublas_version() >=
                     CUBLAS_FP8_TENSOR_SCALING_GROUPED_GEMM_HOPPER_VERSION,
                 "FP8 tensor scaling grouped GEMM on Hopper (SM90) requires cuBLAS 13.5+, "
                 "but run-time cuBLAS version is ",
                 transformer_engine::cuda::cublas_version());
    }
    set_fp8_scale_pointers(matmulDesc, setup_workspace.a_scale_inv_ptrs,
                           setup_workspace.b_scale_inv_ptrs);
  }
  if (config.sm_count != 0) {
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                     CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                     &config.sm_count, sizeof(config.sm_count)));
  }
  cublasLtMatmulAlgo_t algo = select_grouped_gemm_algo(
      handle, matmulDesc, descA, descB, descC, descD, config.avg_m, config.avg_n, config.avg_k);

  // Hopper uses a single scalar alpha/beta for the whole grouped GEMM;
  // Blackwell+ uses per-matrix alpha/beta arrays.
  void *alpha_arg = config.use_per_group_alpha_beta
                        ? static_cast<void *>(setup_workspace.alpha_ptrs)
                        : config.alpha_dptr;
  void *beta_arg = config.use_per_group_alpha_beta ? static_cast<void *>(setup_workspace.beta_ptrs)
                                                   : config.beta_dptr;

  NVTE_CHECK_CUBLAS(cublasLtMatmul(handle, &matmulDesc, alpha_arg, setup_workspace.A_ptrs, &descA,
                                   setup_workspace.B_ptrs, &descB, beta_arg, setup_workspace.C_ptrs,
                                   &descC, setup_workspace.D_ptrs, &descD, &algo,
                                   cublas_workspace_ptr, kGroupedGemmCublasWorkspaceSize, stream));
}

}  // namespace grouped_gemm
}  // namespace transformer_engine

#endif  // CUBLAS_VERSION >= CUBLAS_GROUPED_GEMM_VERSION
