/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>

#include "../common.h"
#include "../util/logging.h"

namespace {

cudaDataType_t get_cuda_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return CUDA_R_16F;
    case DType::kFloat32:
      return CUDA_R_32F;
    case DType::kBFloat16:
      return CUDA_R_16BF;
    case DType::kFloat8E4M3:
      return CUDA_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return CUDA_R_8F_E5M2;
    default:
      NVTE_ERROR("Invalid type");
  }
}

uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (;; alignment /= 2) {
    if (address % alignment == 0) {
      return alignment;
    }
  }
}

}  // namespace

namespace transformer_engine {

void cublas_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                 const Tensor *inputBias, Tensor *outputPreGelu, int m, int n, int k, int lda,
                 int ldb, int ldd, cublasOperation_t transa, cublasOperation_t transb, bool grad,
                 void *workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                 int math_sm_count, int m_split, int n_split, bool gemm_producer,
                 const Tensor *inputCounter, cudaStream_t stream) {
  void *A = inputA->data.dptr;
  void *A_scale_inverse = inputA->scale_inv.dptr;
  void *B = inputB->data.dptr;
  void *B_scale_inverse = inputB->scale_inv.dptr;
  void *C = outputD->data.dptr;
  void *D = outputD->data.dptr;
  void *D_scale = outputD->scale.dptr;
  void *D_amax = outputD->amax.dptr;
  void *bias_ptr = inputBias->data.dptr;
  const bool bias = bias_ptr != nullptr;
  void *pre_gelu_out = outputPreGelu->data.dptr;
  void *counter = nullptr;
  if (inputCounter != nullptr) {
    counter = inputCounter->data.dptr;
  }
  const bool gelu = pre_gelu_out != nullptr;
  const bool use_fp8 = is_fp8_dtype(inputA->data.dtype) || is_fp8_dtype(inputB->data.dtype);
  const cudaDataType_t A_type = get_cuda_dtype(inputA->data.dtype);
  const cudaDataType_t B_type = get_cuda_dtype(inputB->data.dtype);
  const cudaDataType_t D_type = get_cuda_dtype(outputD->data.dtype);
  const cudaDataType_t bias_type = get_cuda_dtype(inputBias->data.dtype);

  NVTE_CHECK(!is_fp8_dtype(inputA->data.dtype) || A_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");
  NVTE_CHECK(!is_fp8_dtype(inputB->data.dtype) || B_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");

  // check consistency of arguments:
  // if fp8 is desired, context cannot be null
  // fp8 + gelu fusion + fp8 aux is unavailable right now.
  if (use_fp8 && gelu) {
    NVTE_CHECK(!is_fp8_dtype(outputPreGelu->data.dtype),
               "fp8 Aux output for gemm + gelu fusion not supported!");
  }
  if (is_fp8_dtype(outputD->data.dtype)) {
    NVTE_CHECK(!accumulate, "Accumulation mode not supported with FP8 GEMM output!");
  }

  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  cublasLtHandle_t handle;
  NVTE_CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t)ldd;

  // Use TF32 only for pure FP32 GEMM.
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  if (A_type == CUDA_R_32F && B_type == CUDA_R_32F && D_type == CUDA_R_32F) {
    gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type, transa == CUBLAS_OP_N ? m : k,
                                               transa == CUBLAS_OP_N ? k : m, lda));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type, transb == CUBLAS_OP_N ? k : n,
                                               transb == CUBLAS_OP_N ? n : k, ldb));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                   &transa, sizeof(transa)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                   &transb, sizeof(transb)));
  // Set math SM count
  if (math_sm_count != 0) {
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                     &math_sm_count, sizeof(math_sm_count)));
  }

  // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
  // Note: gelu fusion isn't available right now, and we don't need
  // amax(D) either (next op is high precision).
  if (use_fp8) {
    // Split accumulator.
    const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                     &fastAccuMode, sizeof(fastAccuMode)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                     &A_scale_inverse, sizeof(A_scale_inverse)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                     &B_scale_inverse, sizeof(B_scale_inverse)));
    if (is_fp8_dtype(outputD->data.dtype)) {
      // Accumulation mode not supported for FP8 output
      C = nullptr;
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &D_scale, sizeof(D_scale)));
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &D_amax, sizeof(D_amax)));
      // For FP8 output, cuBLAS requires C_type to be same as bias_type
      NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, bias_type, m, n, ldd));
    } else {
      NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
    }
    if (bias) {
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
    }
  } else {
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
  }

  if (bias && gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                     &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld_gelumat, sizeof(ld_gelumat)));
    const cudaDataType_t aux_type = get_cuda_dtype(outputPreGelu->data.dtype);
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &aux_type, sizeof(aux_type)));
  } else if (bias) {
    if (grad) {
      // grad output is always input B
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
    } else {
      epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  } else if (gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
    }
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                     &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld_gelumat, sizeof(ld_gelumat)));
  }

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue)));
#if CUDA_VERSION >= 12020 && CUBLAS_VERSION >= 120205
  if (counter != nullptr) {
    if (m_split == 0) m_split = 1;
    if (n_split == 0) n_split = 1;
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS, &m_split,
        sizeof(m_split)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS, &n_split,
        sizeof(n_split)));
    if (gemm_producer) {
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER, &counter,
          sizeof(counter)));
    } else {
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER, &counter,
          sizeof(counter)));
    }
  }
#endif

  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
  const auto A_alignment = _getAlignment(reinterpret_cast<uintptr_t>(A));
  const auto B_alignment = _getAlignment(reinterpret_cast<uintptr_t>(B));
  const auto C_alignment = _getAlignment(reinterpret_cast<uintptr_t>(C));
  const auto D_alignment = _getAlignment(reinterpret_cast<uintptr_t>(D));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &A_alignment, sizeof(A_alignment)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &B_alignment, sizeof(B_alignment)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &C_alignment, sizeof(C_alignment)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &D_alignment, sizeof(D_alignment)));

  const auto status =
      cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
                                     1, &heuristicResult, &returnedResults);
  NVTE_CHECK(status != CUBLAS_STATUS_NOT_SUPPORTED,
             "Unable to find suitable cuBLAS GEMM algorithm");
  NVTE_CHECK_CUBLAS(status);

  if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");

  // D = alpha * (A * B) + beta * C
  NVTE_CHECK_CUBLAS(cublasLtMatmul(handle, operationDesc,
                                   static_cast<const void *>(&one),         /* alpha */
                                   A,                                       /* A */
                                   Adesc, B,                                /* B */
                                   Bdesc, static_cast<const void *>(&beta), /* beta */
                                   C,                                       /* C */
                                   Cdesc, D,                                /* D */
                                   Ddesc, &heuristicResult.algo,            /* algo */
                                   workspace,                               /* workspace */
                                   workspaceSize, stream));                 /* stream */

  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
}

}  // namespace transformer_engine

void nvte_cublas_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D, const NVTETensor bias,
                      NVTETensor pre_gelu_out, bool transa, bool transb, bool grad,
                      NVTETensor workspace, bool accumulate, bool use_split_accumulator,
                      int math_sm_count, cudaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_gemm);
  using namespace transformer_engine;
  const Tensor *inputA = reinterpret_cast<const Tensor *>(A);
  const Tensor *inputB = reinterpret_cast<const Tensor *>(B);
  Tensor *outputD = reinterpret_cast<Tensor *>(D);
  const Tensor *biasTensor = reinterpret_cast<const Tensor *>(bias);
  Tensor *outputGelu = reinterpret_cast<Tensor *>(pre_gelu_out);
  Tensor *wspace = reinterpret_cast<Tensor *>(workspace);

  const int m = transa ? inputA->data.shape[0] : inputA->data.shape[1];
  const int k = transa ? inputA->data.shape[1] : inputA->data.shape[0];
  const int n = transb ? inputB->data.shape[1] : inputB->data.shape[0];
  int lda, ldb, ldd;
  if (transa && !transb) {  // TN
    lda = k;
    ldb = k;
    ldd = m;
  } else if (!transa && !transb) {  // NN
    lda = m;
    ldb = k;
    ldd = m;
  } else if (!transa && transb) {  // NT
    lda = m;
    ldb = n;
    ldd = m;
  } else {  // TT
    NVTE_ERROR("TT layout not allowed.");
  }

  cublas_gemm(inputA, inputB, outputD, biasTensor, outputGelu, m, n, k, lda, ldb, ldd,
              (transa) ? CUBLAS_OP_T : CUBLAS_OP_N, (transb) ? CUBLAS_OP_T : CUBLAS_OP_N, grad,
              wspace->data.dptr, wspace->data.shape[0], accumulate, use_split_accumulator,
              math_sm_count, 0, 0, false, nullptr, stream);
}

void nvte_cublas_atomic_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D,
                             const NVTETensor bias, NVTETensor pre_gelu_out, bool transa,
                             bool transb, bool grad, NVTETensor workspace, bool accumulate,
                             bool use_split_accumulator, int math_sm_count, int m_split,
                             int n_split, bool gemm_producer, const NVTETensor counter,
                             cudaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_atomic_gemm);

  int cudart_version;
  NVTE_CHECK_CUDA(cudaRuntimeGetVersion(&cudart_version));
  NVTE_CHECK(cudart_version >= 12020, "Cuda version 12.2 is required for atomic gemm.");
  NVTE_CHECK(cublasLtGetVersion() >= 120205, "Cublas version 12.2.5 is required for atomic gemm.");

  using namespace transformer_engine;
  const Tensor *inputA = reinterpret_cast<const Tensor *>(A);
  const Tensor *inputB = reinterpret_cast<const Tensor *>(B);
  Tensor *outputD = reinterpret_cast<Tensor *>(D);
  const Tensor *biasTensor = reinterpret_cast<const Tensor *>(bias);
  Tensor *outputGelu = reinterpret_cast<Tensor *>(pre_gelu_out);
  const Tensor *inputCounter = reinterpret_cast<const Tensor *>(counter);
  Tensor *wspace = reinterpret_cast<Tensor *>(workspace);

  const int m = transa ? inputA->data.shape[0] : inputA->data.shape[1];
  const int k = transa ? inputA->data.shape[1] : inputA->data.shape[0];
  const int n = transb ? inputB->data.shape[1] : inputB->data.shape[0];
  int lda, ldb, ldd;
  if (transa && !transb) {  // TN
    lda = k;
    ldb = k;
    ldd = m;
  } else if (!transa && !transb) {  // NN
    lda = m;
    ldb = k;
    ldd = m;
  } else if (!transa && transb) {  // NT
    lda = m;
    ldb = n;
    ldd = m;
  } else {  // TT
    NVTE_ERROR("TT layout not allowed.");
  }

  cublas_gemm(inputA, inputB, outputD, biasTensor, outputGelu, m, n, k, lda, ldb, ldd,
              (transa) ? CUBLAS_OP_T : CUBLAS_OP_N, (transb) ? CUBLAS_OP_T : CUBLAS_OP_N, grad,
              wspace->data.dptr, wspace->data.shape[0], accumulate, use_split_accumulator,
              math_sm_count, m_split, n_split, gemm_producer, inputCounter, stream);
}
