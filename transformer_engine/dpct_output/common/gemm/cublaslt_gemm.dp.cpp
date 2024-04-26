#define DPCT_COMPAT_RT_VERSION 12010
/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <transformer_engine/gemm.h>
#include <dpct/blas_utils.hpp>

#include <transformer_engine/transformer_engine.h>
#include "../common.h"
#include "../util/logging.h"
#include <dpct/lib_common_utils.hpp>

namespace {

dpct::library_data_t get_cuda_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return dpct::library_data_t::real_half;
    case DType::kFloat32:
      return dpct::library_data_t::real_float;
    case DType::kBFloat16:
      return dpct::library_data_t::real_bfloat16;
    case DType::kFloat8E4M3:
      return 28;
    case DType::kFloat8E5M2:
      return 29;
    default:
      NVTE_ERROR("Invalid type");
  }
}

}  // namespace

namespace transformer_engine {

void cublas_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                 const Tensor *inputBias, Tensor *outputPreGelu, int m, int n,
                 int k, int lda, int ldb, int ldd,
                 oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 bool grad, void *workspace, size_t workspaceSize,
                 bool accumulate, bool use_split_accumulator, int math_sm_count,
                 int m_split, int n_split, bool gemm_producer,
                 const Tensor *inputCounter, dpct::queue_ptr stream) try {
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
  const bool use_fp8 = is_fp8_dtype(inputA->data.dtype) ||
                       is_fp8_dtype(inputB->data.dtype);
  const dpct::library_data_t A_type = get_cuda_dtype(inputA->data.dtype);
  const dpct::library_data_t B_type = get_cuda_dtype(inputB->data.dtype);
  const dpct::library_data_t D_type = get_cuda_dtype(outputD->data.dtype);
  const dpct::library_data_t bias_type = get_cuda_dtype(inputBias->data.dtype);

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
    NVTE_CHECK(!accumulate,
             "Accumulation mode not supported with FP8 GEMM output!");
  }

  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  cublasLtHandle_t handle;
  /*
  DPCT1009:388: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:387: Migration of cublasLtCreate is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t       operationDesc = nullptr;
  cublasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int                             returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t) ldd;

  // Use TF32 only for pure FP32 GEMM.
  dpct::library_data_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  if (A_type == dpct::library_data_t::real_float &&
      B_type == dpct::library_data_t::real_float &&
      D_type == dpct::library_data_t::real_float) {
    gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  /*
  DPCT1009:390: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:389: Migration of cublasLtMatrixLayoutCreate is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &Adesc, A_type, transa == oneapi::mkl::transpose::nontrans ? m : k,
      transa == oneapi::mkl::transpose::nontrans ? k : m, lda));
  /*
  DPCT1009:392: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:391: Migration of cublasLtMatrixLayoutCreate is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &Bdesc, B_type, transb == oneapi::mkl::transpose::nontrans ? k : n,
      transb == oneapi::mkl::transpose::nontrans ? n : k, ldb));
  /*
  DPCT1009:394: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:393: Migration of cublasLtMatrixLayoutCreate is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

  /*
  DPCT1009:396: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:395: Migration of cublasLtMatmulDescCreate is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type,
                                             dpct::library_data_t::real_float));
  /*
  DPCT1009:398: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:397: Migration of cublasLtMatmulDescSetAttribute is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  /*
  DPCT1009:400: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:399: Migration of cublasLtMatmulDescSetAttribute is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  // Set math SM count
  if (math_sm_count != 0) {
      /*
      DPCT1009:402: SYCL uses exceptions to report errors and does not use the
      error codes. The original code was commented out and a warning string was
      inserted. You need to rewrite this code.
      */
      /*
      DPCT1007:401: Migration of cublasLtMatmulDescSetAttribute is not
      supported.
      */
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sm_count,
          sizeof(math_sm_count)));
  }


  // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
  // Note: gelu fusion isn't available right now, and we don't need
  // amax(D) either (next op is high precision).
  if (use_fp8) {
    // Split accumulator.
    const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
    /*
    DPCT1009:404: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:403: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode,
        sizeof(fastAccuMode)));
    /*
    DPCT1009:406: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:405: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &A_scale_inverse,
        sizeof(A_scale_inverse)));
    /*
    DPCT1009:408: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:407: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &B_scale_inverse,
        sizeof(B_scale_inverse)));
    if (is_fp8_dtype(outputD->data.dtype)) {
      // Accumulation mode not supported for FP8 output
      C = nullptr;
      /*
      DPCT1009:410: SYCL uses exceptions to report errors and does not use the
      error codes. The original code was commented out and a warning string was
      inserted. You need to rewrite this code.
      */
      /*
      DPCT1007:409: Migration of cublasLtMatmulDescSetAttribute is not
      supported.
      */
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &D_scale,
          sizeof(D_scale)));
      /*
      DPCT1009:412: SYCL uses exceptions to report errors and does not use the
      error codes. The original code was commented out and a warning string was
      inserted. You need to rewrite this code.
      */
      /*
      DPCT1007:411: Migration of cublasLtMatmulDescSetAttribute is not
      supported.
      */
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &D_amax,
          sizeof(D_amax)));
      // For FP8 output, cuBLAS requires C_type to be same as bias_type
      /*
      DPCT1009:414: SYCL uses exceptions to report errors and does not use the
      error codes. The original code was commented out and a warning string was
      inserted. You need to rewrite this code.
      */
      /*
      DPCT1007:413: Migration of cublasLtMatrixLayoutCreate is not supported.
      */
      NVTE_CHECK_CUBLAS(
          cublasLtMatrixLayoutCreate(&Cdesc, bias_type, m, n, ldd));
    } else {
      /*
      DPCT1009:416: SYCL uses exceptions to report errors and does not use the
      error codes. The original code was commented out and a warning string was
      inserted. You need to rewrite this code.
      */
      /*
      DPCT1007:415: Migration of cublasLtMatrixLayoutCreate is not supported.
      */
      NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
    }
    if (bias) {
      /*
      DPCT1009:418: SYCL uses exceptions to report errors and does not use the
      error codes. The original code was commented out and a warning string was
      inserted. You need to rewrite this code.
      */
      /*
      DPCT1007:417: Migration of cublasLtMatmulDescSetAttribute is not
      supported.
      */
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
          sizeof(bias_type)));
    }
  } else {
    /*
    DPCT1009:420: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:419: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
  }

  if (bias && gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }
    /*
    DPCT1009:422: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:421: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
        sizeof(bias_ptr)));
    /*
    DPCT1009:424: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:423: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu_out,
        sizeof(pre_gelu_out)));
    /*
    DPCT1009:426: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:425: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld_gelumat,
        sizeof(ld_gelumat)));
    const dpct::library_data_t aux_type =
        get_cuda_dtype(outputPreGelu->data.dtype);
    /*
    DPCT1009:428: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:427: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &aux_type,
        sizeof(aux_type)));
  } else if (bias) {
    if (grad) {
      // grad output is always input B
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
    } else {
      epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    /*
    DPCT1009:430: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:429: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
        sizeof(bias_ptr)));
  } else if (gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
    }
    /*
    DPCT1009:432: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:431: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu_out,
        sizeof(pre_gelu_out)));
    /*
    DPCT1009:434: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1007:433: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld_gelumat,
        sizeof(ld_gelumat)));
  }

  /*
  DPCT1009:436: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:435: Migration of cublasLtMatmulDescSetAttribute is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
      sizeof(epilogue)));
#if DPCT_COMPAT_RT_VERSION >= 12020 && CUBLAS_VERSION >= 120205
  if (counter != nullptr) {
    if (m_split == 0) m_split=1;
    if (n_split == 0) n_split=1;
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
       operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS,
       &m_split, sizeof(m_split)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
       operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS,
       &n_split, sizeof(n_split)));
    if (gemm_producer) {
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER,
        &counter, sizeof(counter)));
    } else {
      NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER,
        &counter, sizeof(counter)));
    }
  }
#endif

  /*
  DPCT1009:438: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:437: Migration of cublasLtMatmulPreferenceCreate is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  /*
  DPCT1009:440: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:439: Migration of cublasLtMatmulPreferenceSetAttribute is not
  supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
      sizeof(workspaceSize)));

  /*
  DPCT1007:441: Migration of cublasLtMatmulAlgoGetHeuristic is not supported.
  */
  const auto status = cublasLtMatmulAlgoGetHeuristic(
      handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
      &heuristicResult, &returnedResults);
  NVTE_CHECK(status != 15, "Unable to find suitable cuBLAS GEMM algorithm");
  /*
  DPCT1009:442: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  NVTE_CHECK_CUBLAS(status);

  if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");

  // D = alpha * (A * B) + beta * C

  /*
  DPCT1009:444: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:443: Migration of cublasLtMatmul is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmul(
      handle, operationDesc, static_cast<const void *>(&one), /* alpha */
      A,                                                      /* A */
      Adesc, B,                                               /* B */
      Bdesc, static_cast<const void *>(&beta),                /* beta */
      C,                                                      /* C */
      Cdesc, D,                                               /* D */
      Ddesc, &heuristicResult.algo,                           /* algo */
      workspace,                                              /* workspace */
      workspaceSize, stream));                                /* stream */

  /*
  DPCT1009:446: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:445: Migration of cublasLtMatmulPreferenceDestroy is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  /*
  DPCT1009:448: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:447: Migration of cublasLtMatrixLayoutDestroy is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
  /*
  DPCT1009:450: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:449: Migration of cublasLtMatrixLayoutDestroy is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  /*
  DPCT1009:452: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:451: Migration of cublasLtMatrixLayoutDestroy is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  /*
  DPCT1009:454: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:453: Migration of cublasLtMatrixLayoutDestroy is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  /*
  DPCT1009:456: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1007:455: Migration of cublasLtMatmulDescDestroy is not supported.
  */
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace transformer_engine

void nvte_cublas_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D,
                      const NVTETensor bias, NVTETensor pre_gelu_out,
                      bool transa, bool transb, bool grad, NVTETensor workspace,
                      bool accumulate, bool use_split_accumulator,
                      int math_sm_count, dpct::queue_ptr stream) {
  NVTE_API_CALL(nvte_cublas_gemm);
  using namespace transformer_engine;
  const Tensor *inputA = reinterpret_cast<const Tensor*>(A);
  const Tensor *inputB = reinterpret_cast<const Tensor*>(B);
  Tensor *outputD = reinterpret_cast<Tensor*>(D);
  const Tensor *biasTensor = reinterpret_cast<const Tensor*>(bias);
  Tensor *outputGelu = reinterpret_cast<Tensor*>(pre_gelu_out);
  Tensor *wspace = reinterpret_cast<Tensor*>(workspace);

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

  cublas_gemm(
      inputA, inputB, outputD, biasTensor, outputGelu, m, n, k, lda, ldb, ldd,
      (transa) ? oneapi::mkl::transpose::trans
               : oneapi::mkl::transpose::nontrans,
      (transb) ? oneapi::mkl::transpose::trans
               : oneapi::mkl::transpose::nontrans,
      grad, wspace->data.dptr, wspace->data.shape[0], accumulate,
      use_split_accumulator, math_sm_count, 0, 0, false, nullptr, stream);
}

void nvte_cublas_atomic_gemm(const NVTETensor A, const NVTETensor B,
                             NVTETensor D, const NVTETensor bias,
                             NVTETensor pre_gelu_out, bool transa, bool transb,
                             bool grad, NVTETensor workspace, bool accumulate,
                             bool use_split_accumulator, int math_sm_count,
                             int m_split, int n_split, bool gemm_producer,
                             const NVTETensor counter,
                             dpct::queue_ptr stream) try {
  NVTE_API_CALL(nvte_cublas_atomic_gemm);

  int cudart_version;
  /*
  DPCT1009:458: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  /*
  DPCT1043:457: The version-related API is different in SYCL. An initial code
  was generated, but you need to adjust it.
  */
  NVTE_CHECK_CUDA(DPCT_CHECK_ERROR(
      cudart_version = dpct::get_major_version(dpct::get_current_device())));
  NVTE_CHECK(cudart_version >= 12020, "Cuda version 12.2 is required for atomic gemm.");
  /*
  DPCT1007:459: Migration of cublasLtGetVersion is not supported.
  */
  NVTE_CHECK(cublasLtGetVersion() >= 120205,
             "Cublas version 12.2.5 is required for atomic gemm.");

  using namespace transformer_engine;
  const Tensor *inputA = reinterpret_cast<const Tensor*>(A);
  const Tensor *inputB = reinterpret_cast<const Tensor*>(B);
  Tensor *outputD = reinterpret_cast<Tensor*>(D);
  const Tensor *biasTensor = reinterpret_cast<const Tensor*>(bias);
  Tensor *outputGelu = reinterpret_cast<Tensor*>(pre_gelu_out);
  const Tensor *inputCounter = reinterpret_cast<const Tensor*>(counter);
  Tensor *wspace = reinterpret_cast<Tensor*>(workspace);

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

  cublas_gemm(inputA, inputB, outputD, biasTensor, outputGelu, m, n, k, lda,
              ldb, ldd,
              (transa) ? oneapi::mkl::transpose::trans
                       : oneapi::mkl::transpose::nontrans,
              (transb) ? oneapi::mkl::transpose::trans
                       : oneapi::mkl::transpose::nontrans,
              grad, wspace->data.dptr, wspace->data.shape[0], accumulate,
              use_split_accumulator, math_sm_count, m_split, n_split,
              gemm_producer, inputCounter, stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
