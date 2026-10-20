/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cuda.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <limits>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/handle_manager.h"
#include "../util/logging.h"
#include "./config.h"

namespace {

using transformer_engine::DIVUP_TO_MULTIPLE;
using transformer_engine::SimpleTensor;

inline void CreateCublasHandle(cublasLtHandle_t *handle) {
  NVTE_CHECK_CUBLAS(cublasLtCreate(handle));
}

using cublasHandleManager =
    transformer_engine::detail::HandleManager<cublasLtHandle_t, CreateCublasHandle>;

uint32_t GetAlignment(uintptr_t address) {
  // Alignment in bytes
  uint32_t alignment = 256;
  while (address % alignment != 0) {
    alignment /= 2;
  }
  return alignment;
}

/* RAII wrapper for a cuBLASLt descriptor
 *
 * Creates the descriptor on construction and destroys it on
 * destruction, so descriptors are not leaked if an error is thrown.
 */
template <typename Handle, auto Create, auto Destroy>
class CublasLtDescriptor {
 public:
  template <typename... Args>
  explicit CublasLtDescriptor(Args... args) {
    NVTE_CHECK_CUBLAS(Create(&handle_, args...));
  }
  ~CublasLtDescriptor() {
    // Cleanup must not throw while unwinding an earlier error.
    if (handle_ != nullptr) {
      static_cast<void>(Destroy(handle_));
    }
  }
  CublasLtDescriptor(CublasLtDescriptor &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  CublasLtDescriptor(const CublasLtDescriptor &) = delete;
  CublasLtDescriptor &operator=(const CublasLtDescriptor &) = delete;
  CublasLtDescriptor &operator=(CublasLtDescriptor &&) = delete;

  Handle get() const { return handle_; }

 private:
  Handle handle_ = nullptr;
};

using CublasLtMatmulDesc =
    CublasLtDescriptor<cublasLtMatmulDesc_t, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy>;
using CublasLtMatrixLayout = CublasLtDescriptor<cublasLtMatrixLayout_t, cublasLtMatrixLayoutCreate,
                                                cublasLtMatrixLayoutDestroy>;
using CublasLtMatmulPreference =
    CublasLtDescriptor<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceCreate,
                       cublasLtMatmulPreferenceDestroy>;

/* Matrix batch in cuBLAS strided-batch layout
 *
 * scale_inv is only set for MXFP8 operands. It holds E8M0 scales
 * that are packed by batch and GEMM-swizzled.
 */
struct StridedBatchOperand {
  const SimpleTensor *data = nullptr;
  const SimpleTensor *scale_inv = nullptr;
  int64_t ld = 0;
  int64_t batch_stride = 0;
};

void CheckStridedBatchBuffer(const StridedBatchOperand &operand, int64_t rows, int64_t cols,
                             int64_t batch_count, const char *name) {
  NVTE_CHECK(operand.data->dptr != nullptr, "Strided batched GEMM ", name,
             " buffer is not allocated.");
  NVTE_CHECK(operand.ld >= rows, "Strided batched GEMM ", name, " leading dimension ", operand.ld,
             " is smaller than the matrix row count ", rows, ".");
  const auto required_span = static_cast<size_t>((batch_count - 1) * operand.batch_stride +
                                                 (cols - 1) * operand.ld + rows);
  NVTE_CHECK(required_span <= operand.data->numel(), "Strided batched GEMM ", name,
             " buffer is too small for the requested layout (required span=", required_span,
             " elements, available=", operand.data->numel(), ").");
}

/* Check size of packed MXFP8 scales
 *
 * For each batch, the matrix is interpreted as row-major with shape
 * [rows, features]. Row-wise scales are scaled along the feature
 * dim and column-wise scales are scaled along the row dim.
 */
void CheckMXFP8BatchScaleBuffer(const SimpleTensor &buffer, bool rowwise, size_t rows,
                                size_t features, size_t batch_count, const char *name) {
  constexpr size_t block_size = 32;
  NVTE_CHECK(buffer.dptr != nullptr, "MXFP8 strided batched GEMM ", name,
             " scales are not allocated.");
  NVTE_CHECK(buffer.dtype == transformer_engine::DType::kFloat8E8M0, "MXFP8 strided batched GEMM ",
             name, " scales must have E8M0 dtype (got ",
             transformer_engine::to_string(buffer.dtype), ").");
  size_t scale_rows = 0;
  size_t scale_cols = 0;
  if (rowwise) {
    NVTE_CHECK(features % block_size == 0, "MXFP8 strided batched GEMM ", name,
               " row-wise scale feature count must be divisible by ", block_size, " (got ",
               features, ").");
    scale_rows = DIVUP_TO_MULTIPLE(rows, 128);
    scale_cols = DIVUP_TO_MULTIPLE(features / block_size, 4);
  } else {
    NVTE_CHECK(rows % block_size == 0, "MXFP8 strided batched GEMM ", name,
               " column-wise scale row count must be divisible by ", block_size, " (got ", rows,
               ").");
    scale_rows = DIVUP_TO_MULTIPLE(rows / block_size, 4);
    scale_cols = DIVUP_TO_MULTIPLE(features, 128);
  }
  const size_t expected_numel = batch_count * scale_rows * scale_cols;
  NVTE_CHECK(buffer.numel() == expected_numel, "MXFP8 strided batched GEMM ", name,
             " packed scale buffer has invalid size (expected ", expected_numel, ", got ",
             buffer.numel(), ").");
}

void strided_batched_gemm(const StridedBatchOperand &A, cublasOperation_t transa,
                          const StridedBatchOperand &B, cublasOperation_t transb,
                          const StridedBatchOperand &C, const StridedBatchOperand &D,
                          const float *alpha, const float *beta, int64_t batch_count, int64_t m,
                          int64_t n, int64_t k, const NVTETensor workspace,
                          const NVTEMatmulConfig config, cudaStream_t stream) {
  using namespace transformer_engine;

  // Workspace
  void *workspace_ptr = nullptr;
  size_t workspace_size = 0;
  if (const Tensor *workspace_tensor = convertNVTETensor(workspace); workspace_tensor != nullptr) {
    workspace_ptr = workspace_tensor->data.dptr;
    workspace_size =
        get_buffer_size_bytes(workspace_tensor->data.numel(), workspace_tensor->data.dtype);
  }

  // Additional config
  MatmulConfig config_;
  if (config != nullptr) {
    config_ = *reinterpret_cast<const MatmulConfig *>(config);
  }
  NVTE_CHECK(config_.bias_tensor == nullptr && config_.dbias_tensor == nullptr &&
                 !config_.with_gelu_epilogue && !config_.with_dgelu_epilogue &&
                 config_.epilogue_aux_tensor == nullptr,
             "Strided batched GEMM does not support fused epilogues yet.");

  // Check dims
  NVTE_CHECK(batch_count >= 0, "Strided batched GEMM got negative batch_count=", batch_count);
  NVTE_CHECK(m >= 0 && n >= 0 && k >= 0, "Strided batched GEMM got invalid dims m=", m, ", n=", n,
             ", k=", k);
  if (batch_count == 0 || m == 0 || n == 0) {
    return;
  }
  NVTE_CHECK(k > 0, "Strided batched GEMM got invalid dim k=", k);
  auto checked_int = [](int64_t value, const char *name) {
    NVTE_CHECK(value > 0, name, " must be positive, got ", value);
    NVTE_CHECK(value <= static_cast<int64_t>(std::numeric_limits<int>::max()), name,
               " is too large for cuBLASLt: ", value);
    return static_cast<int>(value);
  };
  const int m_int = checked_int(m, "m");
  const int n_int = checked_int(n, "n");
  const int k_int = checked_int(k, "k");
  const int batch_count_int = checked_int(batch_count, "batch_count");
  NVTE_CHECK(
      A.batch_stride >= 0 && B.batch_stride >= 0 && C.batch_stride >= 0 && D.batch_stride >= 0,
      "Strided batched GEMM requires non-negative batch strides (got stridea=", A.batch_stride,
      ", strideb=", B.batch_stride, ", stridec=", C.batch_stride, ", strided=", D.batch_stride,
      ").");

  // Check buffers
  const bool use_mxfp8 = A.scale_inv != nullptr;
  const int64_t a_rows = transa == CUBLAS_OP_N ? m : k;
  const int64_t a_cols = transa == CUBLAS_OP_N ? k : m;
  const int64_t b_rows = transb == CUBLAS_OP_N ? k : n;
  const int64_t b_cols = transb == CUBLAS_OP_N ? n : k;
  CheckStridedBatchBuffer(A, a_rows, a_cols, batch_count, "A");
  CheckStridedBatchBuffer(B, b_rows, b_cols, batch_count, "B");
  CheckStridedBatchBuffer(D, m, n, batch_count, "D");
  CheckStridedBatchBuffer(C, m, n, batch_count, "C");
  NVTE_CHECK(is_high_precision_dtype(D.data->dtype),
             "Strided batched GEMM currently supports high-precision output only.");
  NVTE_CHECK(C.data->dtype == D.data->dtype,
             "Strided batched GEMM currently requires C and D to have the same dtype.");
  if (use_mxfp8) {
#if CUBLAS_VERSION >= 120800
    NVTE_CHECK(cuda::cublas_version() >= 120800,
               "MXFP8 strided batched GEMM requires cuBLAS 12.8+, but run-time cuBLAS version is ",
               cuda::cublas_version());
#else
    NVTE_ERROR(
        "MXFP8 strided batched GEMM requires cuBLAS 12.8+, but compile-time cuBLAS version is ",
        CUBLAS_VERSION);
#endif  // CUBLAS_VERSION >= 120800
    NVTE_CHECK(is_fp8_dtype(A.data->dtype) && is_fp8_dtype(B.data->dtype),
               "MXFP8 strided batched GEMM expects FP8 A and B data (got A dtype=",
               to_string(A.data->dtype), ", B dtype=", to_string(B.data->dtype), ").");
    // MXFP8 scales are always along the reduction dim. Interpreted in
    // row-major order, they are row-wise scales for TN layouts.
    const bool a_rowwise = transa == CUBLAS_OP_T;
    const bool b_rowwise = transb == CUBLAS_OP_N;
    CheckMXFP8BatchScaleBuffer(*A.scale_inv, a_rowwise, a_rowwise ? m : k, a_rowwise ? k : m,
                               batch_count, "A");
    CheckMXFP8BatchScaleBuffer(*B.scale_inv, b_rowwise, b_rowwise ? n : k, b_rowwise ? k : n,
                               batch_count, "B");
  } else {
    NVTE_CHECK(is_high_precision_dtype(A.data->dtype) && is_high_precision_dtype(B.data->dtype),
               "Strided batched GEMM expects high-precision A and B data (got A dtype=",
               to_string(A.data->dtype), ", B dtype=", to_string(B.data->dtype), ").");
  }

  // Matrix layouts
  const cudaDataType_t A_type = get_cuda_dtype(A.data->dtype);
  const cudaDataType_t B_type = get_cuda_dtype(B.data->dtype);
  const cudaDataType_t C_type = get_cuda_dtype(C.data->dtype);
  const cudaDataType_t D_type = get_cuda_dtype(D.data->dtype);
  auto make_layout = [batch_count_int, &checked_int](cudaDataType_t dtype, int64_t rows,
                                                     int64_t cols, const StridedBatchOperand &op,
                                                     const char *name) {
    CublasLtMatrixLayout desc(dtype, rows, cols, checked_int(op.ld, name));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.get(), CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count_int, sizeof(batch_count_int)));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(desc.get(),
                                                       CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                       &op.batch_stride, sizeof(op.batch_stride)));
    return desc;
  };
  const auto Adesc = make_layout(A_type, a_rows, a_cols, A, "lda");
  const auto Bdesc = make_layout(B_type, b_rows, b_cols, B, "ldb");
  const auto Cdesc = make_layout(C_type, m_int, n_int, C, "ldc");
  const auto Ddesc = make_layout(D_type, m_int, n_int, D, "ldd");

  // Matmul descriptor
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  if (A_type == CUDA_R_32F && B_type == CUDA_R_32F && D_type == CUDA_R_32F) {
    gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  const CublasLtMatmulDesc operationDesc(gemm_compute_type, CUDA_R_32F);
  auto set_matmul_attr = [&operationDesc](cublasLtMatmulDescAttributes_t attr, const auto &value) {
    NVTE_CHECK_CUBLAS(
        cublasLtMatmulDescSetAttribute(operationDesc.get(), attr, &value, sizeof(value)));
  };
  set_matmul_attr(CUBLASLT_MATMUL_DESC_TRANSA, transa);
  set_matmul_attr(CUBLASLT_MATMUL_DESC_TRANSB, transb);
  set_matmul_attr(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_DEFAULT);
  if (config_.sm_count != 0) {
    set_matmul_attr(CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, config_.sm_count);
  }
  if (use_mxfp8) {
#if CUBLAS_VERSION >= 120800
    const int8_t fast_accum_mode = config_.use_split_accumulator ? 0 : 1;
    set_matmul_attr(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum_mode);
    const void *A_scale_inv = A.scale_inv->dptr;
    const void *B_scale_inv = B.scale_inv->dptr;
    set_matmul_attr(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, A_scale_inv);
    set_matmul_attr(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, B_scale_inv);
    const cublasLtMatmulMatrixScale_t scaling_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    set_matmul_attr(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, scaling_mode);
    set_matmul_attr(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, scaling_mode);
    // Workaround for the same heuristic-cache issue handled in the non-batched MXFP8 GEMM.
    if (cuda::cublas_version() <= 120803) {
      const int64_t dummy_a_vec_stride = 1;
      set_matmul_attr(CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE, dummy_a_vec_stride);
    }
#endif  // CUBLAS_VERSION >= 120800
  }

  // Align workspace to 256 bytes
  uint8_t *aligned_workspace_ptr = nullptr;
  if (workspace_size > 0) {
    NVTE_CHECK(workspace_ptr != nullptr,
               "Strided batched GEMM got non-zero workspace size with null workspace pointer.");
    constexpr uintptr_t required_alignment = 256;
    const uintptr_t workspace_addr = reinterpret_cast<uintptr_t>(workspace_ptr);
    const uintptr_t aligned_addr =
        (workspace_addr + required_alignment - 1) & ~(required_alignment - 1);
    const size_t padding = aligned_addr - workspace_addr;
    NVTE_CHECK(workspace_size >= padding, "Workspace is too small to align to 256 bytes.");
    aligned_workspace_ptr = reinterpret_cast<uint8_t *>(aligned_addr);
    workspace_size -= padding;
  }

  // Algorithm preferences
  const CublasLtMatmulPreference preference;
  auto set_preference_attr = [&preference](cublasLtMatmulPreferenceAttributes_t attr,
                                           const auto &value) {
    NVTE_CHECK_CUBLAS(
        cublasLtMatmulPreferenceSetAttribute(preference.get(), attr, &value, sizeof(value)));
  };
  // Keep split-K partial sums in the compute type, matching the ordinary GEMM path.
  const uint32_t reduction_scheme_mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  set_preference_attr(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, reduction_scheme_mask);
  set_preference_attr(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_size);
  set_preference_attr(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                      GetAlignment(reinterpret_cast<uintptr_t>(A.data->dptr)));
  set_preference_attr(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                      GetAlignment(reinterpret_cast<uintptr_t>(B.data->dptr)));
  set_preference_attr(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                      GetAlignment(reinterpret_cast<uintptr_t>(C.data->dptr)));
  set_preference_attr(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                      GetAlignment(reinterpret_cast<uintptr_t>(D.data->dptr)));

  // Choose algorithm
  cublasLtHandle_t handle = cublasHandleManager::Instance().GetHandle();
  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  const auto status = cublasLtMatmulAlgoGetHeuristic(
      handle, operationDesc.get(), Adesc.get(), Bdesc.get(), Cdesc.get(), Ddesc.get(),
      preference.get(), 1, &heuristic_result, &returned_results);
  NVTE_CHECK(status != CUBLAS_STATUS_NOT_SUPPORTED,
             "Unable to find suitable cuBLAS strided batched GEMM algorithm");
  NVTE_CHECK_CUBLAS(status);
  NVTE_CHECK(returned_results != 0, "Unable to find any suitable strided batched GEMM algorithms");

  // Launch GEMM
  NVTE_CHECK_CUBLAS(cublasLtMatmul(handle, operationDesc.get(), alpha, A.data->dptr, Adesc.get(),
                                   B.data->dptr, Bdesc.get(), beta, C.data->dptr, Cdesc.get(),
                                   D.data->dptr, Ddesc.get(), &heuristic_result.algo,
                                   aligned_workspace_ptr, workspace_size, stream));
}

}  // namespace

void nvte_cublas_gemm_strided_batched(int transa, int transb, const float *alpha,
                                      const NVTETensor A, int64_t lda, int64_t stridea,
                                      const NVTETensor B, int64_t ldb, int64_t strideb,
                                      const float *beta, const NVTETensor C, int64_t ldc,
                                      int64_t stridec, NVTETensor D, int64_t ldd, int64_t strided,
                                      int64_t batch_count, int64_t m, int64_t n, int64_t k,
                                      NVTETensor workspace, NVTEMatmulConfig config,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_gemm_strided_batched);
  using namespace transformer_engine;
  const Tensor *A_tensor = convertNVTETensorCheck(A);
  const Tensor *B_tensor = convertNVTETensorCheck(B);
  const Tensor *C_tensor = convertNVTETensorCheck(C);
  const Tensor *D_tensor = convertNVTETensorCheck(D);
  NVTE_CHECK(is_tensor_scaling(A_tensor->scaling_mode) && is_tensor_scaling(B_tensor->scaling_mode),
             "nvte_cublas_gemm_strided_batched supports only high-precision A and B. "
             "Use nvte_cublas_gemm_strided_batched_mxfp8 for MXFP8.");
  strided_batched_gemm({&A_tensor->data, nullptr, lda, stridea}, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                       {&B_tensor->data, nullptr, ldb, strideb}, transb ? CUBLAS_OP_T : CUBLAS_OP_N,
                       {&C_tensor->data, nullptr, ldc, stridec},
                       {&D_tensor->data, nullptr, ldd, strided}, alpha, beta, batch_count, m, n, k,
                       workspace, config, stream);
}

void nvte_cublas_gemm_strided_batched_mxfp8(int transa, int transb, const float *alpha,
                                            const NVTETensor A_data, const NVTETensor A_scale_inv,
                                            int64_t lda, int64_t stridea, const NVTETensor B_data,
                                            const NVTETensor B_scale_inv, int64_t ldb,
                                            int64_t strideb, const float *beta, const NVTETensor C,
                                            int64_t ldc, int64_t stridec, NVTETensor D, int64_t ldd,
                                            int64_t strided, int64_t batch_count, int64_t m,
                                            int64_t n, int64_t k, NVTETensor workspace,
                                            NVTEMatmulConfig config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_gemm_strided_batched_mxfp8);
  using namespace transformer_engine;
  const Tensor *A_data_tensor = convertNVTETensorCheck(A_data);
  const Tensor *A_scale_tensor = convertNVTETensorCheck(A_scale_inv);
  const Tensor *B_data_tensor = convertNVTETensorCheck(B_data);
  const Tensor *B_scale_tensor = convertNVTETensorCheck(B_scale_inv);
  const Tensor *C_tensor = convertNVTETensorCheck(C);
  const Tensor *D_tensor = convertNVTETensorCheck(D);
  strided_batched_gemm({&A_data_tensor->data, &A_scale_tensor->data, lda, stridea},
                       transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                       {&B_data_tensor->data, &B_scale_tensor->data, ldb, strideb},
                       transb ? CUBLAS_OP_T : CUBLAS_OP_N, {&C_tensor->data, nullptr, ldc, stridec},
                       {&D_tensor->data, nullptr, ldd, strided}, alpha, beta, batch_count, m, n, k,
                       workspace, config, stream);
}
