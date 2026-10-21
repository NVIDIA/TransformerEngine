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

  operator Handle() const { return handle_; }

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

/* Batch of matrices in a tensor
 *
 * The tensor is flattened to 3D based on its batch dim: [G, ..., C]
 * is interpreted as [G, R, C] and [..., G, C] as [R, G, C]. Each
 * batch entry is a row-major [R, C] matrix, i.e. a column-major
 * C x R matrix in cuBLAS terms.
 *
 * scale_inv is only set for MXFP8 operands. It holds E8M0 scales
 * that are packed by batch and GEMM-swizzled.
 */
struct BatchedMatrix {
  const SimpleTensor *data = nullptr;
  const SimpleTensor *scale_inv = nullptr;
  size_t batch_size = 0;
  size_t rows = 0;
  size_t cols = 0;
  int64_t ld = 0;
  int64_t batch_stride = 0;
};

BatchedMatrix MakeBatchedMatrix(const SimpleTensor &data, int batch_dim, const char *name,
                                const SimpleTensor *scale_inv = nullptr) {
  const auto &shape = data.shape;
  const int ndim = static_cast<int>(shape.size());
  NVTE_CHECK(ndim >= 2, "Strided batched GEMM expects ", name, " to have at least 2 dims (got ",
             ndim, ").");
  NVTE_CHECK(batch_dim == 0 || batch_dim == -2, "Strided batched GEMM supports batch_dim=0 or ",
             "batch_dim=-2 (got ", batch_dim, " for ", name, ").");
  NVTE_CHECK(data.dptr != nullptr || data.numel() == 0, "Strided batched GEMM ", name,
             " is not allocated.");
  BatchedMatrix out;
  out.data = &data;
  out.scale_inv = scale_inv;
  out.batch_size = shape[batch_dim < 0 ? ndim + batch_dim : batch_dim];
  out.cols = shape.back();
  out.rows = out.batch_size * out.cols == 0 ? 0 : data.numel() / (out.batch_size * out.cols);
  if (batch_dim == 0) {
    out.ld = out.cols;
    out.batch_stride = out.rows * out.cols;
  } else {
    out.ld = out.batch_size * out.cols;
    out.batch_stride = out.cols;
  }
  return out;
}

/* Check size of packed MXFP8 scales
 *
 * Row-wise scales are scaled along the columns of the row-major
 * matrix and column-wise scales are scaled along its rows.
 */
void CheckMXFP8BatchScaleBuffer(const BatchedMatrix &matrix, bool rowwise, const char *name) {
  constexpr size_t block_size = 32;
  const SimpleTensor &buffer = *matrix.scale_inv;
  NVTE_CHECK(buffer.dptr != nullptr, "MXFP8 strided batched GEMM ", name,
             " scales are not allocated.");
  NVTE_CHECK(buffer.dtype == transformer_engine::DType::kFloat8E8M0, "MXFP8 strided batched GEMM ",
             name, " scales must have E8M0 dtype (got ",
             transformer_engine::to_string(buffer.dtype), ").");
  size_t scale_rows = 0;
  size_t scale_cols = 0;
  if (rowwise) {
    NVTE_CHECK(matrix.cols % block_size == 0, "MXFP8 strided batched GEMM ", name,
               " row-wise scaling requires columns divisible by ", block_size, " (got ",
               matrix.cols, ").");
    scale_rows = DIVUP_TO_MULTIPLE(matrix.rows, 128);
    scale_cols = DIVUP_TO_MULTIPLE(matrix.cols / block_size, 4);
  } else {
    NVTE_CHECK(matrix.rows % block_size == 0, "MXFP8 strided batched GEMM ", name,
               " column-wise scaling requires rows divisible by ", block_size, " (got ",
               matrix.rows, ").");
    scale_rows = DIVUP_TO_MULTIPLE(matrix.rows / block_size, 4);
    scale_cols = DIVUP_TO_MULTIPLE(matrix.cols, 128);
  }
  const size_t expected_numel = matrix.batch_size * scale_rows * scale_cols;
  NVTE_CHECK(buffer.numel() == expected_numel, "MXFP8 strided batched GEMM ", name,
             " packed scale buffer has invalid size (expected ", expected_numel, ", got ",
             buffer.numel(), ").");
}

/* Strided batched GEMM
 *
 * With matrices interpreted in row-major order, computes for each
 * batch entry:
 *   D_i = alpha * op(B_i) * op(A_i) + beta * C_i
 * This matches the column-major convention of nvte_cublas_gemm.
 */
void strided_batched_gemm(const BatchedMatrix &A, cublasOperation_t transa, const BatchedMatrix &B,
                          cublasOperation_t transb, const BatchedMatrix &C, const BatchedMatrix &D,
                          const float *alpha, const float *beta, const NVTETensor workspace,
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

  // GEMM dims
  // Note: In row-major order, op(A) is k x m, op(B) is n x k, and D
  // is n x m.
  const size_t batch_size = D.batch_size;
  const size_t m = D.cols;
  const size_t n = D.rows;
  const size_t k = transa == CUBLAS_OP_N ? A.rows : A.cols;
  NVTE_CHECK(A.batch_size == batch_size && B.batch_size == batch_size && C.batch_size == batch_size,
             "Strided batched GEMM got inconsistent batch sizes (A=", A.batch_size,
             ", B=", B.batch_size, ", C=", C.batch_size, ", D=", D.batch_size, ").");
  NVTE_CHECK((transa == CUBLAS_OP_N ? A.cols : A.rows) == m &&
                 (transb == CUBLAS_OP_N ? B.cols : B.rows) == k &&
                 (transb == CUBLAS_OP_N ? B.rows : B.cols) == n,
             "Strided batched GEMM got incompatible matrix dims (A=", A.rows, "x", A.cols,
             ", transa=", transa == CUBLAS_OP_T, ", B=", B.rows, "x", B.cols,
             ", transb=", transb == CUBLAS_OP_T, ", D=", D.rows, "x", D.cols, ").");
  NVTE_CHECK(C.rows == n && C.cols == m && C.ld == D.ld && C.batch_stride == D.batch_stride,
             "Strided batched GEMM requires C and D to have the same layout.");
  if (batch_size == 0 || m == 0 || n == 0) {
    return;
  }
  NVTE_CHECK(k > 0, "Strided batched GEMM does not support k=0.");
  auto checked_int = [](int64_t value, const char *name) {
    NVTE_CHECK(value <= static_cast<int64_t>(std::numeric_limits<int>::max()), name,
               " is too large for cuBLASLt: ", value);
    return static_cast<int>(value);
  };
  const int batch_size_int = checked_int(batch_size, "batch size");

  // Check data types
  const bool use_mxfp8 = A.scale_inv != nullptr;
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
    // MXFP8 scales are always along the reduction dim
    CheckMXFP8BatchScaleBuffer(A, transa == CUBLAS_OP_T, "A");
    CheckMXFP8BatchScaleBuffer(B, transb == CUBLAS_OP_N, "B");
  } else {
    NVTE_CHECK(is_high_precision_dtype(A.data->dtype) && is_high_precision_dtype(B.data->dtype),
               "Strided batched GEMM expects high-precision A and B data (got A dtype=",
               to_string(A.data->dtype), ", B dtype=", to_string(B.data->dtype), ").");
  }

  // Matrix layouts
  auto make_layout = [batch_size_int, &checked_int](const BatchedMatrix &matrix, const char *name) {
    // Row-major [rows, cols] matrix is column-major cols x rows
    CublasLtMatrixLayout desc(get_cuda_dtype(matrix.data->dtype), matrix.cols, matrix.rows,
                              checked_int(matrix.ld, name));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                       &batch_size_int, sizeof(batch_size_int)));
    NVTE_CHECK_CUBLAS(
        cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                         &matrix.batch_stride, sizeof(matrix.batch_stride)));
    return desc;
  };
  const auto Adesc = make_layout(A, "A leading dim");
  const auto Bdesc = make_layout(B, "B leading dim");
  const auto Cdesc = make_layout(C, "C leading dim");
  const auto Ddesc = make_layout(D, "D leading dim");

  // Matmul descriptor
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  if (A.data->dtype == DType::kFloat32 && B.data->dtype == DType::kFloat32 &&
      D.data->dtype == DType::kFloat32) {
    gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  const CublasLtMatmulDesc operationDesc(gemm_compute_type, CUDA_R_32F);
  auto set_matmul_attr = [&operationDesc](cublasLtMatmulDescAttributes_t attr, const auto &value) {
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, attr, &value, sizeof(value)));
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
        cublasLtMatmulPreferenceSetAttribute(preference, attr, &value, sizeof(value)));
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
  const auto status =
      cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
                                     1, &heuristic_result, &returned_results);
  NVTE_CHECK(status != CUBLAS_STATUS_NOT_SUPPORTED,
             "Unable to find suitable cuBLAS strided batched GEMM algorithm");
  NVTE_CHECK_CUBLAS(status);
  NVTE_CHECK(returned_results != 0, "Unable to find any suitable strided batched GEMM algorithms");

  // Launch GEMM
  NVTE_CHECK_CUBLAS(cublasLtMatmul(handle, operationDesc, alpha, A.data->dptr, Adesc, B.data->dptr,
                                   Bdesc, beta, C.data->dptr, Cdesc, D.data->dptr, Ddesc,
                                   &heuristic_result.algo, aligned_workspace_ptr, workspace_size,
                                   stream));
}

}  // namespace

void nvte_cublas_gemm_strided_batched(int transa, int transb, const float *alpha,
                                      const NVTETensor A, int batch_dim_a, const NVTETensor B,
                                      int batch_dim_b, const float *beta, const NVTETensor C,
                                      NVTETensor D, int batch_dim_d, NVTETensor workspace,
                                      NVTEMatmulConfig config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_gemm_strided_batched);
  using namespace transformer_engine;
  const Tensor *A_tensor = convertNVTETensorCheck(A);
  const Tensor *B_tensor = convertNVTETensorCheck(B);
  const Tensor *C_tensor = convertNVTETensorCheck(C);
  const Tensor *D_tensor = convertNVTETensorCheck(D);
  NVTE_CHECK(is_tensor_scaling(A_tensor->scaling_mode) && is_tensor_scaling(B_tensor->scaling_mode),
             "nvte_cublas_gemm_strided_batched supports only high-precision A and B. "
             "Use nvte_cublas_gemm_strided_batched_mxfp8 for MXFP8.");
  strided_batched_gemm(
      MakeBatchedMatrix(A_tensor->data, batch_dim_a, "A"), transa ? CUBLAS_OP_T : CUBLAS_OP_N,
      MakeBatchedMatrix(B_tensor->data, batch_dim_b, "B"), transb ? CUBLAS_OP_T : CUBLAS_OP_N,
      MakeBatchedMatrix(C_tensor->data, batch_dim_d, "C"),
      MakeBatchedMatrix(D_tensor->data, batch_dim_d, "D"), alpha, beta, workspace, config, stream);
}

void nvte_cublas_gemm_strided_batched_mxfp8(int transa, int transb, const float *alpha,
                                            const NVTETensor A_data, const NVTETensor A_scale_inv,
                                            int batch_dim_a, const NVTETensor B_data,
                                            const NVTETensor B_scale_inv, int batch_dim_b,
                                            const float *beta, const NVTETensor C, NVTETensor D,
                                            int batch_dim_d, NVTETensor workspace,
                                            NVTEMatmulConfig config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_gemm_strided_batched_mxfp8);
  using namespace transformer_engine;
  const Tensor *A_data_tensor = convertNVTETensorCheck(A_data);
  const Tensor *A_scale_tensor = convertNVTETensorCheck(A_scale_inv);
  const Tensor *B_data_tensor = convertNVTETensorCheck(B_data);
  const Tensor *B_scale_tensor = convertNVTETensorCheck(B_scale_inv);
  const Tensor *C_tensor = convertNVTETensorCheck(C);
  const Tensor *D_tensor = convertNVTETensorCheck(D);
  strided_batched_gemm(
      MakeBatchedMatrix(A_data_tensor->data, batch_dim_a, "A", &A_scale_tensor->data),
      transa ? CUBLAS_OP_T : CUBLAS_OP_N,
      MakeBatchedMatrix(B_data_tensor->data, batch_dim_b, "B", &B_scale_tensor->data),
      transb ? CUBLAS_OP_T : CUBLAS_OP_N, MakeBatchedMatrix(C_tensor->data, batch_dim_d, "C"),
      MakeBatchedMatrix(D_tensor->data, batch_dim_d, "D"), alpha, beta, workspace, config, stream);
}
