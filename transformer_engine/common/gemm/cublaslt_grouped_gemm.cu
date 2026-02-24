/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "../util/cuda_runtime.h"
#include "../util/handle_manager.h"
#include "../util/logging.h"
#include "./config.h"

namespace {

inline void CreateCublasHandle(cublasLtHandle_t *handle) {
  NVTE_CHECK_CUBLAS(cublasLtCreate(handle));
}

}  // namespace

#if CUBLAS_VERSION >= 130200

namespace {

// Helper struct to pass per-tensor shape/offset info (pointer or uniform value)
struct TensorShapeInfo {
  const int64_t *first_dims;  // nullptr if uniform
  const int64_t *last_dims;   // nullptr if uniform
  const int64_t *offsets;     // nullptr if need to compute
  int64_t uniform_first;      // used if first_dims == nullptr
  int64_t uniform_last;       // used if last_dims == nullptr

  // Create from GroupedTensor
  static TensorShapeInfo from_tensor(const transformer_engine::GroupedTensor *t) {
    const bool has_first = t->first_dims.has_data();
    const bool has_last = t->last_dims.has_data();
    // When per-tensor dims are not provided, we must be in the uniform-shape case.
    NVTE_CHECK(has_first || t->all_same_first_dim(),
               "GroupedTensor is missing first_dims for varying shapes");
    NVTE_CHECK(has_last || t->all_same_last_dim(),
               "GroupedTensor is missing last_dims for varying shapes");

    const int64_t *first_ptr =
        has_first ? static_cast<const int64_t *>(t->first_dims.dptr) : nullptr;
    const int64_t *last_ptr = has_last ? static_cast<const int64_t *>(t->last_dims.dptr) : nullptr;

    const int64_t uniform_first = has_first ? 0 : static_cast<int64_t>(t->get_common_first_dim());
    const int64_t uniform_last = has_last ? 0 : static_cast<int64_t>(t->get_common_last_dim());

    return {first_ptr, last_ptr,
            t->tensor_offsets.has_data() ? static_cast<const int64_t *>(t->tensor_offsets.dptr)
                                         : nullptr,
            uniform_first, uniform_last};
  }

  // Create for C tensor (uses D's dimensions, only has offsets)
  static TensorShapeInfo create_shape_info_for_C(const transformer_engine::GroupedTensor *C,
                                                 const transformer_engine::GroupedTensor *D) {
    const bool has_first = D->first_dims.has_data();
    const bool has_last = D->last_dims.has_data();
    NVTE_CHECK(has_first || D->all_same_first_dim(),
               "GroupedTensor D is missing first_dims for varying shapes");
    NVTE_CHECK(has_last || D->all_same_last_dim(),
               "GroupedTensor D is missing last_dims for varying shapes");

    const int64_t *first_ptr =
        has_first ? static_cast<const int64_t *>(D->first_dims.dptr) : nullptr;
    const int64_t *last_ptr = has_last ? static_cast<const int64_t *>(D->last_dims.dptr) : nullptr;
    const int64_t uniform_first = has_first ? 0 : static_cast<int64_t>(D->get_common_first_dim());
    const int64_t uniform_last = has_last ? 0 : static_cast<int64_t>(D->get_common_last_dim());

    return {first_ptr, last_ptr,
            C->tensor_offsets.has_data() ? static_cast<const int64_t *>(C->tensor_offsets.dptr)
                                         : nullptr,
            uniform_first, uniform_last};
  }
};

// Helper functions to compute average dimensions from logical_shape for heuristics
// These are hints for cuBLASLt algorithm selection, don't need to be exact
inline int64_t compute_avg_first_dim(const transformer_engine::GroupedTensor *t) {
  // logical_shape[0] is either num_tensors*M (uniform) or sum_of_M (varying first)
  // In both cases, dividing by num_tensors gives the average
  return static_cast<int64_t>(t->logical_shape.data[0]) / static_cast<int64_t>(t->num_tensors);
}

inline int64_t compute_avg_last_dim(const transformer_engine::GroupedTensor *t) {
  if (t->all_same_last_dim()) {
    // logical_shape[1] is the common N
    return static_cast<int64_t>(t->logical_shape.data[1]);
  }
  // When varying, logical_shape[1] should be sum of last dims if provided; otherwise fallback to avg via division.
  return static_cast<int64_t>(t->logical_shape.data[1]) / static_cast<int64_t>(t->num_tensors);
}

// Workspace layout for grouped GEMM
struct GroupedGemmSetupWorkspace {
  void **A_ptrs;
  void **B_ptrs;
  void **C_ptrs;
  void **D_ptrs;
  float **alpha_ptrs;
  float **beta_ptrs;
  // Storage dimensions for cuBLAS matrix layouts
  int *a_rows;
  int *a_cols;
  int *b_rows;
  int *b_cols;
  int *d_rows;  // M (first dim) - also used for C
  int *d_cols;  // N (last dim) - also used for C

  // Initialize from workspace buffer
  // Layout: all pointer arrays first (8-byte aligned), then int arrays (4-byte aligned)
  static GroupedGemmSetupWorkspace from_buffers(char *setup_ws_ptr, size_t num_tensors) {
    GroupedGemmSetupWorkspace ws;
    size_t offset = 0;
    const size_t ptr_size = num_tensors * sizeof(void *);
    const size_t int_size = num_tensors * sizeof(int);

    // Pointer arrays first (all 8-byte aligned)
    ws.A_ptrs = reinterpret_cast<void **>(setup_ws_ptr + offset);
    offset += ptr_size;
    ws.B_ptrs = reinterpret_cast<void **>(setup_ws_ptr + offset);
    offset += ptr_size;
    ws.C_ptrs = reinterpret_cast<void **>(setup_ws_ptr + offset);
    offset += ptr_size;
    ws.D_ptrs = reinterpret_cast<void **>(setup_ws_ptr + offset);
    offset += ptr_size;
    ws.alpha_ptrs = reinterpret_cast<float **>(setup_ws_ptr + offset);
    offset += ptr_size;
    ws.beta_ptrs = reinterpret_cast<float **>(setup_ws_ptr + offset);
    offset += ptr_size;

    // Int arrays for storage dimensions (4-byte aligned)
    ws.a_rows = reinterpret_cast<int *>(setup_ws_ptr + offset);
    offset += int_size;
    ws.a_cols = reinterpret_cast<int *>(setup_ws_ptr + offset);
    offset += int_size;
    ws.b_rows = reinterpret_cast<int *>(setup_ws_ptr + offset);
    offset += int_size;
    ws.b_cols = reinterpret_cast<int *>(setup_ws_ptr + offset);
    offset += int_size;
    ws.d_rows = reinterpret_cast<int *>(setup_ws_ptr + offset);
    offset += int_size;
    ws.d_cols = reinterpret_cast<int *>(setup_ws_ptr + offset);

    return ws;
  }

  // Calculate required size for setup workspace
  static size_t required_setup_size(size_t num_tensors, size_t alignment) {
    const size_t ptr_size = num_tensors * sizeof(void *);
    const size_t int_size = num_tensors * sizeof(int);
    // Layout: 6 ptr arrays, then 6 int arrays
    size_t size = 6 * ptr_size + 6 * int_size;
    size = ((size + alignment - 1) / alignment) * alignment;
    return size;
  }
};

// -----------------------------------------------------------------------------
// Helper routines to keep nvte_grouped_gemm readable
// -----------------------------------------------------------------------------
inline void validate_grouped_gemm_inputs(const transformer_engine::GroupedTensor *inputA,
                                         const transformer_engine::GroupedTensor *inputB,
                                         const transformer_engine::GroupedTensor *inputC,
                                         const transformer_engine::GroupedTensor *outputD,
                                         const transformer_engine::Tensor *alpha_tensor,
                                         const transformer_engine::Tensor *beta_tensor) {
  const size_t num_tensors = inputA->num_tensors;
  NVTE_CHECK(num_tensors >= 1, "Grouped GEMM: number of tensors must be at least 1");
  NVTE_CHECK(inputB->num_tensors == num_tensors,
             "Grouped GEMM: A and B must have the same number of tensors");
  // C can be NULL (will use D as C when beta=0)
  if (inputC != nullptr) {
    NVTE_CHECK(inputC->num_tensors == num_tensors,
               "Grouped GEMM: A and C must have the same number of tensors");
  }
  NVTE_CHECK(outputD->num_tensors == num_tensors,
             "Grouped GEMM: A and D must have the same number of tensors");

  // Validate alpha/beta have per-matrix values
  const size_t alpha_numel = alpha_tensor->data.numel();
  const size_t beta_numel = beta_tensor->data.numel();
  NVTE_CHECK(alpha_numel == num_tensors, "Grouped GEMM: alpha must have num_tensors (", num_tensors,
             ") elements, got ", alpha_numel);
  NVTE_CHECK(beta_numel == num_tensors, "Grouped GEMM: beta must have num_tensors (", num_tensors,
             ") elements, got ", beta_numel);

  auto is_fp8_or_16bit = [](transformer_engine::DType dtype) {
    return dtype == transformer_engine::DType::kFloat8E4M3 ||
           dtype == transformer_engine::DType::kFloat8E5M2 ||
           dtype == transformer_engine::DType::kBFloat16 ||
           dtype == transformer_engine::DType::kFloat16;
  };
  auto is_output_dtype = [](transformer_engine::DType dtype) {
    return dtype == transformer_engine::DType::kBFloat16 ||
           dtype == transformer_engine::DType::kFloat16 ||
           dtype == transformer_engine::DType::kFloat32;
  };
  NVTE_CHECK(is_fp8_or_16bit(inputA->dtype()) && is_fp8_or_16bit(inputB->dtype()),
             "Grouped GEMM inputs must be FP8, BF16, or FP16.");
  // Only check C dtype if C is provided
  if (inputC != nullptr) {
    NVTE_CHECK(is_output_dtype(inputC->dtype()), "Grouped GEMM: C must be BF16, FP16, or FP32.");
  }
  NVTE_CHECK(is_output_dtype(outputD->dtype()), "Grouped GEMM: D must be BF16, FP16, or FP32.");
  NVTE_CHECK(inputA->has_data() || inputA->has_columnwise_data(),
             "Grouped GEMM: A tensor is missing both row-wise and column-wise data");
  NVTE_CHECK(inputB->has_data() || inputB->has_columnwise_data(),
             "Grouped GEMM: B tensor is missing both row-wise and column-wise data");
}

// Select row-wise vs column-wise storage and adjust transpose flag for grouped GEMM.
// Mirrors the non-grouped GEMM logic for FP8 layout handling (TN-only on Hopper) and
// fallback to column-wise data when row-wise is absent.
// Contains all information needed for GEMM setup - shape already accounts for storage layout.
struct GroupedOperandSelection {
  TensorShapeInfo shape;  // Shape info with dims already swapped for columnwise if needed
  char *dptr = nullptr;
  void *scale_inv = nullptr;
  transformer_engine::DType dtype = transformer_engine::DType::kNumTypes;
  bool trans = false;
};

// Helper to create TensorShapeInfo from a GroupedTensor, optionally swapping first/last dims.
// When swap_dims=true, first_dims and last_dims are swapped to account for columnwise storage.
// Note: tensor_offsets are the same for rowwise and columnwise data (same element count per tensor).
inline TensorShapeInfo create_shape_info(const transformer_engine::GroupedTensor *t,
                                         bool swap_dims) {
  const bool has_first = t->first_dims.has_data();
  const bool has_last = t->last_dims.has_data();
  NVTE_CHECK(has_first || t->all_same_first_dim(),
             "GroupedTensor is missing first_dims for varying shapes");
  NVTE_CHECK(has_last || t->all_same_last_dim(),
             "GroupedTensor is missing last_dims for varying shapes");

  const int64_t *first_ptr = has_first ? static_cast<const int64_t *>(t->first_dims.dptr) : nullptr;
  const int64_t *last_ptr = has_last ? static_cast<const int64_t *>(t->last_dims.dptr) : nullptr;
  const int64_t uniform_first = has_first ? 0 : static_cast<int64_t>(t->get_common_first_dim());
  const int64_t uniform_last = has_last ? 0 : static_cast<int64_t>(t->get_common_last_dim());

  const int64_t *offsets_ptr =
      t->tensor_offsets.has_data() ? static_cast<const int64_t *>(t->tensor_offsets.dptr) : nullptr;

  if (swap_dims) {
    // Swap first/last to account for columnwise (transposed) storage
    return {last_ptr, first_ptr, offsets_ptr, uniform_last, uniform_first};
  }
  return {first_ptr, last_ptr, offsets_ptr, uniform_first, uniform_last};
}

inline GroupedOperandSelection select_grouped_operand(const transformer_engine::GroupedTensor *t,
                                                      bool trans, bool is_A) {
  using namespace transformer_engine;
  const bool has_row = t->has_data();
  const bool has_col = t->has_columnwise_data();
  NVTE_CHECK(has_row || has_col,
             "Grouped GEMM operand is missing both row-wise and column-wise data");

  // Currently only unquantized data and tensor-scaled FP8 are supported.
  const auto sm = t->scaling_mode;
  NVTE_CHECK(sm == NVTE_DELAYED_TENSOR_SCALING,
             "Grouped GEMM is only supported with unquantized data and tensor-scaled FP8 data");

  const DType row_dtype = t->data.dtype;
  const DType col_dtype = t->columnwise_data.dtype;
  GroupedOperandSelection sel;
  sel.trans = trans;

  const DType rep_dtype = has_row ? row_dtype : col_dtype;
  const bool is_fp8 = is_fp8_dtype(rep_dtype);
  const bool non_tn_fp8_ok = nvte_is_non_tn_fp8_gemm_supported();

  // Helper to select columnwise storage (swaps dims in shape)
  auto use_columnwise = [&]() {
    sel.dptr = static_cast<char *>(t->columnwise_data.dptr);
    sel.scale_inv = t->columnwise_scale_inv.dptr;
    sel.dtype = col_dtype;
    sel.shape = create_shape_info(t, /*swap_dims=*/true);
  };

  // Helper to select row-wise storage
  auto use_rowwise = [&]() {
    sel.dptr = static_cast<char *>(t->data.dptr);
    sel.scale_inv = t->scale_inv.dptr;
    sel.dtype = row_dtype;
    sel.shape = create_shape_info(t, /*swap_dims=*/false);
  };

  // Hopper-style TN-only FP8: force TN by switching layout and flipping transpose when needed.
  if (is_fp8 && !non_tn_fp8_ok) {
    if (is_A) {
      if (!sel.trans) {
        NVTE_CHECK(has_col, "Grouped GEMM: A is missing column-wise data needed for FP8 TN layout");
        use_columnwise();
        sel.trans = true;  // using pre-transposed storage
        return sel;
      }
    } else {  // B
      if (sel.trans) {
        NVTE_CHECK(has_col, "Grouped GEMM: B is missing column-wise data needed for FP8 TN layout");
        use_columnwise();
        sel.trans = false;  // using pre-transposed storage
        return sel;
      }
    }
  }

  // If only column-wise data is available, mirror the transpose flag (pre-transposed storage).
  if (!has_row && has_col) {
    // On Hopper FP8, this would break TN requirement - should have been handled above
    NVTE_CHECK(
        !is_fp8 || non_tn_fp8_ok,
        "Grouped GEMM: FP8 on Hopper requires row-wise data for this transpose configuration");
    use_columnwise();
    sel.trans = !trans;  // flip transpose for pre-transposed storage
    return sel;
  }

  // Default: use row-wise data
  use_rowwise();
  return sel;
}

inline void *validate_and_get_workspace_ptr(transformer_engine::Tensor *ws, size_t required_size,
                                            const char *workspace_name) {
  NVTE_CHECK(ws != nullptr, workspace_name, " tensor is null.");
  const size_t provided_size = get_buffer_size_bytes(ws->data.numel(), ws->data.dtype);
  NVTE_CHECK(provided_size >= required_size, "Grouped GEMM: Insufficient ", workspace_name,
             ". Required: ", required_size, " bytes, Available: ", provided_size, " bytes.");
  return ws->data.dptr;
}

inline void init_matrix_layouts(cublasLtMatrixLayoutOpaque_t &descA,
                                cublasLtMatrixLayoutOpaque_t &descB,
                                cublasLtMatrixLayoutOpaque_t &descC,
                                cublasLtMatrixLayoutOpaque_t &descD,
                                const GroupedGemmSetupWorkspace &ws,
                                const GroupedOperandSelection &A_sel,
                                const GroupedOperandSelection &B_sel,
                                const transformer_engine::GroupedTensor *D, size_t num_tensors) {
  const cudaDataType_t A_type = get_cuda_dtype(A_sel.dtype);
  const cudaDataType_t B_type = get_cuda_dtype(B_sel.dtype);
  const cudaDataType_t D_type = get_cuda_dtype(D->dtype());

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
                             cublasOperation_t op_B) {
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescInit(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_A,
                                                   sizeof(op_A)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_B,
                                                   sizeof(op_B)));

  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                   &pointer_mode, sizeof(pointer_mode)));

  int64_t alphabeta_batch_stride = 1;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE,
                                                   &alphabeta_batch_stride, sizeof(int64_t)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&matmulDesc,
                                                   CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE,
                                                   &alphabeta_batch_stride, sizeof(int64_t)));
}

inline void set_fp8_scale_pointers(cublasLtMatmulDescOpaque_t &matmulDesc,
                                   const GroupedOperandSelection &A_sel,
                                   const GroupedOperandSelection &B_sel) {
  const bool is_fp8_a = is_fp8_dtype(A_sel.dtype);
  const bool is_fp8_b = is_fp8_dtype(B_sel.dtype);
  if (!is_fp8_a && !is_fp8_b) return;

  if (is_fp8_a) {
    void *a_scale_inv = A_sel.scale_inv;
    NVTE_CHECK(a_scale_inv != nullptr, "FP8 grouped GEMM: A scale_inv is required");
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        &matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_inv, sizeof(a_scale_inv)));
  }
  if (is_fp8_b) {
    void *b_scale_inv = B_sel.scale_inv;
    NVTE_CHECK(b_scale_inv != nullptr, "FP8 grouped GEMM: B scale_inv is required");
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        &matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_inv, sizeof(b_scale_inv)));
  }
}

// Constants for grouped GEMM workspace (declared early for use in heuristics)
static constexpr size_t kGroupedGemmAlignment = 256;
static constexpr size_t kGroupedGemmCublasWorkspaceSize = 32ull * 1024 * 1024;  // 32 MiB

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

// Single kernel that sets up all GEMM parameters.
// Rationale: cuBLASLt grouped matmul API needs flat arrays of pointers and per-matrix dimensions,
// but NVTEGroupedTensor stores a single contiguous buffer + optional per-tensor offsets/shapes.
// We bridge the mismatch on GPU by computing per-group pointers and storage dims in one kernel.
__global__ void setup_grouped_gemm_kernel(
    // Output arrays
    void **A_ptrs, void **B_ptrs, void **C_ptrs, void **D_ptrs, int *a_rows, int *a_cols,
    int *b_rows, int *b_cols, int *d_rows, int *d_cols, float **alpha_ptrs, float **beta_ptrs,
    // Inputs
    char *a_base, char *b_base, char *c_base, char *d_base, TensorShapeInfo A_meta,
    TensorShapeInfo B_meta, TensorShapeInfo C_meta, TensorShapeInfo D_meta, size_t a_elem_size,
    size_t b_elem_size, size_t c_elem_size, size_t d_elem_size, float *alpha_ptr, float *beta_ptr,
    size_t num_tensors) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tensors) return;

  // Get dimensions for this tensor (from array or uniform value)
  int64_t a_first = A_meta.first_dims ? A_meta.first_dims[idx] : A_meta.uniform_first;
  int64_t a_last = A_meta.last_dims ? A_meta.last_dims[idx] : A_meta.uniform_last;
  int64_t b_first = B_meta.first_dims ? B_meta.first_dims[idx] : B_meta.uniform_first;
  int64_t b_last = B_meta.last_dims ? B_meta.last_dims[idx] : B_meta.uniform_last;
  int64_t d_first = D_meta.first_dims ? D_meta.first_dims[idx] : D_meta.uniform_first;
  int64_t d_last = D_meta.last_dims ? D_meta.last_dims[idx] : D_meta.uniform_last;

  // Compute offsets (from array or compute from uniform dims)
  int64_t a_offset =
      A_meta.offsets ? A_meta.offsets[idx] : (idx * A_meta.uniform_first * A_meta.uniform_last);
  int64_t b_offset =
      B_meta.offsets ? B_meta.offsets[idx] : (idx * B_meta.uniform_first * B_meta.uniform_last);
  int64_t c_offset =
      C_meta.offsets ? C_meta.offsets[idx] : (idx * C_meta.uniform_first * C_meta.uniform_last);
  int64_t d_offset =
      D_meta.offsets ? D_meta.offsets[idx] : (idx * D_meta.uniform_first * D_meta.uniform_last);

  // Compute data pointers
  A_ptrs[idx] = a_base + a_offset * a_elem_size;
  B_ptrs[idx] = b_base + b_offset * b_elem_size;
  C_ptrs[idx] = c_base + c_offset * c_elem_size;
  D_ptrs[idx] = d_base + d_offset * d_elem_size;

  // Compute storage dimensions for cuBLAS matrix layouts.
  // For INPUTS (A, B): Row-wise storage is seen as transposed column-major by cuBLAS,
  // so rows=last, cols=first. For columnwise, dims are already swapped.
  a_rows[idx] = static_cast<int>(a_last);
  a_cols[idx] = static_cast<int>(a_first);
  b_rows[idx] = static_cast<int>(b_last);
  b_cols[idx] = static_cast<int>(b_first);
  d_rows[idx] = static_cast<int>(d_last);
  d_cols[idx] = static_cast<int>(d_first);

  // Fill alpha/beta pointers (per-matrix)
  alpha_ptrs[idx] = alpha_ptr + idx;
  beta_ptrs[idx] = beta_ptr + idx;
}

// Launch the setup kernel to populate workspace arrays
inline void launch_grouped_gemm_setup(
    const GroupedGemmSetupWorkspace &ws, const GroupedOperandSelection &A_sel,
    const GroupedOperandSelection &B_sel, const transformer_engine::GroupedTensor *C,
    const transformer_engine::GroupedTensor *D, const transformer_engine::Tensor *alpha_tensor,
    const transformer_engine::Tensor *beta_tensor, size_t num_tensors, cudaStream_t stream) {
  // Use shape info from selection (already accounts for columnwise dimension swap)
  TensorShapeInfo A_meta = A_sel.shape;
  TensorShapeInfo B_meta = B_sel.shape;
  TensorShapeInfo C_meta = TensorShapeInfo::create_shape_info_for_C(C, D);
  TensorShapeInfo D_meta = TensorShapeInfo::from_tensor(D);

  char *c_base = static_cast<char *>(C->data.dptr);
  char *d_base = static_cast<char *>(D->data.dptr);

  const size_t a_elem_size = transformer_engine::typeToSize(A_sel.dtype);
  const size_t b_elem_size = transformer_engine::typeToSize(B_sel.dtype);
  const size_t c_elem_size = transformer_engine::typeToSize(C->dtype());
  const size_t d_elem_size = transformer_engine::typeToSize(D->dtype());

  const int threads_per_block = 256;
  const int num_blocks = (num_tensors + threads_per_block - 1) / threads_per_block;

  setup_grouped_gemm_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      ws.A_ptrs, ws.B_ptrs, ws.C_ptrs, ws.D_ptrs, ws.a_rows, ws.a_cols, ws.b_rows, ws.b_cols,
      ws.d_rows, ws.d_cols, ws.alpha_ptrs, ws.beta_ptrs, A_sel.dptr, B_sel.dptr, c_base, d_base,
      A_meta, B_meta, C_meta, D_meta, a_elem_size, b_elem_size, c_elem_size, d_elem_size,
      static_cast<float *>(alpha_tensor->data.dptr), static_cast<float *>(beta_tensor->data.dptr),
      num_tensors);

  NVTE_CHECK_CUDA(cudaGetLastError());
}

inline size_t grouped_gemm_setup_workspace_size(size_t num_tensors) {
  return GroupedGemmSetupWorkspace::required_setup_size(num_tensors, kGroupedGemmAlignment);
}

}  // namespace

size_t nvte_grouped_gemm_setup_workspace_size(size_t num_tensors) {
  return grouped_gemm_setup_workspace_size(num_tensors);
}

void nvte_grouped_gemm(const NVTEGroupedTensor A, int transa, const NVTEGroupedTensor B, int transb,
                       const NVTEGroupedTensor C, NVTEGroupedTensor D, const NVTETensor alpha,
                       const NVTETensor beta, NVTETensor workspace_setup,
                       NVTETensor workspace_cublas, NVTEGroupedMatmulConfig config,
                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_grouped_gemm);
  using namespace transformer_engine;

  // Grouped GEMM requires Blackwell (SM100) or newer and cuBLAS 13.2+
  const int current_device = transformer_engine::cuda::current_device();
  NVTE_CHECK(transformer_engine::cuda::sm_arch(current_device) >= 100,
             "nvte_grouped_gemm requires Blackwell (SM100) or newer architecture.");
  NVTE_CHECK(transformer_engine::cuda::cublas_version() >= 130200,
             "nvte_grouped_gemm requires cuBLAS 13.2+, but run-time cuBLAS version is ",
             transformer_engine::cuda::cublas_version());

  // Convert to internal types
  const GroupedTensor *inputA = convertNVTEGroupedTensorCheck(A);
  const GroupedTensor *inputB = convertNVTEGroupedTensorCheck(B);
  const GroupedTensor *inputC_raw = convertNVTEGroupedTensor(C);  // Can be NULL
  GroupedTensor *outputD = convertNVTEGroupedTensorCheck(D);
  const Tensor *alpha_tensor = convertNVTETensorCheck(alpha);
  const Tensor *beta_tensor = convertNVTETensorCheck(beta);
  Tensor *wspace_setup = convertNVTETensor(workspace_setup);
  Tensor *wspace_cublas = convertNVTETensor(workspace_cublas);

  // Parse config (if provided)
  GroupedMatmulConfig config_;
  if (config != nullptr) {
    config_ = *reinterpret_cast<GroupedMatmulConfig *>(config);
  }

  // Validate inputs and num_tensors
  validate_grouped_gemm_inputs(inputA, inputB, inputC_raw, outputD, alpha_tensor, beta_tensor);

  // If C is NULL, use D as C (valid when beta=0, cuBLAS won't read C data)
  const GroupedTensor *inputC = (inputC_raw != nullptr) ? inputC_raw : outputD;
  const size_t num_tensors = inputA->num_tensors;

  // Select operand storage (row-wise vs column-wise) and adjust transpose flags to
  // mirror the non-grouped GEMM logic for FP8 layout constraints.
  const auto A_sel = select_grouped_operand(inputA, static_cast<bool>(transa), /*is_A=*/true);
  const auto B_sel = select_grouped_operand(inputB, static_cast<bool>(transb), /*is_A=*/false);

  // Workspaces: setup (pointer arrays) and cuBLAS
  const size_t setup_workspace_size = grouped_gemm_setup_workspace_size(num_tensors);
  const size_t cublas_workspace_size = kGroupedGemmCublasWorkspaceSize;

  void *setup_workspace_ptr = validate_and_get_workspace_ptr(wspace_setup, setup_workspace_size,
                                                             "Grouped GEMM setup workspace");
  void *cublas_workspace_ptr = validate_and_get_workspace_ptr(wspace_cublas, cublas_workspace_size,
                                                              "Grouped GEMM cuBLAS workspace");

  auto setup_workspace = GroupedGemmSetupWorkspace::from_buffers(
      static_cast<char *>(setup_workspace_ptr), num_tensors);
  launch_grouped_gemm_setup(setup_workspace, A_sel, B_sel, inputC, outputD, alpha_tensor,
                            beta_tensor, num_tensors, stream);

  // Get cuBLAS handle
  using cublasHandleManager = detail::HandleManager<cublasLtHandle_t, CreateCublasHandle>;
  cublasLtHandle_t handle = cublasHandleManager::Instance().GetHandle();

  // Setup cuBLAS operations
  cublasOperation_t op_A = A_sel.trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_B = B_sel.trans ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Create grouped matrix layouts
  cublasLtMatrixLayoutOpaque_t descA, descB, descC, descD;
  init_matrix_layouts(descA, descB, descC, descD, setup_workspace, A_sel, B_sel, outputD,
                      num_tensors);

  // Create matmul descriptor
  cublasLtMatmulDescOpaque_t matmulDesc;
  init_matmul_desc(matmulDesc, op_A, op_B);
  set_fp8_scale_pointers(matmulDesc, A_sel, B_sel);

  // Compute average dimensions for heuristics
  // K dimension: if transa, K is A's first dim; if not, K is A's last dim
  // Use original inputA and transa for heuristics (not modified A_sel.trans)
  int64_t avg_m_val = config_.avg_m.value_or(compute_avg_first_dim(outputD));
  int64_t avg_n_val = config_.avg_n.value_or(compute_avg_last_dim(outputD));
  int64_t avg_k_val =
      config_.avg_k.value_or(transa ? compute_avg_first_dim(inputA) : compute_avg_last_dim(inputA));

  // Heuristic selection
  cublasLtMatmulAlgo_t algo = select_grouped_gemm_algo(handle, matmulDesc, descA, descB, descC,
                                                       descD, avg_m_val, avg_n_val, avg_k_val);

  // Execute the grouped GEMM
  NVTE_CHECK_CUBLAS(cublasLtMatmul(handle, &matmulDesc, setup_workspace.alpha_ptrs,
                                   setup_workspace.A_ptrs, &descA, setup_workspace.B_ptrs, &descB,
                                   setup_workspace.beta_ptrs, setup_workspace.C_ptrs, &descC,
                                   setup_workspace.D_ptrs, &descD, &algo, cublas_workspace_ptr,
                                   kGroupedGemmCublasWorkspaceSize, stream));
}

#else  // CUBLAS_VERSION < 130200

void nvte_grouped_gemm(const NVTEGroupedTensor A, int transa, const NVTEGroupedTensor B, int transb,
                       const NVTEGroupedTensor C, NVTEGroupedTensor D, const NVTETensor alpha,
                       const NVTETensor beta, NVTETensor workspace_setup,
                       NVTETensor workspace_cublas, NVTEGroupedMatmulConfig config,
                       cudaStream_t stream) {
  NVTE_ERROR("nvte_grouped_gemm requires cuBLAS 13.2+, but compile-time cuBLAS version is ",
             CUBLAS_VERSION, ". Please upgrade to CUDA 13.1 or newer.");
}

#endif  // CUBLAS_VERSION >= 130200
