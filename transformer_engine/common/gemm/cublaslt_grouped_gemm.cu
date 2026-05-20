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

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../cast/mxfp8/swizzle.cuh"
#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/handle_manager.h"
#include "../util/logging.h"
#include "../util/vectorized_pointwise.h"
#include "./config.h"

namespace {

inline void CreateCublasHandle(cublasLtHandle_t *handle) {
  NVTE_CHECK_CUBLAS(cublasLtCreate(handle));
}

}  // namespace

// MXFP8 support for grouped GEMM requires cuBLAS 13.3+
#define CUBLAS_MXFP8_GROUPED_GEMM_VERSION 130300

// Hopper (SM90) support for grouped GEMM requires cuBLAS 13.4+
#define CUBLAS_GROUPED_GEMM_HOPPER_VERSION 130400

// NVFP4 support for grouped GEMM requires cuBLAS 13.4+
#define CUBLAS_NVFP4_GROUPED_GEMM_VERSION 130400

// FP8 block scaling support for grouped GEMM requires cuBLAS 13.4+
#define CUBLAS_FP8_BLOCK_GROUPED_GEMM_VERSION 130400

// BF16 support for grouped GEMM requires cuBLAS 13.3+
#define CUBLAS_GROUPED_GEMM_VERSION 130300

#if CUBLAS_VERSION >= CUBLAS_GROUPED_GEMM_VERSION

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

// Helper functions to compute average dimensions for cuBLASLt algorithm-selection heuristics.
//
// logical_shape encoding (from build_grouped_tensor):
//   all_same:      {num_tensors * M, N}
//   varying_first: {sum_of_first_dims, common_last}
//   varying_last:  {common_first, sum_of_last_dims}
//   varying_both:  {1, total_elements}   <-- lossy, can't recover per-dim averages
//
// We use all_same_first/last_dim() + get_common_first/last_dim() to get exact
// answers whenever possible, falling back to logical_shape division otherwise.
// For varying_both, per-dim averages are unrecoverable without a D2H copy,
// so we return 1 — a valid non-zero hint that won't skip work.
inline int64_t compute_avg_first_dim(const transformer_engine::GroupedTensor *t) {
  if (t->all_same_first_dim()) {
    return static_cast<int64_t>(t->get_common_first_dim());
  }
  const int64_t n = static_cast<int64_t>(t->num_tensors);
  if (t->all_same_last_dim()) {
    // varying_first only: logical_shape = {sum_of_first_dims, common_last}
    return static_cast<int64_t>(t->logical_shape.data[0]) / n;
  }
  // varying_both: logical_shape = {1, total_elements}, no way to recover avg first dim
  return 1;
}

inline int64_t compute_avg_last_dim(const transformer_engine::GroupedTensor *t) {
  if (t->all_same_last_dim()) {
    // logical_shape[1] is the common N
    return static_cast<int64_t>(t->logical_shape.data[1]);
  }
  // When varying, logical_shape[1] should be sum of last dims if provided; otherwise fallback to avg via division.
  return static_cast<int64_t>(t->logical_shape.data[1]) / static_cast<int64_t>(t->num_tensors);
}

// Constants for grouped GEMM workspace (declared early for use in helpers)
static constexpr size_t kGroupedGemmAlignment = 256;
static constexpr size_t kGroupedGemmCublasWorkspaceSize = 32ull * 1024 * 1024;  // 32 MiB

// Workspace layout for grouped GEMM.
// Layout described once in `from_buffers`; `required_setup_size` runs the same walker
// with base=nullptr to derive the total byte count, so the two stay in sync by construction.
struct GroupedGemmSetupWorkspace {
  void **A_ptrs = nullptr;
  void **B_ptrs = nullptr;
  void **C_ptrs = nullptr;
  void **D_ptrs = nullptr;
  float **alpha_ptrs = nullptr;
  float **beta_ptrs = nullptr;
  // Per-tensor scale_inv pointers (float* for tensor scaling, E8M0* for MXFP8, E4M3* for NVFP4)
  void **a_scale_inv_ptrs = nullptr;
  void **b_scale_inv_ptrs = nullptr;
  // Storage dimensions for cuBLAS matrix layouts
  int *a_rows = nullptr;
  int *a_cols = nullptr;
  int *b_rows = nullptr;
  int *b_cols = nullptr;
  int *d_rows = nullptr;  // M (first dim) - also used for C
  int *d_cols = nullptr;  // N (last dim) - also used for C
  // NVFP4: per-group computed alpha values (alpha * amax_A * amax_B * factor_inv)
  float *nvfp4_computed_alpha = nullptr;
  // End-of-layout offset in bytes (unaligned). required_setup_size rounds this up.
  size_t total_bytes = 0;

  // Walk the layout once. If `base` is non-null, fields are populated; otherwise
  // only `total_bytes` is meaningful (used by required_setup_size).
  static GroupedGemmSetupWorkspace from_buffers(char *base, size_t num_tensors) {
    GroupedGemmSetupWorkspace ws;
    constexpr size_t kPtrAlignment = 16;  // cuBLAS requires 16-byte alignment for pointer arrays
    const size_t ptr_size = num_tensors * sizeof(void *);
    const size_t int_size = num_tensors * sizeof(int);
    const size_t float_size = num_tensors * sizeof(float);
    size_t offset = 0;

    auto align_ptr = [&]() {
      offset = (offset + kPtrAlignment - 1) / kPtrAlignment * kPtrAlignment;
    };
    auto place = [&](auto *&field, size_t size_bytes) {
      using Field = std::remove_reference_t<decltype(field)>;
      if (base != nullptr) field = reinterpret_cast<Field>(base + offset);
      offset += size_bytes;
    };

    // 8 pointer arrays (each 16-byte aligned), then 6 int arrays, then 1 float array.
    align_ptr();
    place(ws.A_ptrs, ptr_size);
    align_ptr();
    place(ws.B_ptrs, ptr_size);
    align_ptr();
    place(ws.C_ptrs, ptr_size);
    align_ptr();
    place(ws.D_ptrs, ptr_size);
    align_ptr();
    place(ws.alpha_ptrs, ptr_size);
    align_ptr();
    place(ws.beta_ptrs, ptr_size);
    align_ptr();
    place(ws.a_scale_inv_ptrs, ptr_size);
    align_ptr();
    place(ws.b_scale_inv_ptrs, ptr_size);
    place(ws.a_rows, int_size);
    place(ws.a_cols, int_size);
    place(ws.b_rows, int_size);
    place(ws.b_cols, int_size);
    place(ws.d_rows, int_size);
    place(ws.d_cols, int_size);
    place(ws.nvfp4_computed_alpha, float_size);

    ws.total_bytes = offset;
    return ws;
  }

  static size_t required_setup_size(size_t num_tensors, size_t alignment) {
    const size_t raw = from_buffers(nullptr, num_tensors).total_bytes;
    return ((raw + alignment - 1) / alignment) * alignment;
  }
};

inline bool grouped_gemm_supports_per_group_alpha_beta(int sm) { return sm >= 100; }

inline size_t validate_grouped_gemm_inputs(
    size_t num_tensors, std::initializer_list<const transformer_engine::GroupedTensor *> inputs,
    const transformer_engine::Tensor *alpha_tensor, const transformer_engine::Tensor *beta_tensor,
    bool supports_per_group_alpha_beta) {
  NVTE_CHECK(num_tensors >= 1, "Grouped GEMM: number of tensors must be at least 1");
  for (const auto *tensor : inputs) {
    NVTE_CHECK(tensor->num_tensors == num_tensors,
               "Grouped GEMM: inputs must have the same number of tensors");
  }

  // Hopper currently requires a uniform alpha/beta scalar for the whole grouped GEMM,
  // while Blackwell+ supports per-matrix alpha/beta.
  const size_t alpha_numel = alpha_tensor->data.numel();
  const size_t beta_numel = beta_tensor->data.numel();
  const size_t expected_alphabeta_numel = supports_per_group_alpha_beta ? num_tensors : 1;
  const char *alphabeta_desc = supports_per_group_alpha_beta ? "num_tensors" : "1";
  NVTE_CHECK(alpha_numel == expected_alphabeta_numel, "Grouped GEMM: alpha must have ",
             alphabeta_desc, " element(s), got ", alpha_numel);
  NVTE_CHECK(beta_numel == expected_alphabeta_numel, "Grouped GEMM: beta must have ",
             alphabeta_desc, " element(s), got ", beta_numel);

  auto is_supported_input_dtype = [](transformer_engine::DType dtype) {
    return dtype == transformer_engine::DType::kFloat8E4M3 ||
           dtype == transformer_engine::DType::kFloat8E5M2 ||
           dtype == transformer_engine::DType::kBFloat16 ||
           dtype == transformer_engine::DType::kFloat16 ||
           dtype == transformer_engine::DType::kFloat4E2M1;
  };
  for (const auto *tensor : inputs) {
    if (tensor->has_data() || tensor->has_columnwise_data()) {
      NVTE_CHECK(is_supported_input_dtype(tensor->dtype()),
                 "Grouped GEMM inputs must be FP8, NVFP4, BF16, or FP16, got ",
                 transformer_engine::to_string(tensor->dtype()), ".");
    }
  }
  // Cross-operand consistency across all inputs (skip tensors without data).
  const transformer_engine::GroupedTensor *ref = nullptr;
  for (const auto *tensor : inputs) {
    if (tensor->has_data() || tensor->has_columnwise_data()) {
      ref = tensor;
      break;
    }
  }
  if (ref != nullptr) {
    const bool ref_is_fp8 = is_fp8_dtype(ref->dtype());
    const bool ref_is_fp4 = is_fp4_dtype(ref->dtype());
    const bool ref_is_mxfp8 = transformer_engine::is_mxfp_scaling(ref->scaling_mode);
    const bool ref_is_nvfp4 = transformer_engine::is_nvfp_scaling(ref->scaling_mode);
    const bool ref_is_fp8_block = transformer_engine::is_fp8_block_scaling(ref->scaling_mode);
    for (const auto *tensor : inputs) {
      if (!(tensor->has_data() || tensor->has_columnwise_data())) continue;
      NVTE_CHECK(is_fp8_dtype(tensor->dtype()) == ref_is_fp8,
                 "Grouped GEMM: A and B must both be FP8 or both be non-FP8.");
      NVTE_CHECK(is_fp4_dtype(tensor->dtype()) == ref_is_fp4,
                 "Grouped GEMM: A and B must both be NVFP4 or both be non-NVFP4.");
      NVTE_CHECK(transformer_engine::is_mxfp_scaling(tensor->scaling_mode) == ref_is_mxfp8,
                 "Grouped GEMM: A and B must both use MXFP8 scaling or both not.");
      NVTE_CHECK(transformer_engine::is_nvfp_scaling(tensor->scaling_mode) == ref_is_nvfp4,
                 "Grouped GEMM: A and B must both use NVFP4 scaling or both not.");
      NVTE_CHECK(transformer_engine::is_fp8_block_scaling(tensor->scaling_mode) == ref_is_fp8_block,
                 "Grouped GEMM: A and B must both use FP8 block scaling or both not.");
      if (ref_is_mxfp8 || transformer_engine::is_nvfp_scaling(tensor->scaling_mode)) {
        NVTE_CHECK(tensor->with_gemm_swizzled_scales,
                   "Grouped GEMM: scales must be swizzled for GEMM (MXFP8/NVFP4).");
      }
    }
  }
  return num_tensors;
}

inline void validate_grouped_gemm_output_dtype(transformer_engine::DType a_dtype,
                                               transformer_engine::DType b_dtype,
                                               transformer_engine::DType output_dtype,
                                               const char *name) {
  const bool is_output_dtype = output_dtype == transformer_engine::DType::kBFloat16 ||
                               output_dtype == transformer_engine::DType::kFloat16 ||
                               output_dtype == transformer_engine::DType::kFloat32;
  NVTE_CHECK(is_output_dtype, "Grouped GEMM: ", name, " must be BF16, FP16, or FP32.");
  if (!is_fp4_dtype(a_dtype) && !is_fp4_dtype(b_dtype)) return;
  NVTE_CHECK(!is_fp4_dtype(output_dtype), "FP4 GEMM output is not supported!");
  NVTE_CHECK(get_cuda_dtype(output_dtype) != CUDA_R_16F, "FP4 GEMM does not support FP16 output!");
}

inline void validate_grouped_gemm_outputs(
    size_t num_tensors, transformer_engine::DType a_dtype, transformer_engine::DType b_dtype,
    std::initializer_list<const transformer_engine::GroupedTensor *> outputs) {
  for (const auto *tensor : outputs) {
    if (tensor == nullptr) {
      continue;
    }
    NVTE_CHECK(tensor->num_tensors == num_tensors,
               "Grouped GEMM: outputs must have the same number of tensors as inputs");
    validate_grouped_gemm_output_dtype(a_dtype, b_dtype, tensor->dtype(), "outputs");
  }
}

inline size_t grouped_gemm_setup_workspace_size(size_t num_tensors) {
  return GroupedGemmSetupWorkspace::required_setup_size(num_tensors, kGroupedGemmAlignment);
}

inline void check_grouped_gemm_requirements(const char *api_name) {
  const int current_device = transformer_engine::cuda::current_device();
  const int sm = transformer_engine::cuda::sm_arch(current_device);
  const int cublas_ver = transformer_engine::cuda::cublas_version();
#if CUBLAS_VERSION >= CUBLAS_GROUPED_GEMM_HOPPER_VERSION
  NVTE_CHECK(sm >= 90, api_name, " requires Hopper (SM90) or newer architecture.");
  NVTE_CHECK(cublas_ver >= CUBLAS_GROUPED_GEMM_VERSION, api_name,
             " requires cuBLAS 13.3+, but run-time cuBLAS version is ", cublas_ver);
  if (sm < 100) {
    NVTE_CHECK(cublas_ver >= CUBLAS_GROUPED_GEMM_HOPPER_VERSION, api_name,
               " on Hopper (SM90) requires cuBLAS 13.4+, but run-time cuBLAS version is ",
               cublas_ver);
  }
#else
  NVTE_CHECK(sm >= 100, api_name, " requires Blackwell (SM100) or newer architecture.");
  NVTE_CHECK(cublas_ver >= CUBLAS_GROUPED_GEMM_VERSION, api_name,
             " requires cuBLAS 13.3+, but run-time cuBLAS version is ", cublas_ver);
#endif
}

inline transformer_engine::GroupedMatmulConfig parse_grouped_gemm_config(
    NVTEGroupedMatmulConfig config) {
  transformer_engine::GroupedMatmulConfig config_;
  if (config != nullptr) {
    config_ = *reinterpret_cast<transformer_engine::GroupedMatmulConfig *>(config);
  }
  return config_;
}

// Contains all information needed for one tensor operand for GEMM setup.
struct GroupedOperandSelection {
  TensorShapeInfo logical_tensor_shape;
  char *dptr = nullptr;
  void *scale_inv = nullptr;  // Contiguous array of scales (input)
  void *amax = nullptr;       // Per-tensor amax values (NVFP4 only)
  transformer_engine::DType dtype = transformer_engine::DType::kNumTypes;
  NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
  bool with_gemm_swizzled_scales = false;
  bool trans = false;
  bool rowwise = true;
  // Whether selected storage is physically transposed relative to logical shape.
  bool storage_transposed = false;
};

inline void validate_nvfp4_grouped_gemm_support(const GroupedOperandSelection &A_sel,
                                                const GroupedOperandSelection &B_sel,
                                                bool use_per_group_alpha_beta) {
  const bool nvfp4 = transformer_engine::is_nvfp_scaling(A_sel.scaling_mode) ||
                     transformer_engine::is_nvfp_scaling(B_sel.scaling_mode);
  if (!nvfp4) return;

  NVTE_CHECK(transformer_engine::is_nvfp_scaling(A_sel.scaling_mode) &&
                 transformer_engine::is_nvfp_scaling(B_sel.scaling_mode),
             "Grouped GEMM: A and B must both use NVFP4 scaling or both not.");
  NVTE_CHECK(use_per_group_alpha_beta,
             "Grouped GEMM: NVFP4 requires per-group alpha/beta support because each group "
             "has its own amax-derived global scale.");
}

// FP8 block scaling grouped GEMM is only supported on Hopper (SM90). 
inline void validate_fp8_block_grouped_gemm_support(const GroupedOperandSelection &A_sel,
                                                    const GroupedOperandSelection &B_sel, int sm) {
  const bool a_fp8_block = transformer_engine::is_fp8_block_scaling(A_sel.scaling_mode);
  const bool b_fp8_block = transformer_engine::is_fp8_block_scaling(B_sel.scaling_mode);
  if (!a_fp8_block && !b_fp8_block) return;

  NVTE_CHECK(a_fp8_block && b_fp8_block,
             "Grouped GEMM: A and B must both use FP8 block scaling or both not.");
  NVTE_CHECK(sm == 90,
             "Grouped GEMM: FP8 block scaling is only supported on Hopper (SM90); "
             "use MXFP8 on Blackwell (SM100) or newer.");
}

inline bool is_compatible_grouped_scaling_mode(NVTEScalingMode a_mode, NVTEScalingMode b_mode) {
  const bool a_fp8_block = transformer_engine::is_fp8_block_scaling(a_mode);
  const bool b_fp8_block = transformer_engine::is_fp8_block_scaling(b_mode);
  if (a_fp8_block || b_fp8_block) {
    return a_fp8_block && b_fp8_block &&
           !(a_mode == NVTE_BLOCK_SCALING_2D && b_mode == NVTE_BLOCK_SCALING_2D);
  }
  return a_mode == b_mode;
}

struct GroupedGemmConfig {
  bool use_split_accumulator = false;
  bool use_fp8 = false;
  bool use_per_group_alpha_beta = false;
  void *alpha_dptr = nullptr;
  void *beta_dptr = nullptr;
  int64_t avg_m = 0;
  int64_t avg_n = 0;
  int64_t avg_k = 0;
  int sm_count = 0;
};

constexpr int kMaxGroups = 64;
// Arguments for the grouped GEMM kernel that operates on multiple output tensors.
struct MultiTensorGroupGemmOutputArgs {
  void *data_ptrs[kMaxGroups];
  int rows[kMaxGroups];
  int cols[kMaxGroups];
};

// Arguments for the grouped GEMM kernel that operates on multiple inputA tensors.
struct MultiTensorGroupGemmInputArgs {
  void *data_ptrs[kMaxGroups];
  void *scale_inv_ptrs[kMaxGroups];
  void *amax_ptrs[kMaxGroups];
  int rows[kMaxGroups];
  int cols[kMaxGroups];
};
struct MultiTensorListInfo {
  bool all_row = true;
  bool all_col = true;
  transformer_engine::DType row_dtype = transformer_engine::DType::kNumTypes;
  transformer_engine::DType col_dtype = transformer_engine::DType::kNumTypes;
  NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
  bool with_gemm_swizzled_scales = false;
};

struct OperandStorageChoice {
  bool use_rowwise = true;
  // Only meaningful when use_rowwise == false (columnwise storage). Indicates that the
  // columnwise buffer is physically transposed relative to logical shape.
  bool storage_transposed = false;
  bool trans = false;
};

inline OperandStorageChoice choose_grouped_operand_storage(bool trans, bool is_A, bool is_mxfp8,
                                                           bool is_fp8, bool is_nvfp4,
                                                           bool is_fp8_block, bool non_tn_fp8_ok,
                                                           bool has_row, bool has_col,
                                                           const char *name) {
  NVTE_CHECK(has_row || has_col, "Grouped GEMM: ", name,
             " is missing both row-wise and column-wise data");
  if (is_mxfp8) {
    if (is_A) {
      if (trans) {
        NVTE_CHECK(has_row, "Grouped GEMM: MXFP8 transposed ", name, " is missing row-wise data");
        return {true, false, trans};
      }
      NVTE_CHECK(has_col, "Grouped GEMM: MXFP8 non-transposed ", name,
                 " is missing column-wise data");
      return {false, false, trans};
    }
    if (trans) {
      NVTE_CHECK(has_col, "Grouped GEMM: MXFP8 transposed ", name, " is missing column-wise data");
      return {false, false, trans};
    }
    NVTE_CHECK(has_row, "Grouped GEMM: MXFP8 non-transposed ", name, " is missing row-wise data");
    return {true, false, trans};
  }

  // FP8 block scaling on Hopper: force TN by using transposed columnwise data.
  if (is_fp8_block && !non_tn_fp8_ok) {
    if (is_A && !trans) {
      NVTE_CHECK(has_col, "Grouped GEMM: ", name,
                 " is missing column-wise data needed for TN layout");
      return {false, true, true};
    }
    if (!is_A && trans) {
      NVTE_CHECK(has_col, "Grouped GEMM: ", name,
                 " is missing column-wise data needed for TN layout");
      return {false, true, false};
    }
  }

  // NVFP4: force TN by switching layout and flipping transpose.
  // NVFP4 columnwise data is the transposed tensor quantized rowwise.
  if (is_nvfp4) {
    if (is_A && !trans) {
      NVTE_CHECK(has_col, "Grouped GEMM: ", name,
                 " is missing column-wise data needed for TN layout");
      return {false, true, true};
    }
    if (!is_A && trans) {
      NVTE_CHECK(has_col, "Grouped GEMM: ", name,
                 " is missing column-wise data needed for TN layout");
      return {false, true, false};
    }
  }

  // Hopper-style TN-only FP8 (tensor scaling): force TN by switching layout and flipping transpose.
  if (is_fp8 && !non_tn_fp8_ok) {
    if (is_A && !trans) {
      NVTE_CHECK(has_col, "Grouped GEMM: ", name,
                 " is missing column-wise data needed for TN layout");
      return {false, true, true};
    }
    if (!is_A && trans) {
      NVTE_CHECK(has_col, "Grouped GEMM: ", name,
                 " is missing column-wise data needed for TN layout");
      return {false, true, false};
    }
  }

  // If only column-wise data is available, mirror the transpose flag (pre-transposed storage).
  if (!has_row && has_col) {
    NVTE_CHECK(!is_fp8 || non_tn_fp8_ok,
               "Grouped GEMM: FP8 on Hopper requires row-wise data for this transpose config.");
    return {false, true, !trans};
  }

  NVTE_CHECK(has_row, "Grouped GEMM: ", name, " is missing row-wise data");
  return {true, false, trans};
}

// Build Kernel Arguments detailing out addresses and other metadata for list of C/D tensors
// passed to the grouped GEMM kernel. Use-case: C/D --> List of wgrads for experts in MOE
inline MultiTensorGroupGemmOutputArgs build_grouped_gemm_multi_out_args(
    const NVTETensor *tensor_list, size_t list_size, size_t expected_num_tensors,
    transformer_engine::DType expected_dtype, const char *name) {
  MultiTensorGroupGemmOutputArgs args{};
  if (list_size == 0) {
    NVTE_CHECK(tensor_list == nullptr, "Grouped GEMM: ", name, "_list provided with num_", name,
               "_tensors=0");
    return args;
  }
  NVTE_CHECK(tensor_list != nullptr, "Grouped GEMM: ", name, "_list is null but num_", name,
             "_tensors=", list_size);
  NVTE_CHECK(list_size == expected_num_tensors, "Grouped GEMM: ", name,
             "_list must have num_tensors (", expected_num_tensors, ") entries, got ", list_size);
  NVTE_CHECK(list_size <= static_cast<size_t>(kMaxGroups), "Grouped GEMM: ", name,
             "_list supports up to ", kMaxGroups, " tensors per kernel, got ", list_size);

  for (size_t i = 0; i < list_size; ++i) {
    const transformer_engine::Tensor *t =
        transformer_engine::convertNVTETensorCheck(tensor_list[i]);
    NVTE_CHECK(t->has_data(), "Grouped GEMM: ", name, "_list tensor ", i, " has no data");
    NVTE_CHECK(t->dtype() == expected_dtype, "Grouped GEMM: ", name, "_list tensor ", i,
               " dtype mismatch. Expected ", transformer_engine::to_string(expected_dtype), " got ",
               transformer_engine::to_string(t->dtype()));
    const auto &shape = t->shape();
    NVTE_CHECK(shape.size() == 2, "Grouped GEMM: ", name, "_list tensor ", i, " must be 2D.");
    args.data_ptrs[i] = t->data.dptr;
    args.rows[i] = static_cast<int>(shape[1]);
    args.cols[i] = static_cast<int>(shape[0]);
  }
  return args;
}

// Build Kernel Arguments detailing out addresses and other metadata for list of A tensors
// passed to the grouped GEMM kernel. Use-case: A --> List of Expert weights
inline MultiTensorGroupGemmInputArgs build_grouped_gemm_multi_inputA_args(
    const NVTETensor *tensor_list, size_t list_size, bool use_rowwise, bool storage_transposed,
    bool requires_scale_inv, int64_t *avg_first_dim, int64_t *avg_last_dim, const char *name) {
  using namespace transformer_engine;
  MultiTensorGroupGemmInputArgs args{};
  *avg_first_dim = 0;
  *avg_last_dim = 0;
  if (list_size == 0) {
    return args;
  }
  for (size_t i = 0; i < list_size; ++i) {
    const transformer_engine::Tensor *t =
        transformer_engine::convertNVTETensorCheck(tensor_list[i]);
    const transformer_engine::SimpleTensor &data = use_rowwise ? t->data : t->columnwise_data;
    const transformer_engine::SimpleTensor &scale_inv =
        use_rowwise ? t->scale_inv : t->columnwise_scale_inv;
    NVTE_CHECK(data.has_data(), "Grouped GEMM: ", name, "_list tensor ", i,
               " is missing required data.");
    args.data_ptrs[i] = data.dptr;
    const auto &shape = t->shape();
    NVTE_CHECK(shape.size() == 2, "Grouped GEMM: ", name, "_list tensor ", i, " must be 2D.");
    const size_t first_dim = shape[0];
    const size_t last_dim = shape[1];
    if (storage_transposed) {
      args.rows[i] = static_cast<int>(first_dim);
      args.cols[i] = static_cast<int>(last_dim);
    } else {
      args.rows[i] = static_cast<int>(last_dim);
      args.cols[i] = static_cast<int>(first_dim);
    }
    *avg_first_dim += static_cast<int64_t>(first_dim);
    *avg_last_dim += static_cast<int64_t>(last_dim);

    if (requires_scale_inv) {
      NVTE_CHECK(scale_inv.has_data(), "Grouped GEMM: ", name, "_list tensor ", i,
                 " requires scale_inv.");
      args.scale_inv_ptrs[i] = scale_inv.dptr;
    } else {
      args.scale_inv_ptrs[i] = nullptr;
    }

    const transformer_engine::SimpleTensor &amax_src = use_rowwise ? t->amax : t->columnwise_amax;
    args.amax_ptrs[i] = amax_src.has_data() ? amax_src.dptr : nullptr;
  }
  *avg_first_dim /= static_cast<int64_t>(list_size);
  *avg_last_dim /= static_cast<int64_t>(list_size);
  return args;
}

inline MultiTensorListInfo validate_grouped_gemm_multi_inputA_list(const NVTETensor *tensor_list,
                                                                   size_t list_size,
                                                                   size_t expected_num_tensors,
                                                                   const char *name) {
  using namespace transformer_engine;
  MultiTensorListInfo info{};
  if (list_size == 0) {
    NVTE_CHECK(tensor_list == nullptr, "Grouped GEMM: ", name, "_list provided with num_", name,
               "_tensors=0");
    return info;
  }
  NVTE_CHECK(tensor_list != nullptr, "Grouped GEMM: ", name, "_list is null but num_", name,
             "_tensors=", list_size);
  NVTE_CHECK(list_size == expected_num_tensors, "Grouped GEMM: ", name,
             "_list must have num_tensors (", expected_num_tensors, ") entries, got ", list_size);
  NVTE_CHECK(list_size <= static_cast<size_t>(kMaxGroups), "Grouped GEMM: ", name,
             "_list supports up to ", kMaxGroups, " tensors per kernel, got ", list_size);

  const transformer_engine::Tensor *t0 = transformer_engine::convertNVTETensorCheck(tensor_list[0]);
  info.scaling_mode = t0->scaling_mode;
  info.with_gemm_swizzled_scales = t0->with_gemm_swizzled_scales;
  const bool mxfp8 = transformer_engine::is_mxfp_scaling(info.scaling_mode);
  const bool nvfp4 = transformer_engine::is_nvfp_scaling(info.scaling_mode);
  const bool fp8_block = transformer_engine::is_fp8_block_scaling(info.scaling_mode);
  NVTE_CHECK(info.scaling_mode == NVTE_DELAYED_TENSOR_SCALING || mxfp8 || nvfp4 || fp8_block,
             "Grouped GEMM: input list only supports tensor scaling, MXFP8, NVFP4, "
             "or FP8 block scaling.");

  for (size_t i = 0; i < list_size; ++i) {
    const transformer_engine::Tensor *t =
        transformer_engine::convertNVTETensorCheck(tensor_list[i]);
    NVTE_CHECK(t->scaling_mode == info.scaling_mode, "Grouped GEMM: ", name,
               "_list tensors must share the same scaling mode.");
    NVTE_CHECK(t->with_gemm_swizzled_scales == info.with_gemm_swizzled_scales,
               "Grouped GEMM: ", name, "_list tensors must share GEMM swizzled scale state.");

    if (t->has_data()) {
      if (info.row_dtype == DType::kNumTypes) {
        info.row_dtype = t->data.dtype;
      }
      // Check all tensors have the same dtype
      NVTE_CHECK(t->data.dtype == info.row_dtype, "Grouped GEMM: ", name,
                 "_list rowwise dtypes must match.");
    } else {
      // All tensors must have either data or columnwise data
      info.all_row = false;
    }

    if (t->has_columnwise_data()) {
      if (info.col_dtype == DType::kNumTypes) {
        info.col_dtype = t->columnwise_data.dtype;
      }
      NVTE_CHECK(t->columnwise_data.dtype == info.col_dtype, "Grouped GEMM: ", name,
                 "_list columnwise dtypes must match.");
    } else {
      // All tensors must have either data or columnwise data
      info.all_col = false;
    }
  }

  return info;
}

// Helper to create TensorShapeInfo from a GroupedTensor. Grouped tensor metadata is logical
// shape; storage-specific transposes are handled when building cuBLAS matrix layouts.
inline TensorShapeInfo create_shape_info(const transformer_engine::GroupedTensor *t) {
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

  return {first_ptr, last_ptr, offsets_ptr, uniform_first, uniform_last};
}

inline GroupedOperandSelection select_grouped_operand(const transformer_engine::GroupedTensor *t,
                                                      bool trans, bool is_A) {
  using namespace transformer_engine;
  const bool has_row = t->has_data();
  const bool has_col = t->has_columnwise_data();

  if (!has_row && !has_col) {
    GroupedOperandSelection sel{};
    sel.trans = trans;
    sel.scaling_mode = t->scaling_mode;
    sel.dtype = t->dtype();
    sel.logical_tensor_shape = create_shape_info(t);
    return sel;
  }

  const auto sm = t->scaling_mode;
  const bool mxfp8 = is_mxfp_scaling(sm);
  const bool nvfp4 = is_nvfp_scaling(sm);
  const bool fp8_block = is_fp8_block_scaling(sm);

  // Validate scaling mode
  NVTE_CHECK(sm == NVTE_DELAYED_TENSOR_SCALING || mxfp8 || nvfp4 || fp8_block,
             "Grouped GEMM is only supported with bf16, fp8 tensor scaling, MXFP8, NVFP4, "
             "and FP8 block scaling");

  const DType row_dtype = t->data.dtype;
  const DType col_dtype = t->columnwise_data.dtype;
  GroupedOperandSelection sel{};
  sel.trans = trans;
  sel.scaling_mode = sm;
  sel.with_gemm_swizzled_scales = t->with_gemm_swizzled_scales;

  const DType rep_dtype = has_row ? row_dtype : col_dtype;
  const bool is_fp8 = is_fp8_dtype(rep_dtype);
  // FP8 block scaling on Hopper requires TN layout (same as tensor scaling)
  const bool non_tn_fp8_ok = fp8_block ? false : nvte_is_non_tn_fp8_gemm_supported();

  // Helper to select columnwise storage.
  // storage_transposed=true: columnwise data is physically transposed relative to logical shape.
  // storage_transposed=false: columnwise data has logical shape (MXFP8).
  auto use_columnwise = [&](bool storage_transposed = true) {
    sel.dptr = static_cast<char *>(t->columnwise_data.dptr);
    sel.scale_inv = t->columnwise_scale_inv.dptr;
    sel.amax = t->columnwise_amax.dptr;
    sel.dtype = col_dtype;
    sel.rowwise = false;
    sel.storage_transposed = storage_transposed;
    sel.logical_tensor_shape = create_shape_info(t);
  };

  // Helper to select row-wise storage
  auto use_rowwise = [&]() {
    sel.dptr = static_cast<char *>(t->data.dptr);
    sel.scale_inv = t->scale_inv.dptr;
    sel.amax = t->amax.dptr;
    sel.dtype = row_dtype;
    sel.rowwise = true;
    sel.logical_tensor_shape = create_shape_info(t);
  };

  const auto choice =
      choose_grouped_operand_storage(trans, is_A, mxfp8, is_fp8, nvfp4, fp8_block, non_tn_fp8_ok,
                                     has_row, has_col, is_A ? "A" : "B");
  sel.trans = choice.trans;
  if (choice.use_rowwise) {
    use_rowwise();
  } else {
    use_columnwise(choice.storage_transposed);
  }
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

  // 2D by 2D is not supported
  NVTE_CHECK(!(a_scaling_mode == NVTE_BLOCK_SCALING_2D && b_scaling_mode == NVTE_BLOCK_SCALING_2D),
             "Only 1D by 1D, 1D by 2D, and 2D by 1D block scaling GEMM is supported, "
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

struct GroupedGemmWorkspace {
  GroupedGemmSetupWorkspace setup_workspace;
  void *cublas_workspace_ptr = nullptr;
  size_t num_tensors = 0;
};

inline GroupedGemmWorkspace setup_grouped_gemm_workspace(transformer_engine::Tensor *wspace_setup,
                                                         transformer_engine::Tensor *wspace_cublas,
                                                         size_t num_tensors) {
  const size_t setup_workspace_size = grouped_gemm_setup_workspace_size(num_tensors);
  const size_t cublas_workspace_size = kGroupedGemmCublasWorkspaceSize;
  void *setup_workspace_ptr = validate_and_get_workspace_ptr(wspace_setup, setup_workspace_size,
                                                             "Grouped GEMM setup workspace");
  void *cublas_workspace_ptr = validate_and_get_workspace_ptr(wspace_cublas, cublas_workspace_size,
                                                              "Grouped GEMM cuBLAS workspace");
  constexpr uintptr_t kSetupBaseAlignment = 16;
  NVTE_CHECK(reinterpret_cast<uintptr_t>(setup_workspace_ptr) % kSetupBaseAlignment == 0,
             "Grouped GEMM setup workspace must be ", kSetupBaseAlignment,
             "-byte aligned (cuBLAS requires this for pointer arrays).");
  auto setup_workspace = GroupedGemmSetupWorkspace::from_buffers(
      static_cast<char *>(setup_workspace_ptr), num_tensors);
  return {std::move(setup_workspace), cublas_workspace_ptr, num_tensors};
}

inline void execute_grouped_gemm(const GroupedGemmSetupWorkspace &setup_workspace,
                                 const GroupedOperandSelection &A_sel,
                                 const GroupedOperandSelection &B_sel,
                                 transformer_engine::DType d_dtype, size_t num_tensors,
                                 const GroupedGemmConfig &config, void *cublas_workspace_ptr,
                                 cudaStream_t stream) {
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

// Device helper: compute the element offset for tensor `idx` given shape metadata.
// Three cases:
//   1. Explicit per-tensor offset array provided  → use it directly.
//   2. Per-tensor first/last dims provided but no offsets → cumulative sum of (first*last) products.
//   3. Fully uniform shapes                        → idx * uniform_first * uniform_last.
__forceinline__ __device__ int64_t compute_grouped_tensor_offset(const TensorShapeInfo &meta,
                                                                 size_t idx) {
  if (meta.offsets) {
    return meta.offsets[idx];
  } else if (meta.first_dims != nullptr || meta.last_dims != nullptr) {
    // offset[i] = sum_{j < i} (first_dims[j] * last_dims[j])
    int64_t cumsum = 0;
    for (size_t i = 0; i < idx; i++) {
      int64_t f = meta.first_dims ? meta.first_dims[i] : meta.uniform_first;
      int64_t l = meta.last_dims ? meta.last_dims[i] : meta.uniform_last;
      cumsum += f * l;
    }
    return cumsum;
  } else {
    return static_cast<int64_t>(idx) * meta.uniform_first * meta.uniform_last;
  }
}

__forceinline__ __device__ int64_t padded_mxfp8_scale_inv_bytes(int64_t first, int64_t last,
                                                                bool rowwise) {
  namespace mxfp8_swizzle = transformer_engine::dispatch::mxfp8::swizzle;
  constexpr int64_t kMxfp8BlockSize = 32;
  // x is the dimension along which quantization is applied, y is other dimension
  const int64_t scale_tile_y = static_cast<int64_t>(mxfp8_swizzle::GEMM_SWIZZLED_SCALE_TILE_DIM_Y);
  const int64_t scale_tile_x = static_cast<int64_t>(mxfp8_swizzle::GEMM_SWIZZLED_SCALE_TILE_DIM_X);
  // Padded byte size of the swizzled MXFP8 scale_inv for a single tensor with data
  // shape (first, last). Rowwise scales use rows=first, cols=last; columnwise
  // scales swap the orientation since they are stored in column-major order.
  const int64_t scale_dim_y = rowwise ? first : last;
  const int64_t padded_scale_dim_y =
      ((scale_dim_y + scale_tile_y - 1) / scale_tile_y) * scale_tile_y;
  const int64_t data_dim_x = rowwise ? last : first;
  const int64_t scale_dim_x = (data_dim_x + kMxfp8BlockSize - 1) / kMxfp8BlockSize;
  const int64_t padded_scale_dim_x =
      ((scale_dim_x + scale_tile_x - 1) / scale_tile_x) * scale_tile_x;
  // MXFP8 scales are E8M0 (1 byte per element), so element count == byte count.
  return padded_scale_dim_y * padded_scale_dim_x;
}

__forceinline__ __device__ int64_t padded_nvfp4_scale_inv_bytes(int64_t first, int64_t last,
                                                                bool rowwise) {
  namespace mxfp8_swizzle = transformer_engine::dispatch::mxfp8::swizzle;
  constexpr int64_t kNvfp4BlockSize = 16;
  const int64_t scale_tile_y = static_cast<int64_t>(mxfp8_swizzle::GEMM_SWIZZLED_SCALE_TILE_DIM_Y);
  const int64_t scale_tile_x = static_cast<int64_t>(mxfp8_swizzle::GEMM_SWIZZLED_SCALE_TILE_DIM_X);
  const int64_t scale_dim_y = rowwise ? first : last;
  const int64_t data_dim_x = rowwise ? last : first;
  const int64_t padded_scale_dim_y =
      ((scale_dim_y + scale_tile_y - 1) / scale_tile_y) * scale_tile_y;
  const int64_t scale_dim_x = (data_dim_x + kNvfp4BlockSize - 1) / kNvfp4BlockSize;
  const int64_t padded_scale_dim_x =
      ((scale_dim_x + scale_tile_x - 1) / scale_tile_x) * scale_tile_x;
  // E4M3 scales are 1 byte per element.
  return padded_scale_dim_y * padded_scale_dim_x;
}

// FP8 block-scaling scale_inv layout matches the quantizer in get_scales() for logical dims.
__forceinline__ __device__ int64_t padded_block_1d_scale_inv_floats(int64_t first, int64_t last,
                                                                    bool rowwise) {
  constexpr int64_t kBlockLen = 128;
  constexpr int64_t kRowAlign = 4;
  const int64_t scale_dim_y = rowwise ? last : first;
  const int64_t data_dim_x = rowwise ? first : last;
  const int64_t y = (scale_dim_y + kBlockLen - 1) / kBlockLen;
  const int64_t x = ((data_dim_x + kRowAlign - 1) / kRowAlign) * kRowAlign;
  return y * x;
}

__forceinline__ __device__ int64_t padded_block_2d_scale_inv_floats(int64_t first, int64_t last,
                                                                    bool rowwise) {
  constexpr int64_t kBlockLen = 128;
  constexpr int64_t kRowAlign = 4;
  const int64_t scale_dim_y = rowwise ? first : last;
  const int64_t data_dim_x = rowwise ? last : first;
  const int64_t y = (scale_dim_y + kBlockLen - 1) / kBlockLen;
  const int64_t x_ceil = (data_dim_x + kBlockLen - 1) / kBlockLen;
  const int64_t x = ((x_ceil + kRowAlign - 1) / kRowAlign) * kRowAlign;
  return y * x;
}

// Generic prefix-sum of per-tensor padded scale_inv sizes — used to locate where
// tensor `idx`'s scales start in a contiguous grouped scale_inv buffer.
// `PaddedFn` is a callable (int64_t first, int64_t last) -> int64_t returning the
// recipe-specific padded size (bytes for MXFP8/NVFP4, floats for FP8 block scaling).
template <typename PaddedFn>
__forceinline__ __device__ int64_t compute_grouped_scale_inv_offset(const TensorShapeInfo &meta,
                                                                    size_t idx, PaddedFn padded) {
  if (meta.first_dims != nullptr || meta.last_dims != nullptr) {
    int64_t cumsum = 0;
    for (size_t i = 0; i < idx; i++) {
      const int64_t f = meta.first_dims ? meta.first_dims[i] : meta.uniform_first;
      const int64_t l = meta.last_dims ? meta.last_dims[i] : meta.uniform_last;
      cumsum += padded(f, l);
    }
    return cumsum;
  }
  return static_cast<int64_t>(idx) * padded(meta.uniform_first, meta.uniform_last);
}

// Linear scan to find which tensor contains the given row.
// Returns the tensor index and writes the exclusive end-row of that tensor to *out_tensor_row_end.
__forceinline__ __device__ int find_tensor_for_row(const int64_t *first_dims, int64_t uniform_first,
                                                   int row, int num_tensors,
                                                   int *out_tensor_row_end) {
  int offset = 0;
  for (int i = 0; i < num_tensors; i++) {
    int dim = first_dims ? static_cast<int>(first_dims[i]) : static_cast<int>(uniform_first);
    offset += dim;
    if (row < offset) {
      *out_tensor_row_end = offset;
      return i;
    }
  }
  *out_tensor_row_end = offset;
  return num_tensors - 1;
}

// Kernel that performs (optionally scaled) bias addition to Grouped GEMM output tensors.
// SM-filling grid with grid-stride over row chunks.
// 2D grid: blockIdx.x = SM-filling row blocks, blockIdx.y = column chunk.
// Each block grid-strides over kRowsPerBlock-sized row chunks, processing
// all chunks that map to it. Safe when sum(first_dims) <= total_rows.
template <typename T, int kVec, bool UseScale, int kBlockDim, int kRowsPerBlock>
__global__ void grouped_bias_add_kernel(char *__restrict__ d_base,
                                        const char *__restrict__ bias_base,
                                        const float *__restrict__ scale_base,
                                        TensorShapeInfo d_meta, int n, int total_rows,
                                        int num_tensors) {
  using VecStorage = transformer_engine::VectorizedStorage<T, kVec>;
  using VecType = typename VecStorage::LType;

  const int tid = static_cast<int>(threadIdx.x);
  const int row_bid = static_cast<int>(blockIdx.x);
  const int col_bid = static_cast<int>(blockIdx.y);
  const int row_grid_stride = static_cast<int>(gridDim.x);

  // Single-warp reduction to compute valid_rows = sum(first_dims).
  // kMaxGroups <= 64 so warp 0 (32 lanes) covers it with <=2 loads each.
  __shared__ int s_valid_rows;
  if (tid < 32) {
    int local_sum = 0;
    for (int i = tid; i < num_tensors; i += 32) {
      local_sum += d_meta.first_dims ? static_cast<int>(d_meta.first_dims[i])
                                     : static_cast<int>(d_meta.uniform_first);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (tid == 0) s_valid_rows = local_sum;
  }
  __syncthreads();
  const int valid_rows = s_valid_rows;

  const int block_cols = kBlockDim * kVec;
  const int col = col_bid * block_cols + tid * kVec;
  if (col >= n) return;

  T *__restrict__ d = reinterpret_cast<T *>(d_base);
  const T *__restrict__ bias = reinterpret_cast<const T *>(bias_base);

  // Grid-stride loop over row chunks.
  for (int chunk_start = row_bid * kRowsPerBlock; chunk_start < valid_rows;
       chunk_start += row_grid_stride * kRowsPerBlock) {
    const int row_start = chunk_start;
    const int row_end = min(row_start + kRowsPerBlock, valid_rows);

    // Linear scan to find the starting row's tensor and its boundary.
    int tensor_row_end;
    int tensor_idx = find_tensor_for_row(d_meta.first_dims, d_meta.uniform_first, row_start,
                                         num_tensors, &tensor_row_end);
    int bias_idx = tensor_idx * n;

    VecStorage b_in;

    // Walk tensor segments within this chunk's row range.
    int seg_start = row_start;
    while (seg_start < row_end) {
      while (tensor_idx < num_tensors - 1 && tensor_row_end <= seg_start) {
        tensor_idx++;
        bias_idx += n;
        int dim = d_meta.first_dims ? static_cast<int>(d_meta.first_dims[tensor_idx])
                                    : static_cast<int>(d_meta.uniform_first);
        tensor_row_end += dim;
      }
      b_in.scratch_.aligned = *reinterpret_cast<const VecType *>(bias + bias_idx + col);
      const int seg_end = min(tensor_row_end, row_end);

      for (int row = seg_start; row < seg_end; row++) {
        T *d_ptr = d + row * n + col;
        VecStorage d_in;
        d_in.scratch_.aligned = *reinterpret_cast<const VecType *>(d_ptr);

        [[maybe_unused]] float s_val;
        if constexpr (UseScale) s_val = scale_base[row];

#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          if constexpr (UseScale) {
            d_in.scratch_.separate[i] =
                static_cast<T>(fmaf(static_cast<float>(b_in.scratch_.separate[i]), s_val,
                                    static_cast<float>(d_in.scratch_.separate[i])));
          } else {
            d_in.scratch_.separate[i] =
                static_cast<T>(static_cast<float>(d_in.scratch_.separate[i]) +
                               static_cast<float>(b_in.scratch_.separate[i]));
          }
        }
        *reinterpret_cast<VecType *>(d_ptr) = d_in.scratch_.aligned;
      }

      seg_start = seg_end;
    }
  }
}

// Single kernel that sets up all GEMM parameters.
// Rationale: cuBLASLt grouped matmul API needs flat arrays of pointers and per-matrix dimensions,
// but NVTEGroupedTensor stores a single contiguous buffer + optional per-tensor offsets/shapes.
// We bridge the mismatch on GPU by computing per-group pointers and storage dims in one kernel.
__global__ void setup_grouped_gemm_kernel(
    // Output arrays
    void **A_ptrs, void **B_ptrs, void **C_ptrs, void **D_ptrs, int *a_rows, int *a_cols,
    int *b_rows, int *b_cols, int *d_rows, int *d_cols, float **alpha_ptrs, float **beta_ptrs,
    void **a_scale_inv_ptrs, void **b_scale_inv_ptrs,
    // Inputs
    char *a_base, char *b_base, char *c_base, char *d_base, TensorShapeInfo A_meta,
    TensorShapeInfo B_meta, TensorShapeInfo C_meta, TensorShapeInfo D_meta, size_t a_bits_per_elem,
    size_t b_bits_per_elem, size_t c_elem_size, size_t d_elem_size, float *alpha_ptr,
    float *beta_ptr, bool use_per_group_alpha_beta,
    // Scale inputs: for tensor scaling, pass float* and set mxfp8_base to nullptr
    // For MXFP8, pass nullptr for tensor_scale and set mxfp8_base
    float *a_scale_base, float *b_scale_base, bool a_rowwise, bool b_rowwise,
    bool a_storage_transposed, bool b_storage_transposed, NVTEScalingMode a_scaling_mode,
    NVTEScalingMode b_scaling_mode, size_t num_tensors,
    MultiTensorGroupGemmInputArgs a_multi_tensor_args,
    MultiTensorGroupGemmOutputArgs c_multi_tensor_args,
    MultiTensorGroupGemmOutputArgs d_multi_tensor_args,
    // NVFP4: per-group amax values and output buffer for computed alpha
    float *a_amax, float *b_amax, float *nvfp4_computed_alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tensors) return;

  // Get dimensions for this tensor (from array or uniform value)
  const bool has_a_multi_tensor = (a_base == nullptr);
  const bool has_c_multi_tensor = (c_base == nullptr);
  const bool has_d_multi_tensor = (d_base == nullptr);
  int64_t a_first = 0;
  int64_t a_last = 0;
  if (!has_a_multi_tensor) {
    a_first = A_meta.first_dims ? A_meta.first_dims[idx] : A_meta.uniform_first;
    a_last = A_meta.last_dims ? A_meta.last_dims[idx] : A_meta.uniform_last;
  }
  int64_t b_first = B_meta.first_dims ? B_meta.first_dims[idx] : B_meta.uniform_first;
  int64_t b_last = B_meta.last_dims ? B_meta.last_dims[idx] : B_meta.uniform_last;
  int64_t d_first = D_meta.first_dims ? D_meta.first_dims[idx] : D_meta.uniform_first;
  int64_t d_last = D_meta.last_dims ? D_meta.last_dims[idx] : D_meta.uniform_last;

  // Compute offsets (from explicit array, cumulative from per-tensor dims, or uniform)
  int64_t a_offset = has_a_multi_tensor ? 0 : compute_grouped_tensor_offset(A_meta, idx);
  int64_t b_offset = compute_grouped_tensor_offset(B_meta, idx);
  int64_t c_offset = compute_grouped_tensor_offset(C_meta, idx);
  int64_t d_offset = compute_grouped_tensor_offset(D_meta, idx);

  // Compute data pointers
  A_ptrs[idx] = has_a_multi_tensor ? a_multi_tensor_args.data_ptrs[idx]
                                   : (a_base + (a_offset * a_bits_per_elem) / 8);
  B_ptrs[idx] = b_base + (b_offset * b_bits_per_elem) / 8;
  C_ptrs[idx] =
      has_c_multi_tensor ? c_multi_tensor_args.data_ptrs[idx] : (c_base + c_offset * c_elem_size);
  D_ptrs[idx] =
      has_d_multi_tensor ? d_multi_tensor_args.data_ptrs[idx] : (d_base + d_offset * d_elem_size);

  // Compute storage dimensions for cuBLAS matrix layouts from logical dims.
  // Rowwise and MXFP8 columnwise storage use logical row-major layout, viewed as
  // column-major rows=last, cols=first. Transposed columnwise storage reverses this.
  if (has_a_multi_tensor) {
    a_rows[idx] = a_multi_tensor_args.rows[idx];
    a_cols[idx] = a_multi_tensor_args.cols[idx];
  } else if (a_storage_transposed) {
    a_rows[idx] = static_cast<int>(a_first);
    a_cols[idx] = static_cast<int>(a_last);
  } else {
    a_rows[idx] = static_cast<int>(a_last);
    a_cols[idx] = static_cast<int>(a_first);
  }
  if (b_storage_transposed) {
    b_rows[idx] = static_cast<int>(b_first);
    b_cols[idx] = static_cast<int>(b_last);
  } else {
    b_rows[idx] = static_cast<int>(b_last);
    b_cols[idx] = static_cast<int>(b_first);
  }
  if (has_d_multi_tensor) {
    d_rows[idx] = d_multi_tensor_args.rows[idx];
    d_cols[idx] = d_multi_tensor_args.cols[idx];
  } else {
    d_rows[idx] = static_cast<int>(d_last);
    d_cols[idx] = static_cast<int>(d_first);
  }

  // Fill alpha/beta pointers.
  // Hopper uses one shared alpha/beta scalar for all groups; Blackwell+ uses per-matrix scalars.
  // For NVFP4 on Blackwell+: compute per-group alpha that includes global scale (amax).
  // A's amax: grouped path indexes a_amax[idx]; discrete path reads amax_ptrs[idx].
  if (use_per_group_alpha_beta) {
    float a_amax_val = 0.0f;
    bool has_a_amax = false;
    if (has_a_multi_tensor) {
      auto *a_amax_p = static_cast<float *>(a_multi_tensor_args.amax_ptrs[idx]);
      if (a_amax_p != nullptr) {
        a_amax_val = *a_amax_p;
        has_a_amax = true;
      }
    } else if (a_amax != nullptr) {
      a_amax_val = a_amax[idx];
      has_a_amax = true;
    }
    if (has_a_amax && b_amax && nvfp4_computed_alpha) {
      constexpr float factor_inv = 1.0f / (6.0f * 6.0f * 448.0f * 448.0f);
      nvfp4_computed_alpha[idx] = alpha_ptr[idx] * a_amax_val * b_amax[idx] * factor_inv;
      alpha_ptrs[idx] = &nvfp4_computed_alpha[idx];
    } else {
      alpha_ptrs[idx] = alpha_ptr + idx;
    }
    beta_ptrs[idx] = beta_ptr + idx;
  } else {
    // Hopper: use single scalar for the whole grouped GEMM
    alpha_ptrs[idx] = alpha_ptr;
    beta_ptrs[idx] = beta_ptr;
  }

  // Fill scale pointers (per-matrix). For MXFP8/NVFP4 and FP8 block scaling, the per-expert
  // scale_inv buffer is padded to a layout that depends on the recipe — offsets are computed
  // from the same padded sizes that the quantizer uses at allocation, not from data_offset.
  //   NVTE_MXFP8_1D_SCALING : E8M0 byte stream; padded swizzled 128x4 tile, block_size=32.
  //   NVTE_NVFP4_1D_SCALING : E4M3 byte stream; padded swizzled 128x4 tile, block_size=16.
  //   NVTE_BLOCK_SCALING_1D : float32 array;    ceildiv(./128) * roundup(./4) per tensor.
  //   NVTE_BLOCK_SCALING_2D : float32 array;    ceildiv(./128) * roundup(ceildiv(./128), 4).
  //   otherwise (tensor)    : one float per tensor, indexed by tensor index.
  auto fill_scale_ptr = [&](void **ptrs, void *base, const TensorShapeInfo &meta, bool op_rowwise,
                            NVTEScalingMode op_scaling_mode) {
    int64_t byte_offset = -1;
    int64_t float_offset = -1;
    switch (op_scaling_mode) {
      case NVTE_MXFP8_1D_SCALING:
        byte_offset = compute_grouped_scale_inv_offset(meta, idx, [=](int64_t f, int64_t l) {
          return padded_mxfp8_scale_inv_bytes(f, l, op_rowwise);
        });
        break;
      case NVTE_NVFP4_1D_SCALING:
        byte_offset = compute_grouped_scale_inv_offset(meta, idx, [=](int64_t f, int64_t l) {
          return padded_nvfp4_scale_inv_bytes(f, l, op_rowwise);
        });
        break;
      case NVTE_BLOCK_SCALING_1D:
        float_offset = compute_grouped_scale_inv_offset(meta, idx, [=](int64_t f, int64_t l) {
          return padded_block_1d_scale_inv_floats(f, l, op_rowwise);
        });
        break;
      case NVTE_BLOCK_SCALING_2D:
        float_offset = compute_grouped_scale_inv_offset(meta, idx, [=](int64_t f, int64_t l) {
          return padded_block_2d_scale_inv_floats(f, l, op_rowwise);
        });
        break;
      default:
        float_offset = static_cast<int64_t>(idx);
        break;
    }
    if (byte_offset >= 0) {
      ptrs[idx] = static_cast<char *>(base) + byte_offset;
    } else {
      ptrs[idx] = static_cast<float *>(base) + float_offset;
    }
  };

  if (a_scale_base) {
    fill_scale_ptr(a_scale_inv_ptrs, a_scale_base, A_meta, a_rowwise, a_scaling_mode);
  } else {
    a_scale_inv_ptrs[idx] = a_multi_tensor_args.scale_inv_ptrs[idx];
  }
  if (b_scale_base) {
    fill_scale_ptr(b_scale_inv_ptrs, b_scale_base, B_meta, b_rowwise, b_scaling_mode);
  } else {
    b_scale_inv_ptrs[idx] = nullptr;
  }
}

// Launch the setup kernel to populate workspace arrays
inline void launch_grouped_gemm_setup(
    const GroupedGemmSetupWorkspace &ws, const GroupedOperandSelection &A_sel,
    const GroupedOperandSelection &B_sel, const transformer_engine::GroupedTensor *C,
    const transformer_engine::GroupedTensor *D, const transformer_engine::Tensor *alpha_tensor,
    const transformer_engine::Tensor *beta_tensor, bool use_per_group_alpha_beta,
    size_t num_tensors, cudaStream_t stream,
    const MultiTensorGroupGemmInputArgs &a_multi_tensor_args, const NVTETensor *C_list,
    const NVTETensor *D_list, char *a_base, transformer_engine::DType c_dtype,
    transformer_engine::DType d_dtype) {
  // Use logical shape info from selection; storage transposes are tracked separately.
  TensorShapeInfo A_meta = A_sel.logical_tensor_shape;
  TensorShapeInfo B_meta = B_sel.logical_tensor_shape;
  TensorShapeInfo C_meta{};
  TensorShapeInfo D_meta{};

  const bool has_d_multi_tensor = (D_list != nullptr);
  const bool has_c_multi_tensor = (C_list != nullptr) || has_d_multi_tensor;
  MultiTensorGroupGemmOutputArgs c_multi_tensor_args{};
  MultiTensorGroupGemmOutputArgs d_multi_tensor_args{};
  if (has_d_multi_tensor) {
    d_multi_tensor_args =
        build_grouped_gemm_multi_out_args(D_list, num_tensors, num_tensors, d_dtype, "D");
  }
  if (C_list != nullptr) {
    c_multi_tensor_args =
        build_grouped_gemm_multi_out_args(C_list, num_tensors, num_tensors, d_dtype, "C");
  } else if (has_d_multi_tensor) {
    c_multi_tensor_args = d_multi_tensor_args;
  }

  char *c_base = nullptr;
  char *d_base = nullptr;

  if (!has_c_multi_tensor) {
    NVTE_CHECK(C != nullptr && D != nullptr,
               "Grouped GEMM: C/D grouped tensors are required when no C list is provided");
    C_meta = TensorShapeInfo::create_shape_info_for_C(C, D);
    c_base = static_cast<char *>(C->data.dptr);
  }
  if (!has_d_multi_tensor) {
    NVTE_CHECK(D != nullptr,
               "Grouped GEMM: D grouped tensor is required when no D list is provided");
    D_meta = TensorShapeInfo::from_tensor(D);
    d_base = static_cast<char *>(D->data.dptr);
  }

  const size_t a_bits_per_elem = transformer_engine::typeToNumBits(A_sel.dtype);
  const size_t b_bits_per_elem = transformer_engine::typeToNumBits(B_sel.dtype);
  const size_t c_elem_size = transformer_engine::typeToSize(c_dtype);
  const size_t d_elem_size = transformer_engine::typeToSize(d_dtype);

  const int threads_per_block = 256;
  const int num_blocks = (num_tensors + threads_per_block - 1) / threads_per_block;

  // Pass scale buffers as void* and let the kernel interpret them via each operand's scaling mode.

  const bool a_rowwise = A_sel.rowwise;
  const bool b_rowwise = B_sel.rowwise;

  // NVFP4 alpha needs A's amax from either A_sel.amax (grouped) or amax_ptrs (discrete).
  const bool a_has_amax = (A_sel.amax != nullptr) ||
                          (A_sel.dptr == nullptr && a_multi_tensor_args.amax_ptrs[0] != nullptr);
  const bool needs_nvfp4_alpha = a_has_amax && (B_sel.amax != nullptr);

  setup_grouped_gemm_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      ws.A_ptrs, ws.B_ptrs, ws.C_ptrs, ws.D_ptrs, ws.a_rows, ws.a_cols, ws.b_rows, ws.b_cols,
      ws.d_rows, ws.d_cols, ws.alpha_ptrs, ws.beta_ptrs, ws.a_scale_inv_ptrs, ws.b_scale_inv_ptrs,
      A_sel.dptr, B_sel.dptr, c_base, d_base, A_meta, B_meta, C_meta, D_meta, a_bits_per_elem,
      b_bits_per_elem, c_elem_size, d_elem_size, static_cast<float *>(alpha_tensor->data.dptr),
      static_cast<float *>(beta_tensor->data.dptr), use_per_group_alpha_beta,
      reinterpret_cast<float *>(A_sel.scale_inv), reinterpret_cast<float *>(B_sel.scale_inv),
      a_rowwise, b_rowwise, A_sel.storage_transposed, B_sel.storage_transposed, A_sel.scaling_mode,
      B_sel.scaling_mode, num_tensors, a_multi_tensor_args, c_multi_tensor_args,
      d_multi_tensor_args, A_sel.amax ? static_cast<float *>(A_sel.amax) : nullptr,
      B_sel.amax ? static_cast<float *>(B_sel.amax) : nullptr,
      needs_nvfp4_alpha ? ws.nvfp4_computed_alpha : nullptr);

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

size_t nvte_get_grouped_gemm_setup_workspace_size(size_t num_tensors) {
  NVTE_API_CALL(nvte_get_grouped_gemm_setup_workspace_size);
  return grouped_gemm_setup_workspace_size(num_tensors);
}

void nvte_grouped_gemm(const NVTEGroupedTensor A, int transa, const NVTEGroupedTensor B, int transb,
                       const NVTEGroupedTensor C, NVTEGroupedTensor D, const NVTETensor alpha,
                       const NVTETensor beta, NVTETensor workspace_setup,
                       NVTETensor workspace_cublas, NVTEGroupedMatmulConfig config,
                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_grouped_gemm);
  using namespace transformer_engine;

  // Grouped GEMM requires Blackwell (SM100) or newer with cuBLAS 13.3+,
  // or Hopper (SM90) with cuBLAS 13.4+.
  check_grouped_gemm_requirements("nvte_grouped_gemm");

  const int current_device = transformer_engine::cuda::current_device();
  const int sm = transformer_engine::cuda::sm_arch(current_device);
  const bool use_per_group_alpha_beta = grouped_gemm_supports_per_group_alpha_beta(sm);

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
  GroupedMatmulConfig config_ = parse_grouped_gemm_config(config);

  // Validate inputs and outputs.
  const size_t num_tensors = validate_grouped_gemm_inputs(
      inputA->num_tensors, {inputA, inputB}, alpha_tensor, beta_tensor, use_per_group_alpha_beta);
  validate_grouped_gemm_outputs(num_tensors, inputA->dtype(), inputB->dtype(),
                                {inputC_raw, outputD});

  // If C is NULL, use D as C (valid when beta=0, cuBLAS won't read C data)
  const GroupedTensor *inputC = (inputC_raw != nullptr) ? inputC_raw : outputD;
  // num_tensors validated above.
  // Select operand storage (row-wise vs column-wise) and adjust transpose flags to
  // mirror the non-grouped GEMM logic for FP8 layout constraints.
  auto A_sel = select_grouped_operand(inputA, static_cast<bool>(transa), /*is_A=*/true);
  auto B_sel = select_grouped_operand(inputB, static_cast<bool>(transb), /*is_A=*/false);
  validate_nvfp4_grouped_gemm_support(A_sel, B_sel, use_per_group_alpha_beta);
  validate_fp8_block_grouped_gemm_support(A_sel, B_sel, sm);

  // NVFP4 global-scale alpha requires per-tensor amax for both operands; without it
  // the kernel silently drops the (amax_A * amax_B / factor) factor and produces
  // numerically wrong output.
  if (is_nvfp_scaling(A_sel.scaling_mode)) {
    NVTE_CHECK(A_sel.amax != nullptr, "Grouped GEMM: NVFP4 A is missing amax.");
    NVTE_CHECK(B_sel.amax != nullptr, "Grouped GEMM: NVFP4 B is missing amax.");
  }

  // Workspaces: setup (pointer arrays) and cuBLAS
  auto workspace = setup_grouped_gemm_workspace(wspace_setup, wspace_cublas, num_tensors);

  MultiTensorGroupGemmInputArgs a_multi_tensor_args{};
  launch_grouped_gemm_setup(workspace.setup_workspace, A_sel, B_sel, inputC, outputD, alpha_tensor,
                            beta_tensor, use_per_group_alpha_beta, num_tensors, stream,
                            a_multi_tensor_args, /*C_list=*/nullptr, /*D_list=*/nullptr, A_sel.dptr,
                            inputC->dtype(), outputD->dtype());

  // Compute average dimensions for heuristics
  // K dimension: if transa, K is A's first dim; if not, K is A's last dim
  // Use original inputA and transa for heuristics (not modified A_sel.trans)
  GroupedGemmConfig gemm_config;
  gemm_config.use_split_accumulator = config_.use_split_accumulator;
  gemm_config.use_fp8 = is_fp8_dtype(A_sel.dtype) || is_fp8_dtype(B_sel.dtype);
  gemm_config.use_per_group_alpha_beta = use_per_group_alpha_beta;
  gemm_config.alpha_dptr = alpha_tensor->data.dptr;
  gemm_config.beta_dptr = beta_tensor->data.dptr;
  gemm_config.avg_m = config_.avg_m.value_or(compute_avg_first_dim(outputD));
  gemm_config.avg_n = config_.avg_n.value_or(compute_avg_last_dim(outputD));
  gemm_config.avg_k =
      config_.avg_k.value_or(transa ? compute_avg_first_dim(inputA) : compute_avg_last_dim(inputA));
  gemm_config.sm_count = config_.sm_count;
  execute_grouped_gemm(workspace.setup_workspace, A_sel, B_sel, outputD->dtype(), num_tensors,
                       gemm_config, workspace.cublas_workspace_ptr, stream);
}

void nvte_grouped_gemm_with_discrete_inputA(const NVTETensor *A_list, size_t num_a_tensors,
                                            int transa, const NVTEGroupedTensor B, int transb,
                                            const NVTEGroupedTensor C, NVTEGroupedTensor D,
                                            const NVTETensor alpha, const NVTETensor beta,
                                            NVTETensor workspace_setup, NVTETensor workspace_cublas,
                                            NVTEGroupedMatmulConfig config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_grouped_gemm_with_discrete_inputA);
  using namespace transformer_engine;

  // Grouped GEMM requires Blackwell (SM100) or newer with cuBLAS 13.3+,
  // or Hopper (SM90) with cuBLAS 13.4+.
  check_grouped_gemm_requirements("nvte_grouped_gemm_with_discrete_inputA");

  const int current_device = transformer_engine::cuda::current_device();
  const int sm = transformer_engine::cuda::sm_arch(current_device);
  const bool use_per_group_alpha_beta = grouped_gemm_supports_per_group_alpha_beta(sm);

  NVTE_CHECK(A_list != nullptr, "Grouped GEMM: A_list is null.");
  NVTE_CHECK(num_a_tensors > 0, "Grouped GEMM: num_a_tensors must be > 0.");

  const GroupedTensor *inputB = convertNVTEGroupedTensorCheck(B);
  const GroupedTensor *inputC_raw = convertNVTEGroupedTensor(C);  // Can be NULL
  GroupedTensor *outputD = convertNVTEGroupedTensorCheck(D);
  const Tensor *alpha_tensor = convertNVTETensorCheck(alpha);
  const Tensor *beta_tensor = convertNVTETensorCheck(beta);
  Tensor *wspace_setup = convertNVTETensor(workspace_setup);
  Tensor *wspace_cublas = convertNVTETensor(workspace_cublas);

  // Parse config (if provided)
  GroupedMatmulConfig config_ = parse_grouped_gemm_config(config);

  // Validate inputs and outputs.
  const size_t num_tensors = validate_grouped_gemm_inputs(num_a_tensors, {inputB}, alpha_tensor,
                                                          beta_tensor, use_per_group_alpha_beta);

  // Validate A list and selection
  auto A_list_info =
      validate_grouped_gemm_multi_inputA_list(A_list, num_a_tensors, num_tensors, "A");
  auto is_supported_dtype = [](transformer_engine::DType dtype) {
    return dtype == transformer_engine::DType::kFloat8E4M3 ||
           dtype == transformer_engine::DType::kFloat8E5M2 ||
           dtype == transformer_engine::DType::kFloat4E2M1 ||
           dtype == transformer_engine::DType::kBFloat16 ||
           dtype == transformer_engine::DType::kFloat16;
  };
  NVTE_CHECK(
      is_supported_dtype(A_list_info.all_row ? A_list_info.row_dtype : A_list_info.col_dtype),
      "Grouped GEMM: A_list tensors must be FP8, NVFP4, BF16, or FP16.");

  // Cross-operand consistency (mirrors validate_grouped_gemm_inputs).
  const DType a_rep_dtype = A_list_info.all_row ? A_list_info.row_dtype : A_list_info.col_dtype;
  const bool a_is_fp8 = is_fp8_dtype(a_rep_dtype);
  const bool b_is_fp8 = is_fp8_dtype(inputB->dtype());
  const bool a_is_fp4 = a_rep_dtype == transformer_engine::DType::kFloat4E2M1;
  const bool b_is_fp4 = inputB->dtype() == transformer_engine::DType::kFloat4E2M1;
  const bool a_is_low_precision = a_is_fp8 || a_is_fp4;
  const bool b_is_low_precision = b_is_fp8 || b_is_fp4;
  NVTE_CHECK(a_is_low_precision == b_is_low_precision,
             "Grouped GEMM: A and B must both be low-precision (FP8/NVFP4) or both not.");
  NVTE_CHECK(a_is_fp8 == b_is_fp8, "Grouped GEMM: A and B must both be FP8 or both be non-FP8.");
  NVTE_CHECK(a_is_fp4 == b_is_fp4,
             "Grouped GEMM: A and B must both be NVFP4 or both be non-NVFP4.");
  NVTE_CHECK(is_compatible_grouped_scaling_mode(A_list_info.scaling_mode, inputB->scaling_mode),
             "Grouped GEMM: incompatible A/B scaling modes.");
  if (transformer_engine::is_mxfp_scaling(A_list_info.scaling_mode) ||
      transformer_engine::is_nvfp_scaling(A_list_info.scaling_mode)) {
    NVTE_CHECK(A_list_info.with_gemm_swizzled_scales,
               "Grouped GEMM: A scales must be swizzled for GEMM (MXFP8/NVFP4).");
    NVTE_CHECK(inputB->with_gemm_swizzled_scales,
               "Grouped GEMM: B scales must be swizzled for GEMM (MXFP8/NVFP4).");
  }

  validate_grouped_gemm_outputs(num_tensors, a_rep_dtype, inputB->dtype(), {inputC_raw, outputD});

  // If C is NULL, use D as C (valid when beta=0, cuBLAS won't read C data)
  const GroupedTensor *inputC = (inputC_raw != nullptr) ? inputC_raw : outputD;

  // Select operand storage for B (row-wise vs column-wise)
  auto B_sel = select_grouped_operand(inputB, static_cast<bool>(transb), /*is_A=*/false);

  GroupedOperandSelection A_sel{};
  A_sel.scaling_mode = A_list_info.scaling_mode;
  A_sel.with_gemm_swizzled_scales = A_list_info.with_gemm_swizzled_scales;
  A_sel.trans = static_cast<bool>(transa);
  validate_nvfp4_grouped_gemm_support(A_sel, B_sel, use_per_group_alpha_beta);
  validate_fp8_block_grouped_gemm_support(A_sel, B_sel, sm);

  const DType rep_dtype = A_list_info.all_row ? A_list_info.row_dtype : A_list_info.col_dtype;
  const bool is_fp8 = is_fp8_dtype(rep_dtype);
  const bool mxfp8 = transformer_engine::is_mxfp_scaling(A_list_info.scaling_mode);
  const bool nvfp4 = transformer_engine::is_nvfp_scaling(A_list_info.scaling_mode);
  const bool fp8_block = transformer_engine::is_fp8_block_scaling(A_list_info.scaling_mode);
  // FP8 block scaling on Hopper requires TN layout (matches select_grouped_operand logic for B).
  const bool non_tn_fp8_ok = fp8_block ? false : nvte_is_non_tn_fp8_gemm_supported();

  int64_t avg_first_dim = 0;
  int64_t avg_last_dim = 0;
  MultiTensorGroupGemmInputArgs a_multi_tensor_args{};

  const auto choice = choose_grouped_operand_storage(static_cast<bool>(transa), /*is_A=*/true,
                                                     mxfp8, is_fp8, nvfp4, fp8_block, non_tn_fp8_ok,
                                                     A_list_info.all_row, A_list_info.all_col, "A");
  A_sel.trans = choice.trans;
  A_sel.rowwise = choice.use_rowwise;
  A_sel.storage_transposed = choice.storage_transposed;
  if (choice.use_rowwise) {
    NVTE_CHECK(A_list_info.all_row, "Grouped GEMM: A_list is missing row-wise data");
    A_sel.dtype = A_list_info.row_dtype;
  } else {
    NVTE_CHECK(A_list_info.all_col, "Grouped GEMM: A_list is missing column-wise data");
    A_sel.dtype = A_list_info.col_dtype;
  }
  const bool requires_a_scale_inv = is_fp8 || nvfp4 || fp8_block;
  a_multi_tensor_args = build_grouped_gemm_multi_inputA_args(
      A_list, num_a_tensors, choice.use_rowwise, choice.storage_transposed, requires_a_scale_inv,
      &avg_first_dim, &avg_last_dim, "A");

  // Discrete A_list: per-tensor pointers come from `a_multi_tensor_args` (data/scale/amax).
  A_sel.scale_inv = nullptr;
  A_sel.dptr = nullptr;
  A_sel.amax = nullptr;

  if (nvfp4) {
    for (size_t i = 0; i < num_tensors; ++i) {
      NVTE_CHECK(a_multi_tensor_args.amax_ptrs[i] != nullptr, "Grouped GEMM: NVFP4 A_list tensor ",
                 i, " is missing amax.");
    }
    NVTE_CHECK(B_sel.amax != nullptr, "Grouped GEMM: NVFP4 B is missing amax.");
  }

  // Workspaces: setup (pointer arrays) and cuBLAS
  auto workspace = setup_grouped_gemm_workspace(wspace_setup, wspace_cublas, num_tensors);

  launch_grouped_gemm_setup(workspace.setup_workspace, A_sel, B_sel, inputC, outputD, alpha_tensor,
                            beta_tensor, use_per_group_alpha_beta, num_tensors, stream,
                            a_multi_tensor_args, /*C_list=*/nullptr, /*D_list=*/nullptr, nullptr,
                            inputC->dtype(), outputD->dtype());

  GroupedGemmConfig gemm_config;
  gemm_config.use_split_accumulator = config_.use_split_accumulator;
  gemm_config.use_fp8 = is_fp8_dtype(A_sel.dtype) || is_fp8_dtype(B_sel.dtype);
  gemm_config.use_per_group_alpha_beta = use_per_group_alpha_beta;
  gemm_config.alpha_dptr = alpha_tensor->data.dptr;
  gemm_config.beta_dptr = beta_tensor->data.dptr;
  gemm_config.avg_m = config_.avg_m.value_or(compute_avg_first_dim(outputD));
  gemm_config.avg_n =
      config_.avg_n.value_or(transb ? compute_avg_first_dim(inputB) : compute_avg_last_dim(inputB));
  gemm_config.avg_k = config_.avg_k.value_or(transa ? avg_first_dim : avg_last_dim);
  gemm_config.sm_count = config_.sm_count;
  execute_grouped_gemm(workspace.setup_workspace, A_sel, B_sel, outputD->dtype(), num_tensors,
                       gemm_config, workspace.cublas_workspace_ptr, stream);
}

void nvte_grouped_gemm_with_discrete_out(const NVTEGroupedTensor A, int transa,
                                         const NVTEGroupedTensor B, int transb,
                                         const NVTETensor *C_list, size_t num_c_tensors,
                                         NVTETensor *D_list, size_t num_d_tensors,
                                         const NVTETensor alpha, const NVTETensor beta,
                                         NVTETensor workspace_setup, NVTETensor workspace_cublas,
                                         NVTEGroupedMatmulConfig config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_grouped_gemm_with_discrete_out);
  using namespace transformer_engine;

  // Grouped GEMM requires Blackwell (SM100) or newer with cuBLAS 13.3+,
  // or Hopper (SM90) with cuBLAS 13.4+.
  check_grouped_gemm_requirements("nvte_grouped_gemm_with_discrete_out");

  const int current_device = transformer_engine::cuda::current_device();
  const int sm = transformer_engine::cuda::sm_arch(current_device);
  const bool use_per_group_alpha_beta = grouped_gemm_supports_per_group_alpha_beta(sm);

  NVTE_CHECK(D_list != nullptr, "Grouped GEMM: D_list is null.");
  NVTE_CHECK(num_d_tensors > 0, "Grouped GEMM: num_d_tensors must be > 0.");
  if (num_c_tensors > 0) {
    NVTE_CHECK(C_list != nullptr, "Grouped GEMM: C_list is null but num_c_tensors > 0.");
  }

  const GroupedTensor *inputA = convertNVTEGroupedTensorCheck(A);
  const GroupedTensor *inputB = convertNVTEGroupedTensorCheck(B);
  const Tensor *alpha_tensor = convertNVTETensorCheck(alpha);
  const Tensor *beta_tensor = convertNVTETensorCheck(beta);
  Tensor *wspace_setup = convertNVTETensor(workspace_setup);
  Tensor *wspace_cublas = convertNVTETensor(workspace_cublas);

  const Tensor *d0 = convertNVTETensorCheck(D_list[0]);
  const DType d_dtype = d0->dtype();

  const size_t num_tensors = validate_grouped_gemm_inputs(
      inputA->num_tensors, {inputA, inputB}, alpha_tensor, beta_tensor, use_per_group_alpha_beta);
  NVTE_CHECK(num_d_tensors == num_tensors, "Grouped GEMM: D_list must have num_tensors (",
             num_tensors, ") entries, got ", num_d_tensors);
  if (num_c_tensors > 0) {
    NVTE_CHECK(num_c_tensors == num_tensors, "Grouped GEMM: C_list must have num_tensors (",
               num_tensors, ") entries, got ", num_c_tensors);
  }
  validate_grouped_gemm_output_dtype(inputA->dtype(), inputB->dtype(), d_dtype, "D");

  // Parse config (if provided)
  GroupedMatmulConfig config_ = parse_grouped_gemm_config(config);

  // Select operand storage (row-wise vs column-wise) and adjust transpose flags to
  // mirror the non-grouped GEMM logic for FP8 layout constraints.
  auto A_sel = select_grouped_operand(inputA, static_cast<bool>(transa), /*is_A=*/true);
  auto B_sel = select_grouped_operand(inputB, static_cast<bool>(transb), /*is_A=*/false);
  validate_nvfp4_grouped_gemm_support(A_sel, B_sel, use_per_group_alpha_beta);
  validate_fp8_block_grouped_gemm_support(A_sel, B_sel, sm);

  // NVFP4 global-scale alpha requires per-tensor amax for both operands.
  if (is_nvfp_scaling(A_sel.scaling_mode)) {
    NVTE_CHECK(A_sel.amax != nullptr, "Grouped GEMM: NVFP4 A is missing amax.");
    NVTE_CHECK(B_sel.amax != nullptr, "Grouped GEMM: NVFP4 B is missing amax.");
  }

  // Workspaces: setup (pointer arrays) and cuBLAS
  auto workspace = setup_grouped_gemm_workspace(wspace_setup, wspace_cublas, num_tensors);

  MultiTensorGroupGemmInputArgs a_multi_tensor_args{};
  launch_grouped_gemm_setup(workspace.setup_workspace, A_sel, B_sel, /*C=*/nullptr, /*D=*/nullptr,
                            alpha_tensor, beta_tensor, use_per_group_alpha_beta, num_tensors,
                            stream, a_multi_tensor_args, C_list, D_list, A_sel.dptr, d_dtype,
                            d_dtype);

  GroupedGemmConfig gemm_config;
  gemm_config.use_split_accumulator = config_.use_split_accumulator;
  gemm_config.use_fp8 = is_fp8_dtype(A_sel.dtype) || is_fp8_dtype(B_sel.dtype);
  gemm_config.use_per_group_alpha_beta = use_per_group_alpha_beta;
  gemm_config.alpha_dptr = alpha_tensor->data.dptr;
  gemm_config.beta_dptr = beta_tensor->data.dptr;
  gemm_config.avg_m =
      config_.avg_m.value_or(transa ? compute_avg_last_dim(inputA) : compute_avg_first_dim(inputA));
  gemm_config.avg_n =
      config_.avg_n.value_or(transb ? compute_avg_first_dim(inputB) : compute_avg_last_dim(inputB));
  gemm_config.avg_k =
      config_.avg_k.value_or(transa ? compute_avg_first_dim(inputA) : compute_avg_last_dim(inputA));
  gemm_config.sm_count = config_.sm_count;
  execute_grouped_gemm(workspace.setup_workspace, A_sel, B_sel, d_dtype, num_tensors, gemm_config,
                       workspace.cublas_workspace_ptr, stream);
}

namespace {

void launch_grouped_bias_add(const transformer_engine::GroupedTensor *outputD,
                             const transformer_engine::GroupedTensor *bias_tensor,
                             const float *scale_ptr, bool use_scale, cudaStream_t stream) {
  using namespace transformer_engine;

  const char *api_name = use_scale ? "Grouped scaled bias add" : "Grouped bias add";

  NVTE_CHECK(outputD->num_tensors >= 1, api_name, ": number of tensors must be at least 1");
  NVTE_CHECK(outputD->num_tensors == bias_tensor->num_tensors, api_name,
             ": output and bias must have the same number of tensors");
  NVTE_CHECK(outputD->has_data(), api_name, ": output is missing row-wise data");
  NVTE_CHECK(bias_tensor->has_data(), api_name, ": bias is missing row-wise data");
  NVTE_CHECK(outputD->dtype() == bias_tensor->dtype(), api_name,
             ": output and bias must have matching dtypes");
  NVTE_CHECK(bias_tensor->all_same_first_dim(), api_name,
             ": bias must have uniform first dim (expected 1)");
  NVTE_CHECK(bias_tensor->get_common_first_dim() == 1, api_name, ": bias first dim must be 1");
  NVTE_CHECK(outputD->all_same_last_dim() && bias_tensor->all_same_last_dim(), api_name,
             ": requires uniform last dim for output and bias");
  NVTE_CHECK(outputD->get_common_last_dim() == bias_tensor->get_common_last_dim(), api_name,
             ": output and bias last dims must match");

  const TensorShapeInfo d_meta = TensorShapeInfo::from_tensor(outputD);

  const DType dtype = outputD->dtype();
  constexpr int kThreads = 128;

  const int num_tensors = static_cast<int>(outputD->num_tensors);
  NVTE_CHECK(num_tensors <= kMaxGroups, api_name, " supports at most ", kMaxGroups,
             " tensors, got ", num_tensors);
  const int total_rows = static_cast<int>(outputD->logical_shape.data[0]);
  const int n = static_cast<int>(outputD->get_common_last_dim());

  const size_t elem_size = typeToSize(dtype);
  const int kVec = (elem_size <= 2) ? 8 : 4;
  NVTE_CHECK(n % kVec == 0, api_name, ": requires last dim divisible by ", kVec);

  constexpr int kRowsPerBlock = 16;
  constexpr int kBlocksPerSM = 32;

  const int num_sms = transformer_engine::cuda::sm_count();

  const int block_cols = kThreads * kVec;
  const int col_blocks = (n + block_cols - 1) / block_cols;
  const int max_row_chunks = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
  const int row_blocks = std::min(max_row_chunks, num_sms * kBlocksPerSM / col_blocks);
  const dim3 grid(std::max(1, row_blocks), col_blocks);
  const dim3 block(kThreads);

  auto launch = [&](auto use_scale_tag) {
    constexpr bool kUseScale = decltype(use_scale_tag)::value;
    if (elem_size <= 2) {
      constexpr int kV = 8;
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(dtype, T, {
        grouped_bias_add_kernel<T, kV, kUseScale, kThreads, kRowsPerBlock>
            <<<grid, block, 0, stream>>>(static_cast<char *>(outputD->data.dptr),
                                         static_cast<const char *>(bias_tensor->data.dptr),
                                         scale_ptr, d_meta, n, total_rows, num_tensors);
      });
    } else {
      constexpr int kV = 4;
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(dtype, T, {
        grouped_bias_add_kernel<T, kV, kUseScale, kThreads, kRowsPerBlock>
            <<<grid, block, 0, stream>>>(static_cast<char *>(outputD->data.dptr),
                                         static_cast<const char *>(bias_tensor->data.dptr),
                                         scale_ptr, d_meta, n, total_rows, num_tensors);
      });
    }
  };

  if (use_scale) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

void nvte_grouped_bias_add(const NVTEGroupedTensor output, const NVTEGroupedTensor bias,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_grouped_bias_add);
  using namespace transformer_engine;
  const GroupedTensor *outputD = convertNVTEGroupedTensorCheck(output);
  const GroupedTensor *bias_tensor = convertNVTEGroupedTensorCheck(bias);
  launch_grouped_bias_add(outputD, bias_tensor, nullptr, false, stream);
}

void nvte_grouped_scaled_bias_add(const NVTEGroupedTensor output, const NVTEGroupedTensor bias,
                                  const NVTETensor scale, cudaStream_t stream) {
  NVTE_API_CALL(nvte_grouped_scaled_bias_add);
  using namespace transformer_engine;
  const GroupedTensor *outputD = convertNVTEGroupedTensorCheck(output);
  const GroupedTensor *bias_tensor = convertNVTEGroupedTensorCheck(bias);
  const Tensor *scale_tensor = convertNVTETensorCheck(scale);

  NVTE_CHECK(scale_tensor->data.dptr != nullptr,
             "Grouped scaled bias add: scale tensor must not be null");
  NVTE_CHECK(scale_tensor->dtype() == DType::kFloat32,
             "Grouped scaled bias add: scale must be float32");
  NVTE_CHECK(scale_tensor->data.shape.size() == 1,
             "Grouped scaled bias add: scale must be 1D, got ", scale_tensor->data.shape.size(),
             "D");
  const size_t total_rows = static_cast<size_t>(outputD->logical_shape.data[0]);
  NVTE_CHECK(scale_tensor->data.shape[0] == total_rows, "Grouped scaled bias add: scale size (",
             scale_tensor->data.shape[0], ") must equal total rows (", total_rows, ")");

  const float *scale_ptr = static_cast<const float *>(scale_tensor->data.dptr);
  launch_grouped_bias_add(outputD, bias_tensor, scale_ptr, true, stream);
}

#else  // CUBLAS_VERSION < CUBLAS_GROUPED_GEMM_VERSION

void nvte_grouped_gemm(const NVTEGroupedTensor A, int transa, const NVTEGroupedTensor B, int transb,
                       const NVTEGroupedTensor C, NVTEGroupedTensor D, const NVTETensor alpha,
                       const NVTETensor beta, NVTETensor workspace_setup,
                       NVTETensor workspace_cublas, NVTEGroupedMatmulConfig config,
                       cudaStream_t stream) {
  NVTE_ERROR("nvte_grouped_gemm requires cuBLAS 13.3+, but compile-time cuBLAS version is ",
             CUBLAS_VERSION, ". Please upgrade to cuBLAS 13.3 (shipped with CUDA 13.2) or newer.");
}

void nvte_grouped_gemm_with_discrete_inputA(const NVTETensor *A_list, size_t num_a_tensors,
                                            int transa, const NVTEGroupedTensor B, int transb,
                                            const NVTEGroupedTensor C, NVTEGroupedTensor D,
                                            const NVTETensor alpha, const NVTETensor beta,
                                            NVTETensor workspace_setup, NVTETensor workspace_cublas,
                                            NVTEGroupedMatmulConfig config, cudaStream_t stream) {
  NVTE_ERROR(
      "nvte_grouped_gemm_with_discrete_inputA requires cuBLAS 13.3+, but compile-time "
      "cuBLAS version is ",
      CUBLAS_VERSION, ". Please upgrade to cuBLAS 13.3 (shipped with CUDA 13.2) or newer.");
}

void nvte_grouped_gemm_with_discrete_out(const NVTEGroupedTensor A, int transa,
                                         const NVTEGroupedTensor B, int transb,
                                         const NVTETensor *C_list, size_t num_c_tensors,
                                         NVTETensor *D_list, size_t num_d_tensors,
                                         const NVTETensor alpha, const NVTETensor beta,
                                         NVTETensor workspace_setup, NVTETensor workspace_cublas,
                                         NVTEGroupedMatmulConfig config, cudaStream_t stream) {
  NVTE_ERROR(
      "nvte_grouped_gemm_with_discrete_out requires cuBLAS 13.3+, but compile-time "
      "cuBLAS version is ",
      CUBLAS_VERSION, ". Please upgrade to cuBLAS 13.3 (shipped with CUDA 13.2) or newer.");
}

void nvte_grouped_bias_add(const NVTEGroupedTensor output, const NVTEGroupedTensor bias,
                           cudaStream_t stream) {
  NVTE_ERROR("nvte_grouped_bias_add requires cuBLAS 13.3+, but compile-time cuBLAS version is ",
             CUBLAS_VERSION, ". Please upgrade to cuBLAS 13.3 (shipped with CUDA 13.2) or newer.");
}

void nvte_grouped_scaled_bias_add(const NVTEGroupedTensor output, const NVTEGroupedTensor bias,
                                  const NVTETensor scale, cudaStream_t stream) {
  NVTE_ERROR(
      "nvte_grouped_scaled_bias_add requires cuBLAS 13.3+, but compile-time cuBLAS version "
      "is ",
      CUBLAS_VERSION, ". Please upgrade to cuBLAS 13.3 (shipped with CUDA 13.2) or newer.");
}

size_t nvte_get_grouped_gemm_setup_workspace_size(size_t num_tensors) {
  NVTE_ERROR(
      "nvte_get_grouped_gemm_setup_workspace_size requires cuBLAS 13.3+, but compile-time cuBLAS "
      "version is ",
      CUBLAS_VERSION, ". Please upgrade to cuBLAS 13.3 (shipped with CUDA 13.2) or newer.");
  return 0;
}

#endif  // CUBLAS_VERSION >= CUBLAS_GROUPED_GEMM_VERSION

namespace {

__global__ void convert_int32_to_int64_kernel(const int32_t *src, int64_t *dst, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dst[idx] = static_cast<int64_t>(src[idx]);
}

// Like convert_int32_to_int64_kernel but scales each element by multiplier.
// Used to convert per-expert slice counts to per-expert row counts for multi-dim tensors.
__global__ void convert_int32_to_int64_with_multiplier_kernel(const int32_t *src, int64_t *dst,
                                                              size_t n, int64_t multiplier) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dst[idx] = static_cast<int64_t>(src[idx]) * multiplier;
}

// Computes exclusive prefix sums: offsets[0]=0, offsets[i]=sum(first_dims[0..i-1]*last_dim).
// Produces n_groups+1 values. Single-threaded sequential scan; n_groups is typically small.
__global__ void compute_grouped_tensor_offsets_kernel(const int64_t *first_dims, int64_t *offsets,
                                                      size_t n_groups, int64_t last_dim) {
  offsets[0] = 0;
  for (size_t i = 0; i < n_groups; i++) {
    offsets[i + 1] = offsets[i] + first_dims[i] * last_dim;
  }
}

}  // namespace

void nvte_convert_int32_to_int64(const int32_t *src, int64_t *dst, size_t n, cudaStream_t stream) {
  NVTE_API_CALL(nvte_convert_int32_to_int64);
  if (n == 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  convert_int32_to_int64_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void nvte_convert_int32_to_int64_with_multiplier(const int32_t *src, int64_t *dst, size_t n,
                                                 int64_t multiplier, cudaStream_t stream) {
  NVTE_API_CALL(nvte_convert_int32_to_int64_with_multiplier);
  if (n == 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  convert_int32_to_int64_with_multiplier_kernel<<<blocks, threads, 0, stream>>>(src, dst, n,
                                                                                multiplier);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void nvte_compute_grouped_tensor_offsets(const int64_t *first_dims, int64_t *offsets,
                                         size_t n_groups, int64_t last_dim, cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_grouped_tensor_offsets);
  // Always write at least offsets[0]=0 (needed even for n_groups==0).
  compute_grouped_tensor_offsets_kernel<<<1, 1, 0, stream>>>(first_dims, offsets, n_groups,
                                                             last_dim);
  NVTE_CHECK_CUDA(cudaGetLastError());
}
