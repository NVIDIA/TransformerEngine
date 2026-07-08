/*************************************************************************
* Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
*
* See LICENSE for license information.
************************************************************************/

/*! \file grouped_gemm_common.h
 *  \brief Backend-agnostic types shared by the grouped GEMM dispatch layer and
 *         the cuBLAS grouped GEMM backend.
 *
 *  The grouped GEMM implementation is split across translation units:
 *    - grouped_gemm.cu           : common setup kernels, dispatch glue and the
 *                                  public nvte_grouped_* entry points.
 *    - cublaslt_grouped_gemm.cu  : the cuBLAS(Lt) backend (execute_grouped_gemm).
 *    - nvfp4_cutlass_grouped_gemm.cu : the CUTLASS NVFP4 backend.
 *
 *  This header carries only what more than one of those units needs: the
 *  cuBLAS version thresholds, the workspace constants, the device-side setup
 *  workspace layout, the per-operand selection/config structs, and the
 *  declaration of the cuBLAS backend entry point execute_grouped_gemm().
 */

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_GROUPED_GEMM_COMMON_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_GROUPED_GEMM_COMMON_H_

#include <cuda_runtime_api.h>
#include <transformer_engine/transformer_engine.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../common.h"
#include "../util/logging.h"

// MXFP8 support for grouped GEMM requires cuBLAS 13.3+
#define CUBLAS_MXFP8_GROUPED_GEMM_VERSION 130300

// Hopper (SM90) support for grouped GEMM requires cuBLAS 13.4+
#define CUBLAS_GROUPED_GEMM_HOPPER_VERSION 130400

// NVFP4 support for grouped GEMM requires cuBLAS 13.4+
#define CUBLAS_NVFP4_GROUPED_GEMM_VERSION 130400

// FP8 block scaling support for grouped GEMM requires cuBLAS 13.4+
#define CUBLAS_FP8_BLOCK_GROUPED_GEMM_VERSION 130400

// FP8 tensor scaling (per-tensor current/delayed scaling) support for grouped GEMM
// on Hopper (SM90) requires cuBLAS 13.5+
#define CUBLAS_FP8_TENSOR_SCALING_GROUPED_GEMM_HOPPER_VERSION 130500

// BF16 support for grouped GEMM requires cuBLAS 13.3+
#define CUBLAS_GROUPED_GEMM_VERSION 130300

namespace transformer_engine {
namespace grouped_gemm {

// Constants for grouped GEMM workspace.
static constexpr size_t kGroupedGemmAlignment = 256;
static constexpr size_t kGroupedGemmCublasWorkspaceSize = 32ull * 1024 * 1024;  // 32 MiB

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
  // Per-tensor scale_inv pointers (float* for tensor/FP8 block scaling, E8M0* for MXFP8,
  // E4M3* for NVFP4)
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
    // Int/float arrays follow without extra align_ptr(): cuBLAS only requires 16-byte
    // alignment for the pointer arrays above; int and float need just their natural
    // 4-byte alignment. The offset is 16-byte aligned after the last align_ptr() and
    // each subsequent place() adds N*4 bytes, so it stays a multiple of 4.
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
    // Additional alignment bytes is to take care of the case where the buffer
    // is not already aligned.
    return raw + alignment;
  }
};

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

// cuBLAS(Lt) grouped GEMM backend. Defined in cublaslt_grouped_gemm.cu and called from the
// dispatch layer in grouped_gemm.cu. Consumes the device-side setup_workspace arrays and the
// caller-provided cuBLAS workspace (also reused as scratch by the CUTLASS path). Only defined
// when the compile-time cuBLAS version supports grouped GEMM; the dispatch layer guards the
// call site with the same version check.
void execute_grouped_gemm(const GroupedGemmSetupWorkspace &setup_workspace,
                          const GroupedOperandSelection &A_sel,
                          const GroupedOperandSelection &B_sel, transformer_engine::DType d_dtype,
                          size_t num_tensors, const GroupedGemmConfig &config,
                          void *cublas_workspace_ptr, cudaStream_t stream);

}  // namespace grouped_gemm
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_GROUPED_GEMM_COMMON_H_
