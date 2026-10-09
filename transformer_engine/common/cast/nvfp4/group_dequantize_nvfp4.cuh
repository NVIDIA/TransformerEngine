/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_dequantize_nvfp4.cuh
 *  \brief CUDA kernels to dequantize grouped tensors from NVFP4.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../utils.cuh"
#include "../core/grouped_layout.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif  // FP4_TYPE_SUPPORTED

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace group_dequantize_kernel {
#if FP4_TYPE_SUPPORTED

// Index of the tensor that owns row `row`. The groups are stacked along the
// first dimension and share the last dimension `cols`.
template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_row_tensor_id(const size_t row, const size_t cols,
                                                    const size_t rows_per_tensor,
                                                    const size_t num_tensors,
                                                    const int64_t *const offsets) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    return row / rows_per_tensor;
  } else {
    return common::find_tensor_from_offsets(offsets, num_tensors, row * cols);
  }
}

// One thread dequantizes one 16-element block, with the same arithmetic as
// dequantize_nvfp4.cuh so the result matches per-tensor dequantization bitwise.
// Scales are read in the compact layout written by the grouped NVFP4 quantizer:
// one row of cols / 16 E4M3 scales per row of the stacked tensor.
template <typename OType, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(512)
    group_dequantize_fp4_kernel(const void *const input, OType *output, const fp8e4m3 *const scales,
                                const float *const amax, const bool amax_per_tensor,
                                const size_t num_rows, const size_t cols,
                                const size_t num_scale_cols, const size_t rows_per_tensor,
                                const size_t num_tensors, const int64_t *const offsets,
                                const int64_t *const first_dims) {
  if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
    // The dense scale indexing below needs each tensor's first dimension to be a multiple of
    // 128. Validate every tensor once, in the first block, as group_quantize_mxfp8.cuh does.
    if (blockIdx.x == 0) {
      for (size_t t = threadIdx.x; t < num_tensors; t += blockDim.x) {
        common::get_tensor_rows_num<SHAPE_REP>(t, num_rows, first_dims, num_tensors);
      }
    }
  }

  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t x = thread_idx % num_scale_cols;
  const size_t y = thread_idx / num_scale_cols;

  if (y >= num_rows) {
    return;
  }
  if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
    // Rows past the last group (unused capacity) are left untouched.
    if (static_cast<int64_t>(y * cols) >= offsets[num_tensors]) {
      return;
    }
  }

  union fp4vec {
    uint64_t vec;
    fp4e2m1x4 small_vec[4];
  };
  using OVec = Vec<OType, 4>;
  const uint64_t *const input_vectorized = reinterpret_cast<const uint64_t *>(input);
  OVec *output_vec = reinterpret_cast<OVec *>(output);

  const size_t tensor_id =
      get_row_tensor_id<SHAPE_REP>(y, cols, rows_per_tensor, num_tensors, offsets);
  const size_t my_index = x + y * num_scale_cols;
  const size_t my_output_index = my_index * 4;
  fp4vec value;
  value.vec = input_vectorized[my_index];
  const fp8e4m3 scale = scales[my_index];
  // Without an amax (second-level scaling disabled) the global scale is 1, as in
  // dequantize_nvfp4.cuh.
  constexpr float unit_global_scale_amax = 6.0f * 448.0f;
  float tensor_amax = unit_global_scale_amax;
  if (amax != nullptr) {
    tensor_amax = amax_per_tensor ? amax[tensor_id] : amax[0];
  }
  constexpr float factor_inv = 1.0f / unit_global_scale_amax;
  const float final_scale = static_cast<float>(scale) * tensor_amax * factor_inv;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float4 current = static_cast<float4>(value.small_vec[i]);
    OVec out;
    out.data.elt[0] = static_cast<OType>(current.x * final_scale);
    out.data.elt[1] = static_cast<OType>(current.y * final_scale);
    out.data.elt[2] = static_cast<OType>(current.z * final_scale);
    out.data.elt[3] = static_cast<OType>(current.w * final_scale);
    output_vec[my_output_index + i] = out;
  }
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace group_dequantize_kernel

inline void group_dequantize(const GroupedTensor *input, GroupedTensor *output,
                             cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace group_dequantize_kernel;

  NVTE_CHECK(input->has_data(),
             "Grouped NVFP4 dequantization reads rowwise data, but the input has none.");
  NVTE_CHECK(input->dtype() == DType::kFloat4E2M1, "Input must have FP4 type.");
  NVTE_CHECK(is_high_precision_dtype(output->dtype()), "Output must be in higher precision.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output logical shapes need to match.");
  NVTE_CHECK(!input->with_gemm_swizzled_scales,
             "Grouped NVFP4 dequantization requires scales in compact format.");
  NVTE_CHECK(input->scale_inv.dtype == DType::kFloat8E4M3,
             "Grouped NVFP4 dequantization requires E4M3 scales.");

  const size_t num_tensors = input->num_tensors;
  // A missing amax means second-level scaling is disabled.
  const bool has_amax = input->amax.has_data();
  const size_t amax_numel = has_amax ? input->amax.numel() : 0;
  if (has_amax) {
    NVTE_CHECK(input->amax.dtype == DType::kFloat32,
               "Grouped NVFP4 dequantization requires an FP32 amax.");
    NVTE_CHECK(amax_numel == 1 || amax_numel == num_tensors,
               "Grouped NVFP4 dequantization requires one amax or one amax per tensor (got ",
               amax_numel, " for ", num_tensors, " tensors).");
  }

  // The grouped NVFP4 quantizer stacks the groups along the first dimension,
  // so only a shared last dimension is supported. Every group's first
  // dimension must be a multiple of 128, which makes the padded per-tensor
  // scale layout identical to one dense [rows, cols / 16] array. This is
  // checked here for equal shapes; for a varying first dimension the kernel
  // reports it through common::get_tensor_rows_num.
  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (input->all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (input->all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else {
    NVTE_ERROR("Grouped NVFP4 dequantization requires all tensors to share the last dimension.");
  }

  const size_t num_rows = input->logical_shape.data[0];
  const size_t cols = input->logical_shape.data[1];
  constexpr size_t FP4_BLOCK_SIZE = 16;
  NVTE_CHECK(cols % 128 == 0,
             "Last dimension of a grouped NVFP4 tensor should be divisible by 128, but got ", cols,
             ".");
  size_t rows_per_tensor = 0;
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    NVTE_CHECK(num_rows % num_tensors == 0, "First dimension (", num_rows,
               ") must be divisible by the number of tensors (", num_tensors, ").");
    rows_per_tensor = num_rows / num_tensors;
    NVTE_CHECK(rows_per_tensor % 128 == 0,
               "Rows per tensor of a grouped NVFP4 tensor should be divisible by 128, but got ",
               rows_per_tensor, ".");
  } else {
    NVTE_CHECK(input->tensor_offsets.has_data() && input->first_dims.has_data(),
               "Grouped NVFP4 dequantization with a varying first dimension requires "
               "tensor_offsets and first_dims.");
  }

  const size_t num_scale_cols = cols / FP4_BLOCK_SIZE;
  NVTE_CHECK(input->scale_inv.numel() >= num_rows * num_scale_cols,
             "Grouped NVFP4 scale_inv is too small (", input->scale_inv.numel(), " < ",
             num_rows * num_scale_cols, ").");

  const size_t total = num_rows * num_scale_cols;
  const size_t threads = 512;
  const size_t blocks = DIVUP(total, threads);
  const bool amax_per_tensor = amax_numel == num_tensors && num_tensors > 1;
  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(input->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(input->first_dims.dptr);
  const fp8e4m3 *const scales_ptr = reinterpret_cast<const fp8e4m3 *>(input->scale_inv.dptr);
  const float *const amax_ptr =
      has_amax ? reinterpret_cast<const float *>(input->amax.dptr) : nullptr;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      output->dtype(), OType,
      if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
        group_dequantize_fp4_kernel<OType, ShapeRepresentation::SAME_BOTH_DIMS>
            <<<blocks, threads, 0, stream>>>(
                input->data.dptr, reinterpret_cast<OType *>(output->data.dptr), scales_ptr,
                amax_ptr, amax_per_tensor, num_rows, cols, num_scale_cols, rows_per_tensor,
                num_tensors, offsets_ptr, first_dims_ptr);
      } else {
        group_dequantize_fp4_kernel<OType, ShapeRepresentation::VARYING_FIRST_DIM>
            <<<blocks, threads, 0, stream>>>(
                input->data.dptr, reinterpret_cast<OType *>(output->data.dptr), scales_ptr,
                amax_ptr, amax_per_tensor, num_rows, cols, num_scale_cols, rows_per_tensor,
                num_tensors, offsets_ptr, first_dims_ptr);
      });  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_NVFP4_CUH_
