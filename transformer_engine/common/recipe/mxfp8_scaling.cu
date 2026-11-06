/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <limits>

#include "../common.h"
#include "../util/ptx.cuh"
#include "../utils.cuh"
#include "partial_cast.h"

namespace transformer_engine {
namespace mxfp8_scaling_recipe {

constexpr int rowwise_row_padding = 128;  // Row padding of rowwise_scale and rowwise_amax
constexpr int rowwise_col_padding = 4;    // Column padding of rowwise_scale and rowwise_amax
constexpr int colwise_row_padding = 4;    // Row padding of colwise_scale and colwise_amax
constexpr int colwise_col_padding = 128;  // Column padding of colwise_scale and colwise_amax

constexpr int kRowsPerTile = 32;   // Rows each block processes
constexpr int kColsPerTile = 128;  // Columns each block processes

constexpr int kThreadsPerBlock = 128;

template <typename IType, bool kWithColumnwise>
__global__ void __launch_bounds__(kThreadsPerBlock)
    mxfp8_scaling_compute_partial_amax_kernel(const IType *input, IType *amax_rowwise,
                                              IType *amax_colwise, int amax_rowwise_stride,
                                              int amax_colwise_stride, int rows, int cols,
                                              size_t start_offset, size_t len) {
  __shared__ float smem_amax_rowwise[kRowsPerTile][kColsPerTile / 32];

  size_t end_offset = start_offset + len;
  const IType *input_minus_offset = input - start_offset;
  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;
  int c = blockIdx.x * kColsPerTile + threadIdx.x;
  int r = blockIdx.y * kRowsPerTile;

  float col_amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowsPerTile; i++) {
    size_t idx = r * cols + c;
    float row_amax = 0.0f;

    if (r < rows && c < cols && idx >= start_offset && idx < end_offset) {
      float abs_input = fabs(static_cast<float>(input_minus_offset[idx]));
      row_amax = fmaxf(row_amax, abs_input);
      if constexpr (kWithColumnwise) col_amax = fmaxf(col_amax, abs_input);
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
      float other_row_amax = __shfl_down_sync(0xFFFFFFFF, row_amax, delta);
      row_amax = fmaxf(row_amax, other_row_amax);
    }

    if (lane_idx == 0) {
      smem_amax_rowwise[i][warp_idx] = row_amax;
    }

    r++;
  }

  if constexpr (kWithColumnwise) {
    amax_colwise[blockIdx.y * amax_colwise_stride + c] = static_cast<IType>(col_amax);
  }

  __syncthreads();

  int r_ = threadIdx.x / (kColsPerTile / 32);  // rows in shared memory
  int c_ = threadIdx.x % (kColsPerTile / 32);  // cols in shared memory
  r = blockIdx.y * kRowsPerTile + r_;
  c = blockIdx.x * kColsPerTile / 32 + c_;
  amax_rowwise[r * amax_rowwise_stride + c] = static_cast<IType>(smem_amax_rowwise[r_][c_]);
}

template <typename IType, typename OType, bool kWithColumnwise>
__global__ void __launch_bounds__(kThreadsPerBlock)
    mxfp8_scaling_partial_cast_kernel(const IType *input, OType *output_rowwise,
                                      OType *output_colwise, const e8m0_t *scale_inv_rowwise,
                                      const e8m0_t *scale_inv_colwise, int scale_inv_rowwise_stride,
                                      int scale_inv_colwise_stride, int rows, int cols,
                                      size_t start_offset, size_t len) {
  __shared__ float smem_scales_rowwise[kRowsPerTile][kColsPerTile / 32];
  __shared__ float smem_scales_colwise[kWithColumnwise ? kColsPerTile : 1];

  // Load scales_rowwise
  {
    int r_ = threadIdx.x / (kColsPerTile / 32);  // rows in shared memory
    int c_ = threadIdx.x % (kColsPerTile / 32);  // cols in shared memory
    int r = blockIdx.y * kRowsPerTile + r_;
    int c = blockIdx.x * kColsPerTile / 32 + c_;
    size_t idx = r * scale_inv_rowwise_stride + c;
    smem_scales_rowwise[r_][c_] = ptx::exp2f_rcp<float>(scale_inv_rowwise[idx]);
  }

  // Load scales_colwise
  if constexpr (kWithColumnwise) {
    int c_ = threadIdx.x;
    int r = blockIdx.y * kRowsPerTile / 32;
    int c = blockIdx.x * kColsPerTile + c_;
    size_t idx = r * scale_inv_colwise_stride + c;
    smem_scales_colwise[c_] = ptx::exp2f_rcp<float>(scale_inv_colwise[idx]);
  }

  __syncthreads();

  size_t end_offset = start_offset + len;
  const IType *input_minus_offset = input - start_offset;
  OType *output_rowwise_minus_offset = output_rowwise - start_offset;
  OType *output_colwise_minus_offset = kWithColumnwise ? output_colwise - start_offset : nullptr;
  int warp_idx = threadIdx.x / 32;
  // int lane_idx = threadIdx.x % 32;
  int c = blockIdx.x * kColsPerTile + threadIdx.x;
  int r = blockIdx.y * kRowsPerTile;

#pragma unroll
  for (int i = 0; i < kRowsPerTile; i++) {
    size_t idx = r * cols + c;

    if (r < rows && c < cols && idx >= start_offset && idx < end_offset) {
      float inp = static_cast<float>(input_minus_offset[idx]);
      OType out_rowwise = static_cast<OType>(inp * smem_scales_rowwise[i][warp_idx]);
      output_rowwise_minus_offset[idx] = out_rowwise;
      if constexpr (kWithColumnwise) {
        OType out_colwise = static_cast<OType>(inp * smem_scales_colwise[threadIdx.x]);
        output_colwise_minus_offset[idx] = out_colwise;
      }
    }

    r++;
  }
}

void mxfp8_scaling_compute_partial_amax(const Tensor input, Tensor amax_rowwise,
                                        Tensor amax_colwise, int rows, int cols,
                                        size_t start_offset, cudaStream_t stream) {
  NVTE_CHECK(rows % 32 == 0, "rows must be divisible by 32");
  NVTE_CHECK(cols % 32 == 0, "cols must be divisible by 32");

  NVTE_CHECK(input.data.shape.size() == 1, "input must be a 1D tensor");
  NVTE_CHECK(start_offset + input.data.shape[0] <= static_cast<size_t>(rows) * cols,
             "Invalid start_offset");

  NVTE_CHECK(amax_rowwise.data.shape.size() == 2, "amax_rowwise must be a 2D tensor");
  NVTE_CHECK(amax_rowwise.data.shape[0] % rowwise_row_padding == 0,
             "Wrong padding of amax_rowwise's rows");
  NVTE_CHECK(amax_rowwise.data.shape[0] >= rows, "Invalid rows");
  NVTE_CHECK(amax_rowwise.data.shape[1] % rowwise_col_padding == 0,
             "Wrong padding of amax_rowwise's cols");
  NVTE_CHECK(amax_rowwise.data.shape[1] >= cols / 32, "Invalid cols");
  NVTE_CHECK(amax_rowwise.dtype() == input.dtype(), "Wrong dtype of amax_rowwise");

  const bool with_columnwise = amax_colwise.has_data();
  if (with_columnwise) {
    NVTE_CHECK(amax_colwise.data.shape.size() == 2, "amax_colwise must be a 2D tensor");
    NVTE_CHECK(amax_colwise.data.shape[0] % colwise_row_padding == 0,
               "Wrong padding of amax_colwise's rows");
    NVTE_CHECK(amax_colwise.data.shape[0] >= rows / 32, "Invalid rows");
    NVTE_CHECK(amax_colwise.data.shape[1] % colwise_col_padding == 0,
               "Wrong padding of amax_colwise's cols");
    NVTE_CHECK(amax_colwise.data.shape[1] >= cols, "Invalid cols");
    NVTE_CHECK(amax_colwise.dtype() == input.dtype(), "Wrong dtype of amax_colwise");
  }

  int blocks_x = (cols + kColsPerTile - 1) / kColsPerTile;
  int blocks_y = (rows + kRowsPerTile - 1) / kRowsPerTile;
  dim3 grid(blocks_x, blocks_y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          with_columnwise, kWithColumnwise,
          mxfp8_scaling_compute_partial_amax_kernel<IType, kWithColumnwise>
          <<<grid, kColsPerTile, 0, stream>>>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<IType *>(amax_rowwise.data.dptr),
              with_columnwise ? reinterpret_cast<IType *>(amax_colwise.data.dptr) : nullptr,
              amax_rowwise.data.shape[1], with_columnwise ? amax_colwise.data.shape[1] : 0, rows,
              cols, start_offset, input.data.shape[0]);))
}

void partial_cast(const Tensor &input, Tensor &output, size_t h, size_t w, size_t start_offset,
                  cudaStream_t stream) {
  NVTE_CHECK(h <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                 w <= static_cast<size_t>(std::numeric_limits<int>::max()),
             "MXFP8 matrix dimensions exceed int range");
  const int rows = static_cast<int>(h);
  const int cols = static_cast<int>(w);
  const auto &output_rowwise = output.data;
  const auto &output_colwise = output.columnwise_data;
  const auto &scale_inv_rowwise = output.scale_inv;
  const auto &scale_inv_colwise = output.columnwise_scale_inv;
  const bool with_columnwise = output_colwise.has_data();
  NVTE_CHECK(scale_inv_rowwise.dptr != nullptr, "MXFP8 requires rowwise inverse scales");
  NVTE_CHECK(rows % 32 == 0, "rows must be divisible by 32");
  NVTE_CHECK(cols % 32 == 0, "cols must be divisible by 32");

  NVTE_CHECK(input.data.shape.size() == 1, "input must be a 1D tensor");
  NVTE_CHECK(start_offset + input.data.shape[0] <= static_cast<size_t>(rows) * cols,
             "Invalid start_offset");

  NVTE_CHECK(output_rowwise.shape.size() == 1, "output_rowwise must be a 1D tensor");
  NVTE_CHECK(output_rowwise.shape[0] == input.data.shape[0],
             "Size of input and output_rowwise mismatch");

  NVTE_CHECK(output_rowwise.dtype == DType::kFloat8E4M3, "output_rowwise should be E4M3");

  NVTE_CHECK(scale_inv_rowwise.shape.size() == 2, "scale_inv_rowwise must be a 2D tensor");
  NVTE_CHECK(scale_inv_rowwise.shape[0] % rowwise_row_padding == 0,
             "Wrong padding of scale_inv_rowwise's rows");
  NVTE_CHECK(scale_inv_rowwise.shape[0] >= rows, "Invalid rows");
  NVTE_CHECK(scale_inv_rowwise.shape[1] % rowwise_col_padding == 0,
             "Wrong padding of scale_inv_rowwise's cols");
  NVTE_CHECK(scale_inv_rowwise.shape[1] >= cols / 32, "Invalid cols");
  NVTE_CHECK(scale_inv_rowwise.dtype == DType::kFloat8E8M0, "Wrong dtype of scale_inv_rowwise");

  if (with_columnwise) {
    NVTE_CHECK(output_colwise.dptr != nullptr, "MXFP8 requires columnwise output data");
    NVTE_CHECK(output_colwise.shape.size() == 1, "output_colwise must be a 1D tensor");
    NVTE_CHECK(output_colwise.shape[0] == input.data.shape[0],
               "Size of input and output_colwise mismatch");
    NVTE_CHECK(output_colwise.dtype == DType::kFloat8E4M3, "output_colwise should be E4M3");
    NVTE_CHECK(scale_inv_colwise.dptr != nullptr && scale_inv_colwise.numel() != 0,
               "Columnwise output requires columnwise scales");
    NVTE_CHECK(scale_inv_colwise.shape.size() == 2, "scale_inv_colwise must be a 2D tensor");
    NVTE_CHECK(scale_inv_colwise.shape[0] % colwise_row_padding == 0,
               "Wrong padding of scale_inv_colwise's rows");
    NVTE_CHECK(scale_inv_colwise.shape[0] >= rows / 32, "Invalid rows");
    NVTE_CHECK(scale_inv_colwise.shape[1] % colwise_col_padding == 0,
               "Wrong padding of scale_inv_colwise's cols");
    NVTE_CHECK(scale_inv_colwise.shape[1] >= cols, "Invalid cols");
    NVTE_CHECK(scale_inv_colwise.dtype == DType::kFloat8E8M0, "Wrong dtype of scale_inv_colwise");
  }

  int blocks_x = (cols + kColsPerTile - 1) / kColsPerTile;
  int blocks_y = (rows + kRowsPerTile - 1) / kRowsPerTile;
  dim3 grid(blocks_x, blocks_y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          with_columnwise, kWithColumnwise,
          mxfp8_scaling_partial_cast_kernel<IType, fp8e4m3, kWithColumnwise>
          <<<grid, kColsPerTile, 0, stream>>>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<fp8e4m3 *>(output_rowwise.dptr),
              with_columnwise ? reinterpret_cast<fp8e4m3 *>(output_colwise.dptr) : nullptr,
              reinterpret_cast<const e8m0_t *>(scale_inv_rowwise.dptr),
              with_columnwise ? reinterpret_cast<const e8m0_t *>(scale_inv_colwise.dptr) : nullptr,
              scale_inv_rowwise.shape[1], with_columnwise ? scale_inv_colwise.shape[1] : 0, rows,
              cols, start_offset, input.data.shape[0]);))
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace mxfp8_scaling_recipe
}  // namespace transformer_engine

void nvte_mxfp8_scaling_compute_partial_amax(const NVTETensor input, NVTETensor amax_rowwise,
                                             NVTETensor amax_colwise, int rows, int cols,
                                             size_t start_offset, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mxfp8_scaling_compute_partial_amax);
  using namespace transformer_engine;
  mxfp8_scaling_recipe::mxfp8_scaling_compute_partial_amax(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(amax_rowwise),
      *convertNVTETensorCheck(amax_colwise), rows, cols, start_offset, stream);
}

void nvte_mxfp8_scaling_partial_cast(const NVTETensor input, NVTETensor output_rowwise,
                                     NVTETensor output_colwise, const NVTETensor scale_inv_rowwise,
                                     const NVTETensor scale_inv_colwise, int rows, int cols,
                                     size_t start_offset, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mxfp8_scaling_partial_cast);
  using namespace transformer_engine;
  const auto &row = *convertNVTETensorCheck(output_rowwise);
  const auto &col = *convertNVTETensorCheck(output_colwise);
  const auto &row_scale = *convertNVTETensorCheck(scale_inv_rowwise);
  NVTE_CHECK(row.dtype() == DType::kFloat8E4M3 || row.dtype() == DType::kByte,
             "output_rowwise should be E4M3 or uint8");
  NVTE_CHECK(row_scale.dtype() == DType::kByte || row_scale.dtype() == DType::kFloat8E8M0,
             "Wrong dtype of scale_inv_rowwise");
  TensorWrapper view(NVTE_MXFP8_1D_SCALING);
  view.set_rowwise_data(row.data.dptr, DType::kFloat8E4M3, row.data.shape)
      .set_rowwise_scale_inv(row_scale.data.dptr, DType::kFloat8E8M0, row_scale.data.shape);
  // Normalize empty views by their output size, even if they have a backing
  // pointer. Columnwise scale metadata never selects whether to produce data.
  if (col.has_data() && col.numel() != 0) {
    const auto &col_scale = *convertNVTETensorCheck(scale_inv_colwise);
    NVTE_CHECK(col.dtype() == DType::kFloat8E4M3 || col.dtype() == DType::kByte,
               "output_colwise should be E4M3 or uint8");
    NVTE_CHECK(col_scale.dtype() == DType::kByte || col_scale.dtype() == DType::kFloat8E8M0,
               "Wrong dtype of scale_inv_colwise");
    view.set_columnwise_data(col.data.dptr, DType::kFloat8E4M3, col.data.shape)
        .set_columnwise_scale_inv(col_scale.data.dptr, DType::kFloat8E8M0, col_scale.data.shape);
  }
  NVTE_CHECK(rows > 0 && cols > 0, "Invalid logical matrix dimensions");
  nvte_partial_cast(input, view.data(), static_cast<size_t>(rows), static_cast<size_t>(cols),
                    start_offset, stream);
}
