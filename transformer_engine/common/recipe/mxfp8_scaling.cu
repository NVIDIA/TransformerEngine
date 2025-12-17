/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include "../common.h"
#include "../util/ptx.cuh"
#include "../utils.cuh"

namespace transformer_engine {
namespace mxfp8_scaling_recipe {

constexpr int rowwise_row_padding = 128;  // Row padding of rowwise_scale and rowwise_amax
constexpr int rowwise_col_padding = 4;    // Column padding of rowwise_scale and rowwise_amax
constexpr int colwise_row_padding = 4;    // Row padding of colwise_scale and colwise_amax
constexpr int colwise_col_padding = 128;  // Column padding of colwise_scale and colwise_amax

constexpr int kRowsPerTile = 32;   // Rows each block processes
constexpr int kColsPerTile = 128;  // Columns each block processes

constexpr int kThreadsPerBlock = 128;

template <typename IType>
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
      col_amax = fmaxf(col_amax, abs_input);
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

  amax_colwise[blockIdx.y * amax_colwise_stride + c] = static_cast<IType>(col_amax);

  __syncthreads();

  int r_ = threadIdx.x / (kColsPerTile / 32);  // rows in shared memory
  int c_ = threadIdx.x % (kColsPerTile / 32);  // cols in shared memory
  r = blockIdx.y * kRowsPerTile + r_;
  c = blockIdx.x * kColsPerTile / 32 + c_;
  amax_rowwise[r * amax_rowwise_stride + c] = static_cast<IType>(smem_amax_rowwise[r_][c_]);
}

template <typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    mxfp8_scaling_partial_cast_kernel(const IType *input, OType *output_rowwise,
                                      OType *output_colwise, const e8m0_t *scale_inv_rowwise,
                                      const e8m0_t *scale_inv_colwise, int scale_inv_rowwise_stride,
                                      int scale_inv_colwise_stride, int rows, int cols,
                                      size_t start_offset, size_t len) {
  __shared__ float smem_scales_rowwise[kRowsPerTile][kColsPerTile / 32];
  __shared__ float smem_scales_colwise[kColsPerTile];

  // Load scales_rowwise
  {
    int r_ = threadIdx.x / (kColsPerTile / 32);  // rows in shared memory
    int c_ = threadIdx.x % (kColsPerTile / 32);  // cols in shared memory
    int r = blockIdx.y * kRowsPerTile + r_;
    int c = blockIdx.x * kColsPerTile / 32 + c_;
    size_t idx = r * scale_inv_rowwise_stride + c;
    smem_scales_rowwise[r_][c_] = ptx::exp2f_rcp(scale_inv_rowwise[idx]);
  }

  // Load scales_colwise
  {
    int c_ = threadIdx.x;
    int r = blockIdx.y * kRowsPerTile / 32;
    int c = blockIdx.x * kColsPerTile + c_;
    size_t idx = r * scale_inv_colwise_stride + c;
    smem_scales_colwise[c_] = ptx::exp2f_rcp(scale_inv_colwise[idx]);
  }

  __syncthreads();

  size_t end_offset = start_offset + len;
  const IType *input_minus_offset = input - start_offset;
  OType *output_rowwise_minus_offset = output_rowwise - start_offset;
  OType *output_colwise_minus_offset = output_colwise - start_offset;
  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;
  int c = blockIdx.x * kColsPerTile + threadIdx.x;
  int r = blockIdx.y * kRowsPerTile;

#pragma unroll
  for (int i = 0; i < kRowsPerTile; i++) {
    size_t idx = r * cols + c;

    if (r < rows && c < cols && idx >= start_offset && idx < end_offset) {
      float inp = static_cast<float>(input_minus_offset[idx]);
      OType out_rowwise = static_cast<OType>(inp * smem_scales_rowwise[i][warp_idx]);
      OType out_colwise = static_cast<OType>(inp * smem_scales_colwise[threadIdx.x]);
      output_rowwise_minus_offset[idx] = out_rowwise;
      output_colwise_minus_offset[idx] = out_colwise;
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

  NVTE_CHECK(amax_colwise.data.shape.size() == 2, "amax_colwise must be a 2D tensor");
  NVTE_CHECK(amax_colwise.data.shape[0] % colwise_row_padding == 0,
             "Wrong padding of amax_colwise's rows");
  NVTE_CHECK(amax_colwise.data.shape[0] >= rows / 32, "Invalid rows");
  NVTE_CHECK(amax_colwise.data.shape[1] % colwise_col_padding == 0,
             "Wrong padding of amax_colwise's cols");
  NVTE_CHECK(amax_colwise.data.shape[1] >= cols, "Invalid cols");
  NVTE_CHECK(amax_colwise.dtype() == input.dtype(), "Wrong dtype of amax_colwise");

  int blocks_x = (cols + kColsPerTile - 1) / kColsPerTile;
  int blocks_y = (rows + kRowsPerTile - 1) / kRowsPerTile;
  dim3 grid(blocks_x, blocks_y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      mxfp8_scaling_compute_partial_amax_kernel<IType><<<grid, kColsPerTile, 0, stream>>>(
          reinterpret_cast<const IType *>(input.data.dptr),
          reinterpret_cast<IType *>(amax_rowwise.data.dptr),
          reinterpret_cast<IType *>(amax_colwise.data.dptr), amax_rowwise.data.shape[1],
          amax_colwise.data.shape[1], rows, cols, start_offset, input.data.shape[0]);)
}

void mxfp8_scaling_partial_cast(const Tensor input, Tensor output_rowwise, Tensor output_colwise,
                                const Tensor scale_inv_rowwise, const Tensor scale_inv_colwise,
                                int rows, int cols, size_t start_offset, cudaStream_t stream) {
  NVTE_CHECK(rows % 32 == 0, "rows must be divisible by 32");
  NVTE_CHECK(cols % 32 == 0, "cols must be divisible by 32");

  NVTE_CHECK(input.data.shape.size() == 1, "input must be a 1D tensor");
  NVTE_CHECK(start_offset + input.data.shape[0] <= static_cast<size_t>(rows) * cols,
             "Invalid start_offset");

  NVTE_CHECK(output_rowwise.data.shape.size() == 1, "output_rowwise must be a 1D tensor");
  NVTE_CHECK(output_colwise.data.shape.size() == 1, "output_colwise must be a 1D tensor");
  NVTE_CHECK(output_rowwise.data.shape[0] == input.data.shape[0],
             "Size of input and output_rowwise mismatch");
  NVTE_CHECK(output_colwise.data.shape[0] == input.data.shape[0],
             "Size of input and output_colwise mismatch");

  NVTE_CHECK(output_rowwise.dtype() == DType::kFloat8E4M3 || output_rowwise.dtype() == DType::kByte,
             "output_rowwise should be e4m3 or uint8");
  NVTE_CHECK(output_colwise.dtype() == DType::kFloat8E4M3 || output_colwise.dtype() == DType::kByte,
             "output_colwise should be e4m3 or uint8");

  NVTE_CHECK(scale_inv_rowwise.data.shape.size() == 2, "scale_inv_rowwise must be a 2D tensor");
  NVTE_CHECK(scale_inv_rowwise.data.shape[0] % rowwise_row_padding == 0,
             "Wrong padding of scale_inv_rowwise's rows");
  NVTE_CHECK(scale_inv_rowwise.data.shape[0] >= rows, "Invalid rows");
  NVTE_CHECK(scale_inv_rowwise.data.shape[1] % rowwise_col_padding == 0,
             "Wrong padding of scale_inv_rowwise's cols");
  NVTE_CHECK(scale_inv_rowwise.data.shape[1] >= cols / 32, "Invalid cols");
  NVTE_CHECK(scale_inv_rowwise.dtype() == DType::kByte, "Wrong dtype of scale_inv_rowwise");

  NVTE_CHECK(scale_inv_colwise.data.shape.size() == 2, "scale_inv_colwise must be a 2D tensor");
  NVTE_CHECK(scale_inv_colwise.data.shape[0] % colwise_row_padding == 0,
             "Wrong padding of scale_inv_colwise's rows");
  NVTE_CHECK(scale_inv_colwise.data.shape[0] >= rows / 32, "Invalid rows");
  NVTE_CHECK(scale_inv_colwise.data.shape[1] % colwise_col_padding == 0,
             "Wrong padding of scale_inv_colwise's cols");
  NVTE_CHECK(scale_inv_colwise.data.shape[1] >= cols, "Invalid cols");
  NVTE_CHECK(scale_inv_colwise.dtype() == DType::kByte, "Wrong dtype of scale_inv_colwise");

  int blocks_x = (cols + kColsPerTile - 1) / kColsPerTile;
  int blocks_y = (rows + kRowsPerTile - 1) / kRowsPerTile;
  dim3 grid(blocks_x, blocks_y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      mxfp8_scaling_partial_cast_kernel<IType, fp8e4m3><<<grid, kColsPerTile, 0, stream>>>(
          reinterpret_cast<const IType *>(input.data.dptr),
          reinterpret_cast<fp8e4m3 *>(output_rowwise.data.dptr),
          reinterpret_cast<fp8e4m3 *>(output_colwise.data.dptr),
          reinterpret_cast<const e8m0_t *>(scale_inv_rowwise.data.dptr),
          reinterpret_cast<const e8m0_t *>(scale_inv_colwise.data.dptr),
          scale_inv_rowwise.data.shape[1], scale_inv_colwise.data.shape[1], rows, cols,
          start_offset, input.data.shape[0]);)
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
  mxfp8_scaling_recipe::mxfp8_scaling_partial_cast(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(output_rowwise),
      *convertNVTETensorCheck(output_colwise), *convertNVTETensorCheck(scale_inv_rowwise),
      *convertNVTETensorCheck(scale_inv_colwise), rows, cols, start_offset, stream);
}
