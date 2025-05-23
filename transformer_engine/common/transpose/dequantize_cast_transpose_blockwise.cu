/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime.h>
#include <transformer_engine/transpose.h>

#include <cfloat>
#include <functional>
#include <numeric>

#include "common/common.h"
#include "common/recipe/recipe_common.cuh"
#include "common/utils.cuh"

namespace transformer_engine {
namespace {

constexpr size_t kThreadsPerWarp = 32;
constexpr int kTileDim = 128;  // Fixed to 128 because we are using 1x128 and 128x1 quantization
constexpr int kThreadsPerBlock = 256;  // Thread block size, 8 warps in total

template <typename FP8_TYPE>
__device__ inline void swap_fp8(FP8_TYPE &a, FP8_TYPE &b) {
  FP8_TYPE temp = a;
  a = b;
  b = temp;
}

template <typename T>
__device__ inline float warp_max_reduce_on_float(T &val) {
  float val_float = abs(float(val));
  // butterfly pattern
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 16));
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 8));
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 4));
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 2));
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 1));
  return val_float;
}

// scaling dim = 128
// scaling type = int
// tile shape for each block: 128x128
// Warning: what about the seq_len % 4 != 0? (float4 load scale)
// This kernel required aligned :
// 1. hidden_dim % 4 == 0  # use float to load fp8
// 2. seq_len % 4 == 0     # use float to load fp8
template <typename FP8_TYPE, typename RAW_TYPE>
__global__ void dequantize_cast_transpose_1x128_aligned_kernel(
    FP8_TYPE *input, float *input_scale_inv, FP8_TYPE *output, float *output_scale_inv,
    const size_t hidden_dim, const size_t seq_len, const float epsilon) {
  // define the meta
  auto warp_id = threadIdx.x / kThreadsPerWarp;
  auto lane_id = threadIdx.x % kThreadsPerWarp;
  auto warp_num = blockDim.x / kThreadsPerWarp;
  float *input_fp32 = (float *)input;
  float *output_fp32 = (float *)output;
  auto block_offset_x = blockIdx.x * kTileDim;
  auto block_offset_y = blockIdx.y * kTileDim;
  // Pad 1 element for each {4 rows} to avoid bank conflict
  // Each {4 rows} is put on the shem row
  __shared__ float smem_float_tile[kThreadsPerWarp][kTileDim + 1];
  __shared__ float smem_scale_inv[kTileDim];
  // Use float4 buffer in shared memory to avoid float4 load from global memory which required 4-float alignment.
  float4 *smem_scale_inv_fp128 = reinterpret_cast<float4 *>(smem_scale_inv);

  // Load the scale to shared memory
  for (auto i = threadIdx.x; i < kTileDim; i += blockDim.x) {
    smem_scale_inv[i] = (block_offset_x + i < seq_len)
                            ? input_scale_inv[blockIdx.y * seq_len + block_offset_x + i]
                            : epsilon;
  }

  auto offset_col = (block_offset_y / 4 + lane_id);
  auto offset_row = (block_offset_x + 4 * warp_id);
  for (auto w = warp_id; w < kTileDim / 4 && offset_row < seq_len;
       w += warp_num, offset_row += 4 * warp_num) {
    float buffer_fp32[4];
    FP8_TYPE *buffer_fp8 = (FP8_TYPE *)buffer_fp32;

// 1. Each warp will load 4 rows
#pragma unroll
    for (int i = 0; i < 4; i++) {
      buffer_fp32[i] = (offset_col < hidden_dim / 4 && offset_row + i < seq_len)
                           ? input_fp32[(offset_row + i) * (hidden_dim / 4) + offset_col]
                           : 0;
    }

    // 2. Each thread will hold a 4x4 matrix, then process local transpose on registers.
    swap_fp8<FP8_TYPE>(buffer_fp8[1], buffer_fp8[4]);
    swap_fp8<FP8_TYPE>(buffer_fp8[2], buffer_fp8[8]);
    swap_fp8<FP8_TYPE>(buffer_fp8[3], buffer_fp8[12]);
    swap_fp8<FP8_TYPE>(buffer_fp8[6], buffer_fp8[9]);
    swap_fp8<FP8_TYPE>(buffer_fp8[7], buffer_fp8[13]);
    swap_fp8<FP8_TYPE>(buffer_fp8[11], buffer_fp8[14]);

// 3. Store the result to the shared memory, make shared memory a transpose 128x128 tile.
#pragma unroll
    for (int i = 0; i < 4; i++) {
      smem_float_tile[lane_id][i * kThreadsPerWarp + w] = buffer_fp32[i];
    }
  }
  __syncthreads();

  // 4. Dequantize on the tile on smem column, each warp will hold a line of 128 elements
  float4 scale_buffer_fp128 = smem_scale_inv_fp128[lane_id];
  float *scale_buffer_fp32 = (float *)&scale_buffer_fp128;
  float buffer_fp32;
  RAW_TYPE dequantized[4];
  for (auto w = warp_id; w < kTileDim; w += warp_num) {
    // 4. Dequantize on the tile on smem column, each warp will hold a line of 128 elements
    auto offset_row = (block_offset_y + w);
    auto offset_col = (block_offset_x / 4 + lane_id);
    bool is_valid = (offset_col < seq_len / 4 && offset_row < hidden_dim);

    buffer_fp32 = is_valid ? smem_float_tile[w / 4][(w % 4) * kThreadsPerWarp + lane_id] : 0;
    FP8_TYPE *buffer_fp8 = reinterpret_cast<FP8_TYPE *>(&buffer_fp32);
// Dequantize the local fp8 matrix, each column is shared the same scaling factor
#pragma unroll
    for (int i = 0; i < 4; i++) {
      dequantized[i] = float(buffer_fp8[i]) * scale_buffer_fp32[i];
    }

    // 5. Reduce on line, find the new scaling factor on a row
    float max_val = warp_max_reduce_on_float(dequantized[0]);
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[1]));
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[2]));
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[3]));
    float new_scale = compute_scale_from_types<float, FP8_TYPE>(max_val, epsilon, true);
// 6. Re quantize the fp8 tile on smem
#pragma unroll
    for (int i = 0; i < 4; i++) {
      buffer_fp8[i] = FP8_TYPE(float(dequantized[i]) * new_scale);
    }

    // Store the transposed requantized fp8 tile on smem to the global memory
    if (is_valid) {
      output_fp32[(offset_row) * (seq_len / 4) + offset_col] = buffer_fp32;
    }
    // Store the new scaling factor to the shared memory
    __syncwarp();
    if (lane_id == 0) {
      smem_scale_inv[w] = 1.0f / new_scale;
    }
  }
  __syncthreads();

  // 7. store the new scaling factor to the global memory.
  for (auto i = threadIdx.x; i < kTileDim; i += blockDim.x) {
    if ((block_offset_y + i < hidden_dim))
      output_scale_inv[blockIdx.x * hidden_dim + block_offset_y + i] = smem_scale_inv[i];
  }
}

template <typename FP8_TYPE, typename RAW_TYPE>
__global__ void dequantize_cast_transpose_1x128_kernel(FP8_TYPE *input, float *input_scale_inv,
                                                       FP8_TYPE *output, float *output_scale_inv,
                                                       const size_t hidden_dim,
                                                       const size_t seq_len, const float epsilon) {
  auto warp_id = threadIdx.x / kThreadsPerWarp;
  auto lane_id = threadIdx.x % kThreadsPerWarp;
  auto warp_num = blockDim.x / kThreadsPerWarp;
  auto block_offset_x = blockIdx.x * kTileDim;
  auto block_offset_y = blockIdx.y * kTileDim;
  __shared__ float smem_scale_inv[kTileDim];
  float4 *smem_scale_inv_fp128 = reinterpret_cast<float4 *>(smem_scale_inv);
  __shared__ FP8_TYPE transposed[kTileDim][kTileDim + 4];

  // 1. load the scale from global memory
  for (auto i = threadIdx.x; i < kTileDim; i += blockDim.x) {
    smem_scale_inv[i] = (block_offset_x + i < seq_len)
                            ? input_scale_inv[blockIdx.y * seq_len + block_offset_x + i]
                            : epsilon;
  }

  // 2. load the fp8 activation from global memory, then transpose on the smem
  // Each warp will load 1x128 row
  for (auto x = warp_id; (x < kTileDim); x += warp_num) {
    for (auto y = lane_id; (block_offset_y + y < hidden_dim) && (y < kTileDim);
         y += kThreadsPerWarp) {
      if ((block_offset_x + x < seq_len) && (block_offset_y + y < hidden_dim))
        transposed[y][x] = input[(block_offset_x + x) * hidden_dim + (block_offset_y + y)];
      else
        transposed[y][x] = FP8_TYPE(0.0f);
    }
  }
  __syncthreads();

  // 3. Dequantize the fp8 activation on the smem, then compute the new scaling factor on a row
  float4 sacle_buffer_fp128 = smem_scale_inv_fp128[lane_id];
  float *scale_buffer_fp32 = (float *)&sacle_buffer_fp128;
  RAW_TYPE dequantized[4];
  for (auto w = warp_id; w < kTileDim; w += warp_num) {
    float data_fp32 = reinterpret_cast<float *>(transposed[w])[lane_id];
    FP8_TYPE *data_fp8 = reinterpret_cast<FP8_TYPE *>(&data_fp32);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      dequantized[i] = float(data_fp8[i]) * scale_buffer_fp32[i];
    }

    float max_val = warp_max_reduce_on_float(dequantized[0]);
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[1]));
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[2]));
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[3]));
    float new_scale = compute_scale_from_types<float, FP8_TYPE>(max_val, epsilon, true);

// 4. Re quantize the fp8 tile on smem
#pragma unroll
    for (int i = 0; i < 4; i++) {
      data_fp8[i] = FP8_TYPE(float(dequantized[i]) * new_scale);
    }

    // Store the transposed requantized activation to the global memory
    reinterpret_cast<float *>(transposed[w])[lane_id] = data_fp32;
    // Store the new scaling factor to the shared memory
    __syncwarp();
    if (lane_id == 0) {
      smem_scale_inv[w] = 1.0f / new_scale;
    }
  }
  __syncthreads();

  // 5. store the new scaling factor to the global memory
  for (auto i = threadIdx.x; i < kTileDim; i += blockDim.x) {
    if ((block_offset_y + i < hidden_dim))
      output_scale_inv[blockIdx.x * hidden_dim + block_offset_y + i] = smem_scale_inv[i];
  }

  // 6. store the transposed fp16 activation to the global memory
  for (auto x = warp_id; (block_offset_y + x < hidden_dim) && (x < kTileDim); x += warp_num) {
    auto dest_row = block_offset_y + x;
    for (auto y = lane_id; (block_offset_x + y < seq_len) && (y < kTileDim); y += kThreadsPerWarp) {
      auto dest_col = block_offset_x + y;
      if (dest_row < hidden_dim && dest_col < seq_len) {
        output[dest_row * seq_len + dest_col] = transposed[x][y];
      }
    }
  }
}

}  // namespace
}  // namespace transformer_engine

void nvte_transpose_blockwise(NVTETensor tensor, const NVTEQuantizationConfig quant_config,
                              transformer_engine::DType intermediate_dtype,
                              cudaStream_t stream = 0) {
  NVTE_API_CALL(nvte_transpose_blockwise);
  using namespace transformer_engine;
  auto te_tensor = *reinterpret_cast<const Tensor *>(tensor);

  // Get the data and the scale
  auto rowwise_data = te_tensor.data;
  auto rowwise_scale_inv = te_tensor.scale_inv;
  auto colwise_data = te_tensor.columnwise_data;
  auto colwise_scale_inv = te_tensor.columnwise_scale_inv;

  // Get the shape and the dtype
  auto rowwise_shape = rowwise_data.shape;
  auto rowwise_scale_inv_shape = rowwise_scale_inv.shape;
  auto colwise_shape = colwise_data.shape;
  auto colwise_scale_inv_shape = colwise_scale_inv.shape;
  auto itype = rowwise_data.dtype;

  NVTE_CHECK(rowwise_shape.size() == 2, "rowwise_shape must be 2D");
  NVTE_CHECK(rowwise_scale_inv_shape.size() == 2, "rowwise_scale_inv_shape must be 2D");
  NVTE_CHECK(colwise_shape.size() == 2, "colwise_shape must be 2D");
  NVTE_CHECK(colwise_scale_inv_shape.size() == 2, "colwise_scale_inv_shape must be 2D");
  NVTE_CHECK(rowwise_shape[0] == colwise_shape[1] && rowwise_shape[1] == colwise_shape[0],
             "The shape of rowwise_data and inversed colwise_data must be the same");
  NVTE_CHECK(rowwise_shape[0] == rowwise_scale_inv_shape[1],
             "num of rows of rowwise_data must be equal to num of cols of rowwise_scale_inv, got "
             "rowwise_shape[0]:",
             rowwise_shape[0], ", rowwise_scale_inv_shape[1]:", rowwise_scale_inv_shape[1]);
  NVTE_CHECK(colwise_shape[0] == colwise_scale_inv_shape[1],
             "num of cols of colwise_data must be equal to num of rows of colwise_scale_inv, got "
             "colwise_shape[0]:",
             colwise_shape[0], ", colwise_scale_inv_shape[1]:", colwise_scale_inv_shape[1]);

  const QuantizationConfig *quant_config_cpp =
      reinterpret_cast<const QuantizationConfig *>(quant_config);
  const bool force_pow_2_scales = quant_config_cpp ? quant_config_cpp->force_pow_2_scales : false;
  const float epsilon = quant_config_cpp ? quant_config_cpp->amax_epsilon : 0.0f;
  NVTE_CHECK(force_pow_2_scales,
             "Only power-of-2 scaling is supported for fp8 blockwise transpose");

  const size_t hidden_dim = rowwise_shape[1];
  const size_t seq_len = rowwise_shape[0];

  // Early return if the input tensor is empty
  if (hidden_dim * seq_len == 0) {
    return;
  }

  dim3 grid((seq_len + kTileDim - 1) / kTileDim, (hidden_dim + kTileDim - 1) / kTileDim);
  dim3 block(kThreadsPerBlock);

  if (hidden_dim % 4 == 0 && seq_len % 4 == 0) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        intermediate_dtype, RAW_TYPE,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            itype, FP8_TYPE,
            dequantize_cast_transpose_1x128_aligned_kernel<FP8_TYPE, RAW_TYPE>
            <<<grid, block, 0, stream>>>(reinterpret_cast<FP8_TYPE *>(rowwise_data.dptr),
                                         reinterpret_cast<float *>(rowwise_scale_inv.dptr),
                                         reinterpret_cast<FP8_TYPE *>(colwise_data.dptr),
                                         reinterpret_cast<float *>(colwise_scale_inv.dptr),
                                         hidden_dim, seq_len, epsilon);););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        intermediate_dtype, RAW_TYPE,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            itype, FP8_TYPE,
            dequantize_cast_transpose_1x128_kernel<FP8_TYPE, RAW_TYPE>
            <<<grid, block, 0, stream>>>(reinterpret_cast<FP8_TYPE *>(rowwise_data.dptr),
                                         reinterpret_cast<float *>(rowwise_scale_inv.dptr),
                                         reinterpret_cast<FP8_TYPE *>(colwise_data.dptr),
                                         reinterpret_cast<float *>(colwise_scale_inv.dptr),
                                         hidden_dim, seq_len, epsilon);););
  }
}
