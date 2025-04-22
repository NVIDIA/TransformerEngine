/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <transformer_engine/transpose.h>

#include "common/common.h"
#include "common/utils.cuh"

constexpr int kTileDim = 128;  // Fixed to 128 beacause we are using 1x128 and 128x1 quantization
constexpr int kThreadsPerBlock = 256;  // Thread block size, 8 warps in total

namespace transformer_engine {
namespace detail {

#define warp_size 32
#define FP8_MAX 448.0f
#define EPS 1e-6

__device__ __host__ inline float cast_to_e8(float x) {
  uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&x);
  scale_bits &= 0xFF800000;
  return *reinterpret_cast<float *>(&scale_bits);
}

template <typename FP8_TYPE>
__device__ inline void swap_fp8(FP8_TYPE &a, FP8_TYPE &b) {
  // 方法1：使用临时变量
  FP8_TYPE temp = a;
  a = b;
  b = temp;
}

template <typename T>
__device__ inline float warp_max_reduce_on_float(T &val) {
  float val_float = float(val);
  // 使用 butterfly pattern
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 16));
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 8));
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 4));
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 2));
  val_float = max(val_float, __shfl_xor_sync(0xffffffff, val_float, 1));
  return val_float;
}

// scaling dim = 128
// scaling type = int
// tile shape for each blokc: 128x128
// Warning: what about the seq_len % 4 != 0? (float4 load scale)
// This kernel required aligned :
// 1. hidden_dim % 4 == 0  # use float to load fp8
// 2. seq_len % 4 == 0     # use float to load fp8
template <typename FP8_TYPE>
__global__ void dequantize_cast_transpose_1x128_aligned_kernel(
    FP8_TYPE *input, float *input_scale_inv, FP8_TYPE *output, float *output_scale_inv,
    const int hidden_dim, const int seq_len) {
  // define the meta
  auto warp_id = threadIdx.x / 32;
  auto lane_id = threadIdx.x % 32;
  float *input_fp32 = (float *)input;
  float *output_fp32 = (float *)output;
  auto block_offset_x = blockIdx.x * kTileDim;
  auto block_offset_y = blockIdx.y * kTileDim;
  // Pad 1 element for each {4 rows} to avoid bank conflict
  // Each {4 rows} is put on the shem row
  __shared__ float smem_float_tile[32][kTileDim + 1];
  __shared__ float smem_scale[kTileDim];
  // Use float4 buffer in shared memory to avoid float4 load from global memory which required 4-float alignment.
  float4 *smem_scale_fp128 = reinterpret_cast<float4 *>(smem_scale);

  // Load the scale to shared memory
  for (auto i = threadIdx.x; i < kTileDim; i += blockDim.x) {
    smem_scale[i] = (block_offset_x + i < seq_len)
                        ? input_scale_inv[blockIdx.y * seq_len + block_offset_x + i]
                        : EPS;
  }

  auto offset_col = (block_offset_y / 4 + lane_id);
  auto offset_row = (block_offset_x + 4 * warp_id);
  for (auto w = warp_id; w < kTileDim / 4 && offset_row < seq_len;
       w += blockDim.x / 32, offset_row += 4) {
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
      smem_float_tile[lane_id][i * 32 + w] = buffer_fp32[i];
    }
  }
  __syncthreads();

  // 4. Dequantize on the tile on smem column, each warp will hold a line of 128 elements
  float4 scale_buffer_fp128 = smem_scale_fp128[lane_id];
  float *scale_buffer_fp32 = (float *)&scale_buffer_fp128;
  float buffer_fp32;
  __half dequantized[4];
  for (auto w = warp_id; w < kTileDim; w += blockDim.x / 32) {
    // 4. Dequantize on the tile on smem column, each warp will hold a line of 128 elements
    auto offset_row = (block_offset_y + w);
    auto offset_col = (block_offset_x / 4 + lane_id);
    bool is_valid = (offset_col < seq_len / 4 && offset_row < hidden_dim);

    buffer_fp32 = is_valid ? smem_float_tile[w / 4][(w % 4) * 32 + lane_id] : 0;
    FP8_TYPE *buffer_fp8 = reinterpret_cast<FP8_TYPE *>(&buffer_fp32);
// Dequantize the local fp8 matrix, each column is shared the same scaling factor
#pragma unroll
    for (int i = 0; i < 4; i++) {
      dequantized[i] = float(buffer_fp8[i]) / scale_buffer_fp32[i];
    }

    // store the transposed fp8 tile on smem to the global memory
    if (is_valid) {
      output_fp32[(offset_row) * (seq_len / 4) + offset_col] = buffer_fp32;
    }

    // 5. Reduce on line, find the new scaling factor on a row
    float max_val = warp_max_reduce_on_float(dequantized[0]);
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[1]));
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[2]));
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[3]));

    __syncwarp();
    if (lane_id == 0) {
      smem_scale[w] = cast_to_e8(FP8_MAX / (max_val + EPS));
    }
  }
  __syncthreads();

  // 6. store the new scaling factor to the global memory.
  for (auto i = threadIdx.x; i < kTileDim; i += blockDim.x) {
    if ((block_offset_y + i < hidden_dim))
      output_scale_inv[blockIdx.x * hidden_dim + block_offset_y + i] = smem_scale[i];
  }
}

template <typename FP8_TYPE>
__global__ void dequantize_cast_transpose_1x128_kernel(FP8_TYPE *input, float *input_scale_inv,
                                                       FP8_TYPE *output, float *output_scale_inv,
                                                       const int hidden_dim, const int seq_len) {
  auto warp_id = threadIdx.x / 32;
  auto lane_id = threadIdx.x % 32;
  auto block_offset_x = blockIdx.x * kTileDim;
  auto block_offset_y = blockIdx.y * kTileDim;
  __shared__ float smem_scale[kTileDim];
  float4 *smem_scale_fp128 = reinterpret_cast<float4 *>(smem_scale);
  __shared__ FP8_TYPE transposed[kTileDim][kTileDim + 4];

  // 1. load the scale from global memory
  for (auto i = threadIdx.x; i < kTileDim; i += blockDim.x) {
    smem_scale[i] = (block_offset_x + i < seq_len)
                        ? input_scale_inv[blockIdx.y * seq_len + block_offset_x + i]
                        : EPS;
  }

  // 2. load the fp8 activation from global memory, then transpose on the smem
  // Each warp will load 1x128 row
  for (auto x = warp_id; (x < kTileDim); x += blockDim.x / 32) {
    for (auto y = lane_id; (block_offset_y + y < hidden_dim) && (y < kTileDim); y += warp_size) {
      if ((block_offset_x + x < seq_len) && (block_offset_y + y < hidden_dim))
        transposed[y][x] = input[(block_offset_x + x) * hidden_dim + (block_offset_y + y)];
      else
        transposed[y][x] = FP8_TYPE(0.0f);
    }
  }
  __syncthreads();

  // 3. Dequantize the fp8 activation on the smem, then compute the new scaling factor on a row
  float4 sacle_buffer_fp128 = smem_scale_fp128[lane_id];
  float *scale_buffer_fp32 = (float *)&sacle_buffer_fp128;
  __half dequantized[4];
  for (auto x = warp_id; x < kTileDim; x += blockDim.x / 32) {
    float data_fp32 = reinterpret_cast<float *>(transposed[x])[lane_id];
    FP8_TYPE *data_fp8 = reinterpret_cast<FP8_TYPE *>(&data_fp32);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      dequantized[i] = float(data_fp8[i]) / scale_buffer_fp32[i];
    }

    float max_val = warp_max_reduce_on_float(dequantized[0]);
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[1]));
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[2]));
    max_val = max(max_val, warp_max_reduce_on_float(dequantized[3]));

    __syncwarp();
    if (lane_id == 0) {
      smem_scale[x] = cast_to_e8(FP8_MAX / (max_val + EPS));
    }
  }
  __syncthreads();

  // 4. store the new scaling factor to the global memory
  for (auto i = threadIdx.x; i < kTileDim; i += blockDim.x) {
    if ((block_offset_y + i < hidden_dim))
      output_scale_inv[blockIdx.x * hidden_dim + block_offset_y + i] = smem_scale[i];
  }

  // 5. store the transposed fp16 activation to the global memory
  for (auto x = warp_id; (block_offset_y + x < hidden_dim) && (x < kTileDim);
       x += blockDim.x / 32) {
    auto dest_row = block_offset_y + x;
    for (auto y = lane_id; (block_offset_x + y < seq_len) && (y < kTileDim); y += warp_size) {
      auto dest_col = block_offset_x + y;
      if (dest_row < hidden_dim && dest_col < seq_len) {
        output[dest_row * seq_len + dest_col] = transposed[x][y];
      }
    }
  }
}

}  // namespace detail
}  // namespace transformer_engine

void nvte_transpose_blockwise(NVTETensor input, cudaStream_t stream) {
  NVTE_API_CALL(nvte_transpose_blockwise);
  using namespace transformer_engine;
  auto input_tensor = *reinterpret_cast<const Tensor *>(input);

  // Get the data and the scale
  auto rowwise_data = input_tensor.data;
  auto rowwise_scale_inv = input_tensor.scale_inv;
  auto colwise_data = input_tensor.columnwise_data;
  auto colwise_scale_inv = input_tensor.columnwise_scale_inv;

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

  const size_t hidden_dim = rowwise_shape[1];
  const size_t seq_len = rowwise_shape[0];

  // Early return if the input tensor is empty
  if (hidden_dim * seq_len == 0) {
    return;
  }

  dim3 grid((seq_len + kTileDim - 1) / kTileDim, (hidden_dim + kTileDim - 1) / kTileDim);
  dim3 block(kThreadsPerBlock);

  if (hidden_dim % 4 == 0 && seq_len % 4 == 0) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        itype, FP8_TYPE,
        transformer_engine::detail::dequantize_cast_transpose_1x128_aligned_kernel<FP8_TYPE>
        <<<grid, block, 0, stream>>>(reinterpret_cast<FP8_TYPE *>(rowwise_data.dptr),
                                     reinterpret_cast<float *>(rowwise_scale_inv.dptr),
                                     reinterpret_cast<FP8_TYPE *>(colwise_data.dptr),
                                     reinterpret_cast<float *>(colwise_scale_inv.dptr), hidden_dim,
                                     seq_len););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        itype, FP8_TYPE,
        transformer_engine::detail::dequantize_cast_transpose_1x128_kernel<FP8_TYPE>
        <<<grid, block, 0, stream>>>(reinterpret_cast<FP8_TYPE *>(rowwise_data.dptr),
                                     reinterpret_cast<float *>(rowwise_scale_inv.dptr),
                                     reinterpret_cast<FP8_TYPE *>(colwise_data.dptr),
                                     reinterpret_cast<float *>(colwise_scale_inv.dptr), hidden_dim,
                                     seq_len););
  }
}
