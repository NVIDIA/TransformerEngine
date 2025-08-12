/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include "../common.h"
#include "transformer_engine/dropout.h"

namespace transformer_engine {

using ull = unsigned long long;
struct ull2 {
  unsigned long long x;
  unsigned long long y;
};

inline __device__ uint16_t packed_le_8bit(unsigned A, unsigned B) {
  unsigned T = A & 0x7F7F7F7F;
  unsigned D = B | 0x80808080;
  D = -T + D;
  T = (A ^ B) | 0x7F7F7F7F;
  D = ~(T ^ D);
  asm("lop3.b32 %0, %1, %2, %3, 0x4d;\n\t" : "=r"(D) : "r"(A), "r"(B), "r"(D));
  D = D & 0x80808080;
  T = (D & 0x0000FFFF) | (D >> 17);
  return T;
}
template <typename T>
__global__ void dropout_kernel_fwd(T *input, T *output, uint16_t *mask, int64_t *rng_state,
                                   int n_rng_blocks, unsigned p_dropout_in_uint32_t,
                                   float inv_prob) {
  int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int num_threads = gridDim.y * gridDim.x * blockDim.x;
  uint16_t result = 0;
  size_t ofs = tid;
  curandStatePhilox4_32_10_t state;
  curand_init(rng_state[0], ofs, rng_state[1], &state);
  ull2 input8;
  ull2 *gmem_input_ptr = reinterpret_cast<ull2 *>(input);
  ull2 *gmem_output_ptr = reinterpret_cast<ull2 *>(output);

  while (tid < n_rng_blocks) {
    uint4 random_uint4 = curand4(&state);
    result = packed_le_8bit(random_uint4.x, p_dropout_in_uint32_t);
    result |= packed_le_8bit(random_uint4.y, p_dropout_in_uint32_t) >> 2;
    result |= packed_le_8bit(random_uint4.z, p_dropout_in_uint32_t) >> 4;
    result |= packed_le_8bit(random_uint4.w, p_dropout_in_uint32_t) >> 6;

    uint16_t result_copy = result;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      input8 = gmem_input_ptr[tid * 2 + j];
      T *input_h_8 = reinterpret_cast<T *>(&input8);

#pragma unroll
      for (int i = 0; i < 8; i++) {
        input_h_8[i] = (result_copy & 0x1 == 0x1) ? (T)0.f : input_h_8[i] * (T)inv_prob;
        result_copy = result_copy >> 1;
      }
      gmem_output_ptr[tid * 2 + j].x = input8.x;
      gmem_output_ptr[tid * 2 + j].y = input8.y;
    }

    mask[tid] = result;  //0xFFFF
    tid += num_threads;
    ofs = tid;
    result = 0;
  }
}
template <typename T, typename O>
__global__ void dropout_kernel_fwd_fp8(T *input, float *scale_inv, O *output, uint16_t *mask,
                                       int64_t *rng_state, int n_rng_blocks,
                                       unsigned p_dropout_in_uint32_t, float inv_prob) {
  int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int num_threads = gridDim.y * gridDim.x * blockDim.x;
  uint16_t result = 0;
  size_t ofs = tid;
  curandStatePhilox4_32_10_t state;
  curand_init(rng_state[0], ofs, rng_state[1], &state);
  ull input8;
  ull *gmem_input_ptr = reinterpret_cast<ull *>(input);
  ull2 output8;
  ull2 *gmem_output_ptr = reinterpret_cast<ull2 *>(output);
  float scale_inv_val = *scale_inv;

  while (tid < n_rng_blocks) {
    uint4 random_uint4 = curand4(&state);
    result = packed_le_8bit(random_uint4.x, p_dropout_in_uint32_t);
    result |= packed_le_8bit(random_uint4.y, p_dropout_in_uint32_t) >> 2;
    result |= packed_le_8bit(random_uint4.z, p_dropout_in_uint32_t) >> 4;
    result |= packed_le_8bit(random_uint4.w, p_dropout_in_uint32_t) >> 6;

    uint16_t result_copy = result;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      input8 = gmem_input_ptr[tid * 2 + j];
      T *input_h_8 = reinterpret_cast<T *>(&input8);
      O *output_h_8 = reinterpret_cast<O *>(&output8);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        output_h_8[i] =
            (result_copy & 0x1 == 0x1)
                ? (O)0.f
                : static_cast<O>(static_cast<float>(input_h_8[i]) * scale_inv_val * inv_prob);
        result_copy = result_copy >> 1;
      }
      gmem_output_ptr[tid * 2 + j].x = output8.x;
      gmem_output_ptr[tid * 2 + j].y = output8.y;
    }

    mask[tid] = result;  //0xFFFF
    tid += num_threads;
    ofs = tid;
    result = 0;
  }
}
template <typename T>
__global__ void dropout_kernel_bwd(T *grad_out, uint16_t *in_mask, T *grad_in,
                                   const size_t n_rng_blocks, const float inv_prob) {
  int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int num_threads = gridDim.y * gridDim.x * blockDim.x;
  uint16_t result = 0;
  ull2 input8;
  ull2 *gmem_input_ptr = reinterpret_cast<ull2 *>(grad_out);
  ull2 *gmem_output_ptr = reinterpret_cast<ull2 *>(grad_in);

  while (tid < n_rng_blocks) {
    result = in_mask[tid];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      input8 = gmem_input_ptr[tid * 2 + j];
      T *input_h_8 = reinterpret_cast<T *>(&input8);

#pragma unroll
      for (int i = 0; i < 8; i++) {
        input_h_8[i] = (result & 0x1 == 0x1) ? (T)0.f : input_h_8[i] * (T)inv_prob;
        result = result >> 1;
      }
      gmem_output_ptr[tid * 2 + j] = input8;
    }

    tid += num_threads;
  }
}

void prepare_dropout_fwd(Tensor input, Tensor output, Tensor mask, Tensor rng_state,
                         float dropout_probability, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(input.dim() == 2, "Expected 2-dim tensor.");
  NVTE_CHECK(input.dtype() == DType::kFloat16 || input.dtype() == DType::kBFloat16);

  int blk_size = 128;
  int rng_block_size = 16;
  int n_rng_blks = input.numel() / 16;
  int grid = (n_rng_blks + blk_size - 1) / blk_size;
  dim3 dim_grid(grid);
  dim3 dim_block(blk_size);

  NVTE_CHECK(input.numel() % 16 == 0, "numel must be divisible by 16");
  NVTE_CHECK(input.numel() >= blk_size * rng_block_size,
             "numel must be greater than or equal to blk_size*rng_block_size");
  NVTE_CHECK(dropout_probability != 1.0f, "dropout_probability must not be 1.0f");

  uint8_t p_dropout_in_uint8_t = static_cast<uint8_t>(std::floor(dropout_probability * 255.0));
  unsigned p_dropout_in_uint32_t_8bit = 0;
  unsigned p_drop8 = p_dropout_in_uint8_t;
  for (int i = 0; i < 4; i++) {
    p_dropout_in_uint32_t_8bit = p_dropout_in_uint32_t_8bit | p_drop8;
    p_drop8 = p_drop8 << 8;
  }

  float inv_prob = 1 / (1 - dropout_probability);
  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      input.dtype(), dtype,
      dropout_kernel_fwd<dtype><<<dim_grid, dim_block, 0, stream>>>(
          reinterpret_cast<dtype *>(input.data.dptr), reinterpret_cast<dtype *>(output.data.dptr),
          reinterpret_cast<uint16_t *>(mask.data.dptr),
          reinterpret_cast<int64_t *>(rng_state.data.dptr), n_rng_blks, p_dropout_in_uint32_t_8bit,
          inv_prob););
}

void prepare_dropout_fwd_fp8(Tensor input, Tensor output, Tensor mask, Tensor rng_state,
                             float dropout_probability, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(input.dim() == 2, "Expected 2-dim tensor.");
  NVTE_CHECK(output.dtype() == DType::kFloat16 || output.dtype() == DType::kBFloat16);
  NVTE_CHECK(is_tensor_scaling(input.scaling_mode),
             "Only per-tensor scaling is supported for fp8 dropout");
  CheckInputTensor(input, "dropout_fwd_fp8_input");

  int blk_size = 128;
  int rng_block_size = 16;
  int n_rng_blks = input.numel() / 16;
  int grid = (n_rng_blks + blk_size - 1) / blk_size;
  dim3 dim_grid(grid);
  dim3 dim_block(blk_size);

  NVTE_CHECK(input.numel() % 16 == 0, "numel must be divisible by 16");
  NVTE_CHECK(input.numel() >= blk_size * rng_block_size,
             "numel must be greater than or equal to blk_size*rng_block_size");
  NVTE_CHECK(dropout_probability != 1.0f, "dropout_probability must not be 1.0f");

  uint8_t p_dropout_in_uint8_t = static_cast<uint8_t>(std::floor(dropout_probability * 255.0));
  unsigned p_dropout_in_uint32_t_8bit = 0;
  unsigned p_drop8 = p_dropout_in_uint8_t;
  for (int i = 0; i < 4; i++) {
    p_dropout_in_uint32_t_8bit = p_dropout_in_uint32_t_8bit | p_drop8;
    p_drop8 = p_drop8 << 8;
  }
  float inv_prob = 1 / (1 - dropout_probability);

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input.dtype(), itype,
      TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
          output.dtype(), otype,
          dropout_kernel_fwd_fp8<itype, otype>
          <<<dim_grid, dim_block, 0, stream>>>(reinterpret_cast<itype *>(input.data.dptr),
                                               reinterpret_cast<float *>(input.scale_inv.dptr),
                                               reinterpret_cast<otype *>(output.data.dptr),
                                               reinterpret_cast<uint16_t *>(mask.data.dptr),
                                               reinterpret_cast<int64_t *>(rng_state.data.dptr),
                                               n_rng_blks, p_dropout_in_uint32_t_8bit, inv_prob);

      ););
}

void prepare_dropout_bwd(Tensor grad_output, Tensor mask, Tensor grad_input,
                         float dropout_probability, cudaStream_t stream) {
  using namespace transformer_engine;

  int blk_size = 128;
  int n_rng_blks = grad_output.numel() / 16;
  int grid = (n_rng_blks + blk_size - 1) / blk_size;
  dim3 dim_grid(grid);
  dim3 dim_block(blk_size);

  assert(dropout_probability != 1.0f);
  float inv_prob = 1 / (1 - dropout_probability);

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      grad_output.dtype(), dtype,
      dropout_kernel_bwd<dtype><<<dim_grid, dim_block, 0, stream>>>(
          reinterpret_cast<dtype *>(grad_output.data.dptr),
          reinterpret_cast<uint16_t *>(mask.data.dptr),
          reinterpret_cast<dtype *>(grad_input.data.dptr), n_rng_blks, inv_prob););
}

}  // namespace transformer_engine

void nvte_dropout_fwd(NVTETensor input, NVTETensor output, NVTETensor mask, NVTETensor rng_state,
                      float dropout_probability, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dropout_fwd);
  using namespace transformer_engine;

  prepare_dropout_fwd(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                      *convertNVTETensorCheck(mask), *convertNVTETensorCheck(rng_state),
                      dropout_probability, stream);
}

void nvte_dropout_fwd_fp8(NVTETensor input, NVTETensor output, NVTETensor mask,
                          NVTETensor rng_state, float dropout_probability, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dropout_fwd_fp8);
  using namespace transformer_engine;

  prepare_dropout_fwd_fp8(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                          *convertNVTETensorCheck(mask), *convertNVTETensorCheck(rng_state),
                          dropout_probability, stream);
}

void nvte_dropout_bwd(NVTETensor grad_output, NVTETensor mask, NVTETensor grad_input,
                      float dropout_probability, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dropout_bwd);
  using namespace transformer_engine;

  prepare_dropout_bwd(*convertNVTETensorCheck(grad_output), *convertNVTETensorCheck(mask),
                      *convertNVTETensorCheck(grad_input), dropout_probability, stream);
}
