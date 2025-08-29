/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <algorithm>
#include <cmath>

#include "../common.h"
#include "transformer_engine/dropout.h"

namespace transformer_engine {
namespace {

struct ull2 {
  uint64_t x;
  uint64_t y;
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
__global__ void dropout_kernel_fwd_f16(const T *input, T *output, uint16_t *mask,
                                       int64_t *rng_state, int n_rng_blocks,
                                       unsigned p_dropout_in_uint32_t, float inv_prob) {
  int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int num_threads = gridDim.y * gridDim.x * blockDim.x;
  uint16_t result = 0;
  size_t ofs = tid;
  curandStatePhilox4_32_10_t state;
  curand_init(rng_state[0], ofs, rng_state[1], &state);
  ull2 input8;
  const ull2 *gmem_input_ptr = reinterpret_cast<const ull2 *>(input);
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
        input_h_8[i] =
            (result_copy & 0x1 == 0x1)
                ? static_cast<T>(0.f)
                : static_cast<T>(static_cast<float>(input_h_8[i]) * inv_prob);
        result_copy = result_copy >> 1;
      }
      gmem_output_ptr[tid * 2 + j].x = input8.x;
      gmem_output_ptr[tid * 2 + j].y = input8.y;
    }

    mask[tid] = result;
    tid += num_threads;
    ofs = tid;
    result = 0;
  }
}

template <typename T, typename O>
__global__ void dropout_kernel_fwd_fp8(const T *input, float *scale_inv, O *output, uint16_t *mask,
                                       int64_t *rng_state, int n_rng_blocks,
                                       unsigned p_dropout_in_uint32_t, float inv_prob,
                                       bool is_training) {
  int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int num_threads = gridDim.y * gridDim.x * blockDim.x;
  uint16_t result = 0;
  size_t ofs = tid;
  curandStatePhilox4_32_10_t state;
  curand_init(rng_state[0], ofs, rng_state[1], &state);
  uint64_t input8;
  const uint64_t *gmem_input_ptr = reinterpret_cast<const uint64_t *>(input);
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
        if (is_training) {
          output_h_8[i] =
              (result_copy & 0x1 == 0x1)
                  ? static_cast<O>(0.f)
                  : static_cast<O>(static_cast<float>(input_h_8[i]) * scale_inv_val * inv_prob);
          result_copy = result_copy >> 1;
        } else {
          output_h_8[i] = static_cast<O>(static_cast<float>(input_h_8[i]) * scale_inv_val);
        }
      }
      gmem_output_ptr[tid * 2 + j].x = output8.x;
      gmem_output_ptr[tid * 2 + j].y = output8.y;
    }

    mask[tid] = result;
    tid += num_threads;
    ofs = tid;
    result = 0;
  }
}

template <typename T>
__global__ void dropout_kernel_bwd_f16(const T *grad_out, const uint16_t *in_mask, T *grad_in,
                                       const size_t n_rng_blocks, const float inv_prob) {
  int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int num_threads = gridDim.y * gridDim.x * blockDim.x;
  uint16_t result = 0;
  ull2 input8;
  const ull2 *gmem_input_ptr = reinterpret_cast<const ull2 *>(grad_out);
  ull2 *gmem_output_ptr = reinterpret_cast<ull2 *>(grad_in);

  while (tid < n_rng_blocks) {
    result = in_mask[tid];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      input8 = gmem_input_ptr[tid * 2 + j];
      T *input_h_8 = reinterpret_cast<T *>(&input8);

#pragma unroll
      for (int i = 0; i < 8; i++) {
        input_h_8[i] =
            (result & 0x1 == 0x1)
                ? static_cast<T>(0.f)
                : static_cast<T>(static_cast<float>(input_h_8[i]) * inv_prob);
        result = result >> 1;
      }
      gmem_output_ptr[tid * 2 + j] = input8;
    }

    tid += num_threads;
  }
}

}  // namespace

void dropout_fwd(const Tensor &input, Tensor &output, Tensor &mask, Tensor &rng_state,
                 float dropout_probability, cudaStream_t stream) {
  // Check tensors
  const size_t numel = input.numel();
  NVTE_CHECK(input.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be FP16/BF16 tensor or tensor-scaled FP8 tensor, ",
             "but scaling mode is ", to_string(input.scaling_mode), ".");
  NVTE_CHECK(output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Output tensor must be FP16/BF16 tensor, ", "but scaling mode is ",
             to_string(output.scaling_mode), ".");
  NVTE_CHECK(mask.scaling_mode == NVTE_DELAYED_TENSOR_SCALING, "Mask tensor must be INT16 tensor, ",
             "but scaling mode is ", to_string(mask.scaling_mode), ".");
  NVTE_CHECK(rng_state.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "RNG state tensor must be INT64 tensor with two entries, ", "but scaling mode is ",
             to_string(rng_state.scaling_mode), ".");
  NVTE_CHECK(output.dtype() == DType::kFloat16 || output.dtype() == DType::kBFloat16,
             "Output tensor must be FP16/BF16 tensor, but dtype is ", to_string(output.dtype()),
             ".");
  NVTE_CHECK(mask.dtype() == DType::kInt16, "Mask tensor must be INT16 tensor, but dtype is ",
             to_string(mask.dtype()), ".");
  NVTE_CHECK(rng_state.dtype() == DType::kInt64,
             "RNG state tensor must be INT64 tensor with two entries, ", "but dtype is ",
             to_string(rng_state.dtype()), ".");
  NVTE_CHECK(numel % 16 == 0,
             "Input tensor number of elements must be divisible by 16, but shape is ",
             input.shape(), ".");
  NVTE_CHECK(numel == output.numel(), "Input tensor (shape=", input.shape(),
             ") and output tensor (shape=", output.shape(), ") do not match.");
  NVTE_CHECK(numel / 16 == mask.numel(), "Input tensor (shape=", input.shape(),
             ") and mask tensor (shape=", mask.shape(), ") do not match.");
  NVTE_CHECK(rng_state.numel() == 2, "RNG state tensor must be INT64 tensor with two entries, ",
             "but shape is ", rng_state.shape(), ".");
  NVTE_CHECK(input.data.dptr != nullptr, "Input tensor is missing data.");
  NVTE_CHECK(output.data.dptr != nullptr, "Output tensor is missing data.");
  NVTE_CHECK(mask.data.dptr != nullptr, "Mask tensor is missing data.");
  NVTE_CHECK(rng_state.data.dptr != nullptr, "RNG state tensor is missing data.");

  // Convert dropout probablity to scale and 8-bit representation
  NVTE_CHECK(dropout_probability >= 0 && dropout_probability < 1, "Invalid dropout probability (",
             dropout_probability, ").");
  const float scale = 1 / (1 - dropout_probability);
  const uint8_t prob_uint8 = static_cast<uint8_t>(std::floor(dropout_probability * 256));
  NVTE_CHECK(prob_uint8 == dropout_probability * 256, "Dropout probability (", dropout_probability,
             ") is not representable in 8 bits");
  uint32_t prob4 = prob_uint8;
  prob4 = (prob4 << 8) | prob4;
  prob4 = (prob4 << 16) | prob4;

  // CUDA config
  constexpr size_t rng_block_size = 16;
  const size_t num_rng_blocks = numel / rng_block_size;
  const size_t block_size = std::min(num_rng_blocks, static_cast<size_t>(128));
  const size_t num_blocks = DIVUP(num_rng_blocks, block_size);
  NVTE_CHECK(numel >= num_blocks * rng_block_size,
             "Input tensor has invalid shape (shape=", input.shape(), ", num_blocks=", num_blocks,
             ", rng_block_size=", rng_block_size, ").");

  // Launch kernel depending on input dtype
  if (input.dtype() == DType::kFloat16 || input.dtype() == DType::kBFloat16) {
    NVTE_CHECK(input.dtype() == output.dtype(), "Input tensor (dtype=", to_string(input.dtype()),
               ") and output tensor (dtype=", to_string(output.dtype()), ") do not match.");
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        input.dtype(), DType,
        dropout_kernel_fwd_f16<DType><<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const DType *>(input.data.dptr),
            reinterpret_cast<DType *>(output.data.dptr),
            reinterpret_cast<uint16_t *>(mask.data.dptr),
            reinterpret_cast<int64_t *>(rng_state.data.dptr), num_rng_blocks, prob4, scale););
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else if (input.dtype() == DType::kFloat8E4M3 || input.dtype() == DType::kFloat8E5M2) {
    NVTE_CHECK(input.scale_inv.dptr != nullptr, "Input tensor scale-inverse is not allocated.");
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        input.dtype(), IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
            output.dtype(), OType,
            dropout_kernel_fwd_fp8<IType, OType><<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<float *>(input.scale_inv.dptr),
                reinterpret_cast<OType *>(output.data.dptr),
                reinterpret_cast<uint16_t *>(mask.data.dptr),
                reinterpret_cast<int64_t *>(rng_state.data.dptr), num_rng_blocks, prob4, scale,
                true);

        ););
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    NVTE_ERROR("Input tensor must be FP16/BF16 tensor or tensor-scaled FP8 tensor, ",
               "but dtype is ", to_string(input.dtype()), ".");
  }
}

void dropout_bwd(const Tensor &grad_output, const Tensor &mask, Tensor &grad_input,
                 float dropout_probability, cudaStream_t stream) {
  // Check tensors
  const size_t numel = grad_output.numel();
  NVTE_CHECK(grad_output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Grad output tensor must be FP16/BF16 tensor, ", "but scaling mode is ",
             to_string(grad_output.scaling_mode), ".");
  NVTE_CHECK(grad_input.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Grad input tensor must be FP16/BF16 tensor, ", "but scaling mode is ",
             to_string(grad_input.scaling_mode), ".");
  NVTE_CHECK(mask.scaling_mode == NVTE_DELAYED_TENSOR_SCALING, "Mask tensor must be INT16 tensor, ",
             "but scaling mode is ", to_string(mask.scaling_mode), ".");
  NVTE_CHECK(grad_output.dtype() == DType::kFloat16 || grad_output.dtype() == DType::kBFloat16,
             "Grad output tensor must be FP16/BF16 tensor, but dtype is ",
             to_string(grad_output.dtype()), ".");
  NVTE_CHECK(grad_output.dtype() == grad_input.dtype(),
             "Grad output tensor (dtype=", to_string(grad_output.dtype()),
             ") and grad input tensor (dtype=", to_string(grad_input.dtype()), ") do not match.");
  NVTE_CHECK(mask.dtype() == DType::kInt16, "Mask tensor must be INT16 tensor, but dtype is ",
             to_string(mask.dtype()), ".");
  NVTE_CHECK(numel % 16 == 0,
             "Grad output tensor number of elements must be divisible by 16, but shape is ",
             grad_output.shape(), ".");
  NVTE_CHECK(numel == grad_input.numel(), "Grad output tensor (shape=", grad_output.shape(),
             ") and grad input tensor (shape=", grad_input.shape(), ") do not match.");
  NVTE_CHECK(numel / 16 == mask.numel(), "Grad output tensor (shape=", grad_output.shape(),
             ") and mask tensor (shape=", mask.shape(), ") do not match.");
  NVTE_CHECK(grad_output.data.dptr != nullptr, "Grad output tensor is missing data.");
  NVTE_CHECK(grad_input.data.dptr != nullptr, "Grad input tensor is missing data.");
  NVTE_CHECK(mask.data.dptr != nullptr, "Mask tensor is missing data.");

  // Convert dropout probablity to scale
  NVTE_CHECK(dropout_probability >= 0 && dropout_probability < 1, "Invalid dropout probability (",
             dropout_probability, ").");
  const float scale = 1 / (1 - dropout_probability);

  // CUDA config
  constexpr size_t rng_block_size = 16;
  const size_t num_rng_blocks = numel / rng_block_size;
  const size_t block_size = std::min(num_rng_blocks, static_cast<size_t>(128));
  const size_t num_blocks = DIVUP(num_rng_blocks, block_size);

  // Launch kernel
  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      grad_output.dtype(), DType,
      dropout_kernel_bwd_f16<DType><<<num_blocks, block_size, 0, stream>>>(
          reinterpret_cast<const DType *>(grad_output.data.dptr),
          reinterpret_cast<const uint16_t *>(mask.data.dptr),
          reinterpret_cast<DType *>(grad_input.data.dptr), num_rng_blocks, scale););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine

void nvte_dropout_fwd(const NVTETensor input, NVTETensor output, NVTETensor mask,
                      NVTETensor rng_state, float dropout_probability, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dropout_fwd);
  using namespace transformer_engine;
  dropout_fwd(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
              *convertNVTETensorCheck(mask), *convertNVTETensorCheck(rng_state),
              dropout_probability, stream);
}

void nvte_dropout_bwd(const NVTETensor grad_output, const NVTETensor mask, NVTETensor grad_input,
                      float dropout_probability, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dropout_bwd);
  using namespace transformer_engine;
  dropout_bwd(*convertNVTETensorCheck(grad_output), *convertNVTETensorCheck(mask),
              *convertNVTETensorCheck(grad_input), dropout_probability, stream);
}
