/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <cmath>

#include "../common.h"
#include "../utils.cuh"
#include "transformer_engine/dropout.h"

namespace transformer_engine {
namespace {

// RNG kernels process chunks of 16 entries
constexpr size_t rng_chunk_size = 16;

// CUDA block size
constexpr size_t block_size = 128;

// Vector class to help with vectorized memory accesses
template <typename T, size_t kSize>
union Vector {
  using StorageType = typename BytesToType<sizeof(T) * kSize>::Type;
  StorageType storage;
  T entries[kSize];
};

/* Byte-wise less-than comparison
 *
 * Results are stored in each byte's most-significant bit (MSB). All
 * other bits are zero.
 */
__device__ __forceinline__ uint32_t bytewise_less_than(uint32_t a, uint32_t b) {
  // Compare low bits by masking MSBs and subtracting. The resulting
  // MSBs are 0 if the low bits of a are less than the low bits of b.
  uint32_t result = (a | 0x80808080) - (b & 0x7F7F7F7F);

  // Bitwise logical op to get answer in MSBs
  // Equivalent logic: result = (a == b) ? !result : b
  asm("lop3.b32 %0, %1, %2, %3, 0x4D;\n\t" : "=r"(result) : "r"(a), "r"(b), "r"(result));

  // Mask out everything except MSBs and return
  result &= 0x80808080;
  return result;
}

/* Generate dropout mask with 16 bits.
 *
 * 1 corresponds to keep and 0 to drop.
 *
 * Consumes 4 values from cuRAND Philox generator.
 */
__device__ __forceinline__ uint16_t make_16bit_mask(uint64_t chunk_idx, uint64_t rng_seed,
                                                    uint64_t rng_offset,
                                                    uint32_t bytewise_drop_prob) {
  // Generate random bits
  curandStatePhilox4_32_10_t state;
  curand_init(rng_seed, chunk_idx, rng_offset, &state);
  const uint4 rand_bits = curand4(&state);

  // Compute mask
  // Note: bytewise_less_than fills MSBs (bits 7, 15, 23, 31). By
  // shifting 2 bits after every call, every other bit will be filled.
  uint32_t result = bytewise_less_than(rand_bits.x, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.y, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.z, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.w, bytewise_drop_prob);

  // Consolidate mask in lowest 16 bits
  result |= result >> 17;

  // Flip bits so 0 corresponds to drop
  result = ~result;

  return result;
}

// Dropout forward with FP16/BF16 input and output.
template <typename T>
__global__ void __launch_bounds__(block_size)
    dropout_kernel_fwd_f16(const T *__restrict__ input_ptr, T *__restrict__ output_ptr,
                           uint8_t *__restrict__ mask_ptr,
                           const uint64_t *__restrict__ rng_state_ptr, size_t num_chunks,
                           uint32_t bytewise_drop_prob, float scale) {
  static_assert(sizeof(T) == 2);

  // Each thread processes a chunk of 16 entries
  const size_t gid = threadIdx.x + blockIdx.x * block_size;
  const size_t nthreads = gridDim.x * block_size;
  for (size_t chunk_idx = gid; chunk_idx < num_chunks; chunk_idx += nthreads) {
    // Generate dropout mask
    auto local_mask =
        make_16bit_mask(chunk_idx, rng_state_ptr[0], rng_state_ptr[1], bytewise_drop_prob);
    reinterpret_cast<uint16_t *>(mask_ptr)[chunk_idx] = local_mask;

    // Read input data
    using VectorType = Vector<T, rng_chunk_size>;
    VectorType local_data;
    local_data = reinterpret_cast<const VectorType *>(input_ptr)[chunk_idx];

    // Apply dropout based on mask
#pragma unroll
    for (size_t i = 0; i < rng_chunk_size; i++) {
      float val = static_cast<float>(local_data.entries[i]);
      if ((local_mask & 0x1) == 0) {
        val = 0;
      }
      val *= scale;
      local_data.entries[i] = static_cast<T>(val);
      local_mask >>= 1;
    }

    // Write output data
    reinterpret_cast<VectorType *>(output_ptr)[chunk_idx] = local_data;
  }
}

// Dropout forward with FP8 input and FP16/BF16 output.
template <typename InputType, typename OutputType>
__global__ void __launch_bounds__(block_size)
    dropout_kernel_fwd_fp8(const InputType *__restrict__ input_ptr,
                           const float *__restrict__ input_scale_inv_ptr,
                           OutputType *__restrict__ output_ptr, uint8_t *__restrict__ mask_ptr,
                           const uint64_t *__restrict__ rng_state_ptr, size_t num_chunks,
                           uint32_t bytewise_drop_prob, float scale) {
  static_assert(sizeof(InputType) == 1);
  static_assert(sizeof(OutputType) == 2);
  const float input_scale_inv = *input_scale_inv_ptr;

  // Each thread processes a chunk of 16 entries
  const size_t gid = threadIdx.x + blockIdx.x * block_size;
  const size_t nthreads = gridDim.x * block_size;
  for (size_t chunk_idx = gid; chunk_idx < num_chunks; chunk_idx += nthreads) {
    // Generate dropout mask
    auto local_mask =
        make_16bit_mask(chunk_idx, rng_state_ptr[0], rng_state_ptr[1], bytewise_drop_prob);
    reinterpret_cast<uint16_t *>(mask_ptr)[chunk_idx] = local_mask;

    // Read input data
    using InputVectorType = Vector<InputType, rng_chunk_size>;
    InputVectorType local_input;
    local_input = reinterpret_cast<const InputVectorType *>(input_ptr)[chunk_idx];

    // Apply dropout based on mask
    using OutputVectorType = Vector<OutputType, rng_chunk_size>;
    OutputVectorType local_output;
#pragma unroll
    for (size_t i = 0; i < rng_chunk_size; i++) {
      float val = static_cast<float>(local_input.entries[i]);
      val *= input_scale_inv;
      if ((local_mask & 0x1) == 0) {
        val = 0;
      }
      val *= scale;
      local_output.entries[i] = static_cast<OutputType>(val);
      local_mask >>= 1;
    }

    // Write output data
    reinterpret_cast<OutputVectorType *>(output_ptr)[chunk_idx] = local_output;
  }
}

// Apply dropout mask and scale.
template <typename T>
__global__ void __launch_bounds__(block_size)
    apply_dropout_mask(const T *__restrict__ input_ptr, const uint8_t *__restrict__ mask_ptr,
                       T *__restrict__ output_ptr, size_t num_chunks, float scale) {
  // Each thread processes a chunk of 8 entries.
  const size_t gid = threadIdx.x + blockIdx.x * block_size;
  const size_t nthreads = gridDim.x * block_size;
  constexpr size_t chunk_size = 8;
  for (size_t chunk_idx = gid; chunk_idx < num_chunks; chunk_idx += nthreads) {
    // Read dropout mask
    uint8_t local_mask = mask_ptr[chunk_idx];

    // Read input data
    using VectorType = Vector<T, chunk_size>;
    VectorType local_data;
    local_data = reinterpret_cast<const VectorType *>(input_ptr)[chunk_idx];

    // Apply dropout based on mask
#pragma unroll
    for (size_t i = 0; i < chunk_size; i++) {
      float val = static_cast<float>(local_data.entries[i]);
      if ((local_mask & 0x1) == 0) {
        val = 0;
      }
      val *= scale;
      local_data.entries[i] = static_cast<T>(val);
      local_mask >>= 1;
    }

    // Write output data
    reinterpret_cast<VectorType *>(output_ptr)[chunk_idx] = local_data;
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
  NVTE_CHECK(mask.scaling_mode == NVTE_DELAYED_TENSOR_SCALING, "Mask tensor must be plain tensor, ",
             "but scaling mode is ", to_string(mask.scaling_mode), ".");
  NVTE_CHECK(rng_state.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "RNG state tensor must be INT64 tensor with two entries, ", "but scaling mode is ",
             to_string(rng_state.scaling_mode), ".");
  NVTE_CHECK(output.dtype() == DType::kFloat16 || output.dtype() == DType::kBFloat16,
             "Output tensor must be FP16/BF16 tensor, but dtype is ", to_string(output.dtype()),
             ".");
  NVTE_CHECK(rng_state.dtype() == DType::kInt64,
             "RNG state tensor must be INT64 tensor with two entries, but dtype is ",
             to_string(rng_state.dtype()), ".");
  NVTE_CHECK(numel % 16 == 0,
             "Input tensor number of elements must be divisible by 16, but shape is ",
             input.shape(), ".");
  NVTE_CHECK(numel == output.numel(), "Input tensor (shape=", input.shape(),
             ") and output tensor (shape=", output.shape(), ") do not match.");
  NVTE_CHECK(typeToNumBits(mask.dtype()) * mask.numel() == numel, "Mask tensor must have ", numel,
             " bits, but found dtype=", to_string(mask.dtype()), " and shape=", mask.shape(), ".");
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
  uint32_t bytewise_drop_prob = static_cast<uint32_t>(std::floor(dropout_probability * 256));
  bytewise_drop_prob |= bytewise_drop_prob << 8;
  bytewise_drop_prob |= bytewise_drop_prob << 16;

  // CUDA config
  const size_t num_chunks = numel / rng_chunk_size;
  const size_t num_blocks = DIVUP(num_chunks, block_size);

  // Launch kernel depending on input dtype
  if (input.dtype() == DType::kFloat16 || input.dtype() == DType::kBFloat16) {
    NVTE_CHECK(input.dtype() == output.dtype(), "Input tensor (dtype=", to_string(input.dtype()),
               ") and output tensor (dtype=", to_string(output.dtype()), ") do not match.");
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        input.dtype(), DType,
        dropout_kernel_fwd_f16<DType><<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const DType *>(input.data.dptr),
            reinterpret_cast<DType *>(output.data.dptr),
            reinterpret_cast<uint8_t *>(mask.data.dptr),
            reinterpret_cast<const uint64_t *>(rng_state.data.dptr), num_chunks, bytewise_drop_prob,
            scale););
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else if (input.dtype() == DType::kFloat8E4M3 || input.dtype() == DType::kFloat8E5M2) {
    NVTE_CHECK(input.scale_inv.dptr != nullptr, "Input tensor scale-inverse is not allocated.");
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        input.dtype(), InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
            output.dtype(), OutputType,
            dropout_kernel_fwd_fp8<InputType, OutputType><<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const InputType *>(input.data.dptr),
                reinterpret_cast<const float *>(input.scale_inv.dptr),
                reinterpret_cast<OutputType *>(output.data.dptr),
                reinterpret_cast<uint8_t *>(mask.data.dptr),
                reinterpret_cast<const uint64_t *>(rng_state.data.dptr), num_chunks,
                bytewise_drop_prob, scale);

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
  NVTE_CHECK(mask.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Mask tensor must be a plain tensor, but scaling mode is ",
             to_string(mask.scaling_mode), ".");
  NVTE_CHECK(grad_output.dtype() == DType::kFloat16 || grad_output.dtype() == DType::kBFloat16,
             "Grad output tensor must be FP16/BF16 tensor, but dtype is ",
             to_string(grad_output.dtype()), ".");
  NVTE_CHECK(grad_output.dtype() == grad_input.dtype(),
             "Grad output tensor (dtype=", to_string(grad_output.dtype()),
             ") and grad input tensor (dtype=", to_string(grad_input.dtype()), ") do not match.");
  NVTE_CHECK(numel % 16 == 0,
             "Grad output tensor number of elements must be divisible by 16, but shape is ",
             grad_output.shape(), ".");
  NVTE_CHECK(numel == grad_input.numel(), "Grad output tensor (shape=", grad_output.shape(),
             ") and grad input tensor (shape=", grad_input.shape(), ") do not match.");
  NVTE_CHECK(typeToNumBits(mask.dtype()) * mask.numel() == numel, "Mask tensor must have ", numel,
             " bits, but found dtype=", to_string(mask.dtype()), " and shape=", mask.shape(), ".");
  NVTE_CHECK(grad_output.data.dptr != nullptr, "Grad output tensor is missing data.");
  NVTE_CHECK(grad_input.data.dptr != nullptr, "Grad input tensor is missing data.");
  NVTE_CHECK(mask.data.dptr != nullptr, "Mask tensor is missing data.");

  // Convert dropout probablity to scale
  NVTE_CHECK(dropout_probability >= 0 && dropout_probability < 1, "Invalid dropout probability (",
             dropout_probability, ").");
  const float scale = 1 / (1 - dropout_probability);

  // CUDA config
  const size_t num_chunks = numel / 8;
  const size_t num_blocks = DIVUP(num_chunks, block_size);

  // Launch kernel
  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      grad_output.dtype(), DType,
      apply_dropout_mask<DType><<<num_blocks, block_size, 0, stream>>>(
          reinterpret_cast<const DType *>(grad_output.data.dptr),
          reinterpret_cast<const uint8_t *>(mask.data.dptr),
          reinterpret_cast<DType *>(grad_input.data.dptr), num_chunks, scale););
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
