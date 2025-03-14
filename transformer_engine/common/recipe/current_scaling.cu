/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "../common.h"
#include "../util/logging.h"
#include "../util/vectorized_pointwise.h"

namespace transformer_engine {
namespace {

constexpr int amax_kernel_threads = 512;

template <int nvec, bool aligned, typename InputType>
__launch_bounds__(amax_kernel_threads) __global__
    void amax_kernel(const InputType *input, float *amax, const size_t N,
                     const size_t num_aligned_elements) {
  VectorizedLoader<InputType, nvec, aligned> loader(input, N);
  InputType max = 0.f;
  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const size_t M = num_aligned_elements;

  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < M; tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const InputType val = static_cast<InputType>(loader.separate()[i]);
      __builtin_assume(max >= InputType{0.f});
      if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
        max = __hmax(__habs(val), max);
#else  // Turing
        max = static_cast<__nv_bfloat16>(
            fmaxf(fabsf(static_cast<float>(val)), static_cast<float>(max)));
#endif
      } else if constexpr (std::is_same_v<InputType, __half>) {
        max = __hmax(__habs(val), max);
      } else {
        max = fmaxf(fabsf(val), max);
      }
    }
  }

  // Reduce amax over block
  max = reduce_max<amax_kernel_threads / THREADS_PER_WARP>(max, warp_id);
  if (threadIdx.x == 0) {
    atomicMaxFloat(amax, max);
  }
}

template <int nvec, typename InputType>
void launch_amax_kernel(const InputType *input, float *amax, const size_t N, cudaStream_t stream) {
  // Zero out amax so we can update with atomic max
  cudaMemsetAsync(amax, 0, sizeof(float), stream);

  // Return immediately if tensor is empty
  if (N == 0) {
    return;
  }

  // Figure out alignment
  auto align = CheckAlignment(N, nvec, input);
  size_t num_aligned_elements = get_num_aligned_elements(input, N, nvec, sizeof(InputType));

  // Figure out CUDA blocks
  constexpr size_t threads = amax_kernel_threads;
  size_t num_blocks = DIVUP(num_aligned_elements, threads);
  constexpr size_t max_blocks = 65535;
  num_blocks = std::min(num_blocks, max_blocks);

  // Launch kernel
  switch (align) {
    case Alignment::SAME_ALIGNED:
      amax_kernel<nvec, true, InputType>
          <<<num_blocks, threads, 0, stream>>>(input, amax, N, num_aligned_elements);
      break;
    case Alignment::SAME_UNALIGNED:
      amax_kernel<nvec, false, InputType>
          <<<num_blocks, threads, 0, stream>>>(input, amax, N, num_aligned_elements);
      break;
    case Alignment::DIFFERENT: {
      // This case is a logic error, since there is only one pointer (input)
      // in the alignment check. Still safe to process without vectorization.
      amax_kernel<1, true, InputType><<<num_blocks, threads, 0, stream>>>(input, amax, N, N);
      break;
    }
  }

  // Check results
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace
}  // namespace transformer_engine

void nvte_compute_amax(const NVTETensor input_, const NVTETensor output_, cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_amax);
  using namespace transformer_engine;

  // Check input tensor
  NVTE_CHECK(input_ != nullptr, "Invalid input tensor (got NULL)");
  const auto &input = *reinterpret_cast<const Tensor *>(input_);
  NVTE_CHECK(input.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor for amax computation must unquantized, "
             "but got scaling_mode=",
             to_string(input.scaling_mode));
  NVTE_CHECK(!is_fp8_dtype(input.data.dtype),
             "Input tensor for amax computation must be unquantized, but got dtype=",
             to_string(input.data.dtype));
  NVTE_CHECK(input.data.dptr != nullptr, "Input tensor for amax computation has no data");
  CheckInputTensor(input, "input_compute_amax");

  // Check output tensor
  NVTE_CHECK(output_ != nullptr, "Invalid output tensor (got NULL)");
  auto &output = *reinterpret_cast<Tensor *>(output_);
  NVTE_CHECK(output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Output tensor for amax computation must be FP8 tensor with per-tensor scaling, "
             "but got scaling_mode=",
             to_string(output.scaling_mode));
  NVTE_CHECK(output.amax.numel() == 1,
             "Output tensor for amax computation has invalid amax tensor "
             "(expected 1 entry, got shape=",
             output.amax.shape, ")");
  NVTE_CHECK(output.amax.dptr != nullptr,
             "Output tensor for amax computation has amax tensor without data");
  NVTE_CHECK(output.amax.dtype == DType::kFloat32,
             "Output tensor for amax computation has invalid amax tensor  "
             "(expected FP32, got dtype=",
             to_string(output.amax.dtype), ")");
  CheckOutputTensor(output, "output_compute_amax");

  // Compute amax
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType, constexpr int nvec = 32 / sizeof(IType);
      launch_amax_kernel<nvec>(reinterpret_cast<const IType *>(input.data.dptr),
                               reinterpret_cast<float *>(output.amax.dptr), input.data.numel(),
                               stream););  // NOLINT(*)
}

namespace transformer_engine {
namespace {

__global__ void compute_scale_from_amax_kernel(const float *amax_ptr, float *scale_ptr,
                                               const float max_fp8, const bool force_pow_2_scales,
                                               const float epsilon) {
  float amax = *amax_ptr;
  if (amax < epsilon) {
    amax = epsilon;
  }

  float scale = 1.f;

  if (isinf(amax) || amax == 0.f) {
    *scale_ptr = scale;
    return;
  }

  scale = max_fp8 / amax;

  // The amax is too small that the scale becoming infinite in FP32. In other word,
  // the scale is not representable in FP32.
  if (isinf(scale)) {
    // use fp32 max to represent the scale
    scale = std::numeric_limits<float>::max();
  }

  if (isnan(scale)) {
    scale = 1.f;
  }

  if (force_pow_2_scales) {
    uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
    scale_bits &= 0xFF800000;
    // If the exponent was zero, we have a logic error.
    __builtin_assume(scale_bits != 0);
    __builtin_assume(scale_bits != 0x80000000);
    scale = *reinterpret_cast<float *>(&scale_bits);
  }

  *scale_ptr = scale;
}

}  // namespace
}  // namespace transformer_engine

void nvte_compute_scale_from_amax(NVTETensor output_, const NVTEQuantizationConfig config_,
                                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_scale_from_amax);
  using namespace transformer_engine;

  // Check output tensor
  NVTE_CHECK(output_ != nullptr, "Invalid output tensor (got NULL)");
  auto &output = *reinterpret_cast<Tensor *>(output_);
  NVTE_CHECK(output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Tensor must be FP8 tensor with per-tensor scaling, "
             "but got scaling_mode=",
             to_string(output.scaling_mode));
  NVTE_CHECK(is_fp8_dtype(output.data.dtype),
             "Tensor must be FP8, but got dtype=", to_string(output.data.dtype));
  NVTE_CHECK(output.amax.numel() == 1,
             "Tensor has invalid amax tensor (expected 1 entry, got shape=", output.amax.shape,
             ")");
  NVTE_CHECK(output.amax.dptr != nullptr, "Tensor has amax tensor without data");
  NVTE_CHECK(output.amax.dtype == DType::kFloat32,
             "Tensor has invalid amax tensor (expected FP32, got dtype=",
             to_string(output.amax.dtype), ")");
  NVTE_CHECK(output.scale.numel() == 1,
             "Tensor has invalid scale tensor (expected 1 entry, got shape=", output.scale.shape,
             ")");
  NVTE_CHECK(output.scale.dptr != nullptr, "Tensor has scale tensor without data");
  NVTE_CHECK(output.scale.dtype == DType::kFloat32,
             "Tensor has invalid scale tensor (expected FP32, got dtype=",
             to_string(output.scale.dtype), ")");

  // Check config
  NVTE_CHECK(config_ != nullptr, "Invalid config (got NULL)");
  const auto &config = *reinterpret_cast<const QuantizationConfig *>(config_);

  // Maximum FP8 value
  float max_fp8 = 0.f;
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output.data.dtype, DType,
                                         max_fp8 = Quantized_Limits<DType>::max_norm;);

  // Update scale
  compute_scale_from_amax_kernel<<<1, 1>>>(reinterpret_cast<const float *>(output.amax.dptr),
                                           reinterpret_cast<float *>(output.scale.dptr), max_fp8,
                                           config.force_pow_2_scales, config.amax_epsilon);
  NVTE_CHECK_CUDA(cudaGetLastError());
}
