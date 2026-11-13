/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "../cast/fp8/group_amax_fp8.cuh"
#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "../util/ptx.cuh"
#include "recipe_common.cuh"

namespace transformer_engine {
namespace {

__launch_bounds__(1) __global__ void zero_amax_kernel(float *amax_ptr, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  *amax_ptr = 0;
}

// Amax is a pure streaming reduction, so the kernel is built to keep as many
// loads in flight as possible: a persistent grid sized to fill the GPU once
// strides over the tensor in 32-byte chunks, each lane issuing amax_kernel_loads
// independent chunk loads per iteration before folding any of them, and every
// CTA reduces and commits to the output once.  The maximum of |x| is exact and
// order-independent, so the result does not depend on the launch shape.
constexpr int amax_kernel_threads = 256;
constexpr int amax_kernel_loads = 4;
constexpr int amax_chunk_bytes = 32;
constexpr int amax_chunk_words = amax_chunk_bytes / sizeof(uint32_t);

__device__ __forceinline__ void load_amax_chunk(uint32_t (&words)[amax_chunk_words],
                                                const uint32_t *src) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  ptx::ld_global_nc_b32x8(words, src);
#else
  const uint4 lo = __ldg(reinterpret_cast<const uint4 *>(src));
  const uint4 hi = __ldg(reinterpret_cast<const uint4 *>(src) + 1);
  words[0] = lo.x, words[1] = lo.y, words[2] = lo.z, words[3] = lo.w;
  words[4] = hi.x, words[5] = hi.y, words[6] = hi.z, words[7] = hi.w;
#endif
}

/*! \brief Atomically fold max(|input|) into *amax.
 *
 * input + head must be 32-byte aligned.  The head elements before it and the tail
 * elements after its num_chunks whole chunks are read one at a time.
 */
template <typename InputType>
__launch_bounds__(amax_kernel_threads) __global__
    void amax_kernel(const InputType *__restrict__ input, float *__restrict__ amax, const int head,
                     const int64_t num_chunks, const int tail, const float *__restrict__ noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  constexpr int kElemsPerChunk = amax_chunk_bytes / sizeof(InputType);
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  constexpr bool kPacked =
      std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>;
#else
  constexpr bool kPacked = false;
#endif
  using Pair =
      std::conditional_t<std::is_same_v<InputType, __nv_bfloat16>, ptx::bf16x2, ptx::fp16x2>;

  const int64_t stride = static_cast<int64_t>(gridDim.x) * amax_kernel_threads;
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * amax_kernel_threads + threadIdx.x;
  const uint32_t *chunks = reinterpret_cast<const uint32_t *>(input + head);

  float max = 0.f;
  Pair max_pair;
  reinterpret_cast<uint32_t &>(max_pair) = 0;
  auto fold = [&](const uint32_t(&words)[amax_chunk_words]) {
    if constexpr (kPacked) {
      // Packed abs-max tree over the chunk's 8 element pairs.
      const Pair *pairs = reinterpret_cast<const Pair *>(words);
      Pair level[amax_chunk_words / 2];
#pragma unroll
      for (int i = 0; i < amax_chunk_words / 2; ++i) {
        ptx::abs_max_2x(level[i], pairs[2 * i], pairs[2 * i + 1]);
      }
      ptx::abs_max_2x(level[0], level[0], level[1]);
      ptx::abs_max_2x(level[2], level[2], level[3]);
      ptx::abs_max_2x(level[0], level[0], level[2]);
      ptx::abs_max_2x(max_pair, max_pair, level[0]);
    } else {
      const InputType *elems = reinterpret_cast<const InputType *>(words);
#pragma unroll
      for (int i = 0; i < kElemsPerChunk; ++i) {
        max = fmaxf(fabsf(static_cast<float>(elems[i])), max);
      }
    }
  };

  // Fewer than 2 chunks' worth of elements, so the first CTA covers them in one pass.
  // Read before the chunks so their latency overlaps the main loop's.
  if (tid < head) {
    max = fmaxf(fabsf(static_cast<float>(input[tid])), max);
  }
  if (tid < tail) {
    const InputType *tail_ptr = input + head + num_chunks * kElemsPerChunk;
    max = fmaxf(fabsf(static_cast<float>(tail_ptr[tid])), max);
  }

  int64_t chunk = tid;
  for (; chunk + (amax_kernel_loads - 1) * stride < num_chunks;
       chunk += amax_kernel_loads * stride) {
    uint32_t words[amax_kernel_loads][amax_chunk_words];
#pragma unroll
    for (int u = 0; u < amax_kernel_loads; ++u) {
      load_amax_chunk(words[u], chunks + (chunk + u * stride) * amax_chunk_words);
    }
#pragma unroll
    for (int u = 0; u < amax_kernel_loads; ++u) {
      fold(words[u]);
    }
  }
  for (; chunk < num_chunks; chunk += stride) {
    uint32_t words[amax_chunk_words];
    load_amax_chunk(words, chunks + chunk * amax_chunk_words);
    fold(words);
  }

  if constexpr (kPacked) {
    // max.xorsign.abs may leave a sign behind; only the magnitude is wanted.
    max = fmaxf(
        max, fmaxf(fabsf(static_cast<float>(max_pair.x)), fabsf(static_cast<float>(max_pair.y))));
  }
  max = reduce_max<amax_kernel_threads / THREADS_PER_WARP>(max, threadIdx.x / THREADS_PER_WARP);
  if (threadIdx.x == 0) {
    atomicMaxFloat(amax, max);
  }
}

template <typename InputType>
void launch_amax_kernel(const InputType *input, float *amax, const size_t N, const float *noop_ptr,
                        cudaStream_t stream) {
  // Zero out amax so we can update with atomic max
  zero_amax_kernel<<<1, 1, 0, stream>>>(amax, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  // Return immediately if tensor is empty
  if (N == 0) {
    return;
  }

  // Split the tensor at its first and last 32-byte boundaries.
  constexpr size_t elems_per_chunk = amax_chunk_bytes / sizeof(InputType);
  NVTE_CHECK(is_aligned_ptr(input, sizeof(InputType)), "Amax input is not aligned to its dtype.");
  const size_t misalignment = reinterpret_cast<uintptr_t>(input) % amax_chunk_bytes;
  const size_t head =
      std::min(N, misalignment == 0 ? 0 : (amax_chunk_bytes - misalignment) / sizeof(InputType));
  const int64_t num_chunks = static_cast<int64_t>((N - head) / elems_per_chunk);
  const size_t tail = N - head - static_cast<size_t>(num_chunks) * elems_per_chunk;

  // Persistent grid: never more CTAs than the GPU holds at once.
  static const int ctas_per_sm = [] {
    int n = 0;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, amax_kernel<InputType>,
                                                                  amax_kernel_threads, 0));
    return n;
  }();
  const int64_t max_ctas = static_cast<int64_t>(ctas_per_sm) * cuda::sm_count();
  // Below that, one chunk per lane: spreading a small tensor over more CTAs
  // beats giving fewer lanes several chunks each.
  const int64_t needed_ctas = DIVUP(num_chunks, static_cast<int64_t>(amax_kernel_threads));
  const int64_t grid = std::max<int64_t>(1, std::min(max_ctas, needed_ctas));
  amax_kernel<InputType><<<grid, amax_kernel_threads, 0, stream>>>(
      input, amax, static_cast<int>(head), num_chunks, static_cast<int>(tail), noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace
}  // namespace transformer_engine

namespace {

void compute_amax_impl(const NVTETensor input_, const NVTETensor output_, cudaStream_t stream,
                       const NVTEQuantizationConfig config_) {
  using namespace transformer_engine;

  // Check input tensor
  NVTE_CHECK(input_ != nullptr, "Invalid input tensor (got NULL)");
  const auto &input = *convertNVTETensorCheck(input_);
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
  auto &output = *convertNVTETensorCheck(output_);
  NVTE_CHECK(output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING ||
                 output.scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Output tensor for amax computation must be FP8 tensor with per-tensor scaling or "
             "NVFP4 1D scaling, "
             "but got scaling_mode=",
             to_string(output.scaling_mode));
  NVTE_CHECK(output.amax.numel() == 1,
             "Output tensor for amax computation has invalid amax tensor "
             "(expected 1 entry, got shape=",
             output.amax.shape, ")");
  NVTE_CHECK(output.amax.dptr != nullptr || output.columnwise_amax.dptr != nullptr,
             "Output tensor for amax computation has amax tensor without data");
  NVTE_CHECK(output.amax.dtype == DType::kFloat32,
             "Output tensor for amax computation has invalid amax tensor  "
             "(expected FP32, got dtype=",
             to_string(output.amax.dtype), ")");
  CheckOutputTensor(output, "output_compute_amax", true);

  float *noop_ptr = nullptr;
  if (config_ != nullptr) {
    const QuantizationConfig *config_cpp = reinterpret_cast<const QuantizationConfig *>(config_);

    // extract noop tensor from quant_config_cpp if it's not null
    const NVTETensor noop = config_cpp ? config_cpp->noop_tensor : nullptr;
    noop_ptr = reinterpret_cast<float *>(
        (noop != nullptr ? convertNVTETensorCheck(noop)->data.dptr : nullptr));
  }

  // Compute amax
  float *amax_ptr = reinterpret_cast<float *>(
      (output.amax.dptr != nullptr) ? output.amax.dptr : output.columnwise_amax.dptr);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      launch_amax_kernel(reinterpret_cast<const IType *>(input.data.dptr), amax_ptr,
                         input.data.numel(), noop_ptr, stream););  // NOLINT(*)
}

void group_compute_amax_impl(const NVTEGroupedTensor input_, NVTEGroupedTensor output_,
                             const NVTEQuantizationConfig config_, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(input_ != nullptr, "Invalid grouped input tensor (got NULL)");
  const auto &input = *convertNVTEGroupedTensorCheck(input_);
  NVTE_CHECK(output_ != nullptr, "Invalid grouped output tensor (got NULL)");
  auto &output = *convertNVTEGroupedTensorCheck(output_);
  NVTE_CHECK(input.num_tensors == output.num_tensors,
             "Number of grouped input and output tensors must match.");
  NVTE_CHECK(input.has_data(), "Grouped amax input must have rowwise data.");
  NVTE_CHECK(!is_fp8_dtype(input.data.dtype),
             "Grouped amax input must be unquantized, but got dtype=", to_string(input.data.dtype));
  NVTE_CHECK(output.amax.has_data() || output.columnwise_amax.has_data(),
             "Grouped amax output must have an amax buffer.");

  CheckInputGroupedTensor(input, "group_compute_amax_input");
  CheckOutputGroupedTensor(output, "group_compute_amax_output", true);

  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (output.all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (output.all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else if (output.all_same_first_dim()) {
    shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
  } else {
    shape_rep = ShapeRepresentation::VARYING_BOTH_DIMS;
  }

  const int64_t *offsets_ptr = reinterpret_cast<const int64_t *>(output.tensor_offsets.dptr);
  const int64_t *first_dims_ptr = reinterpret_cast<const int64_t *>(output.first_dims.dptr);
  const int64_t *last_dims_ptr = reinterpret_cast<const int64_t *>(output.last_dims.dptr);
  if (shape_rep != ShapeRepresentation::SAME_BOTH_DIMS) {
    NVTE_CHECK(offsets_ptr != nullptr,
               "Grouped amax requires tensor_offsets when a grouped dimension varies.");
  }
  if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM ||
      shape_rep == ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(first_dims_ptr != nullptr,
               "Grouped amax requires first_dims for varying first dimensions.");
  }
  if (shape_rep == ShapeRepresentation::VARYING_LAST_DIM ||
      shape_rep == ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(last_dims_ptr != nullptr,
               "Grouped amax requires last_dims for varying last dimensions.");
  }

  float *noop_ptr = nullptr;
  if (config_ != nullptr) {
    const QuantizationConfig *config_cpp = reinterpret_cast<const QuantizationConfig *>(config_);
    const NVTETensor noop = config_cpp ? config_cpp->noop_tensor : nullptr;
    noop_ptr = reinterpret_cast<float *>(
        (noop != nullptr ? convertNVTETensorCheck(noop)->data.dptr : nullptr));
  }

  float *amax_ptr = reinterpret_cast<float *>(
      output.amax.dptr != nullptr ? output.amax.dptr : output.columnwise_amax.dptr);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      dispatch::fp8::launch_grouped_amax_kernel(
          reinterpret_cast<const IType *>(input.data.dptr), amax_ptr, output.num_tensors,
          output.logical_shape.data[0], output.logical_shape.data[1], shape_rep, offsets_ptr,
          first_dims_ptr, last_dims_ptr, noop_ptr, stream););  // NOLINT(*)
}

}  // anonymous namespace

void nvte_compute_amax(const NVTETensor input_, const NVTETensor output_, cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_amax);
  compute_amax_impl(input_, output_, stream, nullptr);
}

void nvte_compute_amax_with_config(const NVTETensor input_, const NVTETensor output_,
                                   const NVTEQuantizationConfig config_, cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_amax_with_config);
  compute_amax_impl(input_, output_, stream, config_);
}

void nvte_group_compute_amax_with_config(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                         const NVTEQuantizationConfig config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_compute_amax_with_config);
  group_compute_amax_impl(input, output, config, stream);
}

namespace transformer_engine {
namespace {

__global__ void compute_scale_from_amax_kernel(const float *amax_ptr, float *scale_ptr,
                                               const float max_fp8, const bool force_pow_2_scales,
                                               const float epsilon, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  *scale_ptr = compute_scale_from_amax(*amax_ptr, max_fp8, force_pow_2_scales, epsilon,
                                       std::numeric_limits<float>::max());
}

}  // namespace
}  // namespace transformer_engine

void nvte_compute_scale_from_amax(NVTETensor output_, const NVTEQuantizationConfig config_,
                                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_scale_from_amax);
  using namespace transformer_engine;

  // Check output tensor
  NVTE_CHECK(output_ != nullptr, "Invalid output tensor (got NULL)");
  auto &output = *convertNVTETensorCheck(output_);
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

  // noop tensor for cuda graph
  float *noop_ptr = nullptr;
  if (config_ != nullptr) {
    const QuantizationConfig *config_cpp = reinterpret_cast<const QuantizationConfig *>(config_);

    // extract noop tensor from quant_config_cpp if it's not null
    const NVTETensor noop = config_cpp ? config_cpp->noop_tensor : nullptr;
    noop_ptr = reinterpret_cast<float *>(
        (noop != nullptr ? convertNVTETensorCheck(noop)->data.dptr : nullptr));
  }

  // Update scale
  compute_scale_from_amax_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const float *>(output.amax.dptr),
      reinterpret_cast<float *>(output.scale.dptr), max_fp8, config.force_pow_2_scales,
      config.amax_epsilon, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());
}
