/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "recipe_common.cuh"

namespace transformer_engine {
namespace {

constexpr int amax_kernel_threads = 512;
constexpr int grouped_amax_kernel_threads = 256;
// Target block-per-tensor multiplier so that, even with few groups, we cover
// enough SMs on H100/GB200-class GPUs (132 SMs) to be HBM-bandwidth bound.
// blocks_per_tensor is bounded by both work and a hard cap to avoid launching
// far more blocks than there is work for.
constexpr int grouped_amax_blocks_per_tensor_cap = 64;
constexpr size_t grouped_amax_min_elts_per_block = 8 * 1024;  // ~16KB of bf16

__launch_bounds__(1) __global__ void zero_amax_kernel(float *amax_ptr, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  *amax_ptr = 0;
}

template <int nvec, bool aligned, typename InputType>
__launch_bounds__(amax_kernel_threads) __global__
    void amax_kernel(const InputType *input, float *amax, const size_t N,
                     const size_t num_aligned_elements, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

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
void launch_amax_kernel(const InputType *input, float *amax, const size_t N, const float *noop_ptr,
                        cudaStream_t stream) {
  // Zero out amax so we can update with atomic max
  zero_amax_kernel<<<1, 1, 0, stream>>>(amax, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

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
          <<<num_blocks, threads, 0, stream>>>(input, amax, N, num_aligned_elements, noop_ptr);
      break;
    case Alignment::SAME_UNALIGNED:
      amax_kernel<nvec, false, InputType>
          <<<num_blocks, threads, 0, stream>>>(input, amax, N, num_aligned_elements, noop_ptr);
      break;
    case Alignment::DIFFERENT: {
      // This case is a logic error, since there is only one pointer (input)
      // in the alignment check. Still safe to process without vectorization.
      amax_kernel<1, true, InputType>
          <<<num_blocks, threads, 0, stream>>>(input, amax, N, N, noop_ptr);
      break;
    }
  }

  // Check results
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Zero per-tensor amax buffer so the main kernel can use atomicMax updates.
__launch_bounds__(grouped_amax_kernel_threads) __global__
    void grouped_amax_zero_kernel(float *amax_ptr, const size_t num_tensors,
                                  const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_tensors) {
    amax_ptr[tid] = 0.0f;
  }
}

// Vectorized per-tensor amax kernel.
//
// Grid: (blocks_per_tensor, num_tensors).
// Each block scans a stride of vectors within its tensor and atomicMaxFloat's
// the result into amax[tensor_id]. Uses 16-byte vector loads, warp-shuffle
// reduction, and (in the SAME_BOTH_DIMS / static-offset paths) avoids any
// per-tensor metadata lookups.
//
// For varying shapes, we use offsets_ptr[i+1] - offsets_ptr[i] as the strict
// upper bound on this tensor's element count. This matches the layout that
// build_grouped_tensor_offsets uses (the "logical" element span for the
// tensor) and means we never read overallocated tail rows/cols.
template <int NVEC, typename InputType, ShapeRepresentation SHAPE_REP>
__launch_bounds__(grouped_amax_kernel_threads) __global__
    void grouped_amax_kernel_v2(const InputType *__restrict__ input, float *__restrict__ amax,
                                const size_t num_tensors, const size_t first_logical_dim,
                                const size_t last_logical_dim,
                                const int64_t *__restrict__ offsets_ptr,
                                const float *__restrict__ noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t tensor_id = blockIdx.y;
  if (tensor_id >= num_tensors) {
    return;
  }

  size_t tensor_base = 0;
  size_t numel = 0;
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t rows = first_logical_dim / num_tensors;
    numel = rows * last_logical_dim;
    tensor_base = tensor_id * numel;
  } else {
    // Varying first / last / both: strictly use the logical offsets so that
    // we never scan unused tail rows.
    tensor_base = static_cast<size_t>(offsets_ptr[tensor_id]);
    const size_t tensor_end = static_cast<size_t>(offsets_ptr[tensor_id + 1]);
    numel = tensor_end > tensor_base ? tensor_end - tensor_base : 0;
  }
  if (numel == 0) {
    return;
  }

  using IVecT = Vec<InputType, NVEC>;
  const InputType *base = input + tensor_base;
  const size_t total_vecs = numel / NVEC;
  const size_t tail_start = total_vecs * NVEC;

  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t threads_per_grid = blockDim.x * gridDim.x;

  float thread_amax = 0.0f;

  // Vectorized 16B-load grid-stride loop.
  const bool aligned = (reinterpret_cast<uintptr_t>(base) % IVecT::BYTES) == 0;
  const size_t start_v = bid * blockDim.x + tid;
  if (aligned) {
    for (size_t v = start_v; v < total_vecs; v += threads_per_grid) {
      IVecT vec;
      vec.load_from(base + v * NVEC);
#pragma unroll
      for (int i = 0; i < NVEC; ++i) {
        thread_amax = fmaxf(thread_amax, fabsf(static_cast<float>(vec.data.elt[i])));
      }
    }
  } else {
    for (size_t v = start_v; v < total_vecs; v += threads_per_grid) {
      const InputType *p = base + v * NVEC;
#pragma unroll
      for (int i = 0; i < NVEC; ++i) {
        thread_amax = fmaxf(thread_amax, fabsf(static_cast<float>(p[i])));
      }
    }
  }

  // Tail: at most NVEC-1 elements, handled only by block 0.
  if (bid == 0) {
    for (size_t i = tail_start + tid; i < numel; i += blockDim.x) {
      thread_amax = fmaxf(thread_amax, fabsf(static_cast<float>(base[i])));
    }
  }

  // Warp-shuffle reduce.
#pragma unroll
  for (int s = THREADS_PER_WARP / 2; s > 0; s >>= 1) {
    thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFFu, thread_amax, s));
  }
  constexpr int kWarps = grouped_amax_kernel_threads / THREADS_PER_WARP;
  __shared__ float warp_amax[kWarps];
  const int warp_id = tid / THREADS_PER_WARP;
  const int lane = tid % THREADS_PER_WARP;
  if (lane == 0) {
    warp_amax[warp_id] = thread_amax;
  }
  __syncthreads();

  if (warp_id == 0) {
    float v = lane < kWarps ? warp_amax[lane] : 0.0f;
#pragma unroll
    for (int s = kWarps / 2; s > 0; s >>= 1) {
      v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, s));
    }
    if (lane == 0) {
      atomicMaxFloat(&amax[tensor_id], v);
    }
  }
}

// Pick blocks-per-tensor so each block has at least ~grouped_amax_min_elts_per_block elements
// of work. Capped at grouped_amax_blocks_per_tensor_cap to avoid launching way more blocks
// than the GPU can resident.
inline size_t choose_grouped_amax_blocks_per_tensor(size_t max_elts_per_tensor) {
  if (max_elts_per_tensor == 0) return 1;
  size_t blocks = (max_elts_per_tensor + grouped_amax_min_elts_per_block - 1) /
                  grouped_amax_min_elts_per_block;
  if (blocks > static_cast<size_t>(grouped_amax_blocks_per_tensor_cap)) {
    blocks = grouped_amax_blocks_per_tensor_cap;
  }
  if (blocks == 0) blocks = 1;
  return blocks;
}

template <typename InputType>
void launch_grouped_amax_kernel(const InputType *input, float *amax, const size_t num_tensors,
                                const size_t first_logical_dim, const size_t last_logical_dim,
                                const ShapeRepresentation shape_rep, const int64_t *offsets_ptr,
                                const int64_t *first_dims_ptr, const int64_t *last_dims_ptr,
                                const float *noop_ptr, cudaStream_t stream) {
  if (num_tensors == 0) {
    return;
  }
  (void)first_dims_ptr;
  (void)last_dims_ptr;

  // Estimate the maximum per-tensor element count without a D2H copy:
  //  - SAME_BOTH_DIMS:    first_logical_dim/num_tensors * last_logical_dim
  //  - others:            an over-estimate first_logical_dim*last_logical_dim
  // This is only used to size the launch (blocks-per-tensor).
  size_t max_elts_per_tensor = 0;
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    max_elts_per_tensor = (first_logical_dim / num_tensors) * last_logical_dim;
  } else {
    max_elts_per_tensor = first_logical_dim * last_logical_dim;
  }

  // Zero out the per-tensor amax buffer so atomicMaxFloat works.
  const size_t zero_blocks =
      (num_tensors + grouped_amax_kernel_threads - 1) / grouped_amax_kernel_threads;
  grouped_amax_zero_kernel<<<zero_blocks, grouped_amax_kernel_threads, 0, stream>>>(
      amax, num_tensors, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  constexpr int kVecBytes = 16;
  constexpr int kNvec = kVecBytes / sizeof(InputType);
  static_assert(kNvec >= 1, "Vector width must be at least 1");

  const size_t blocks_per_tensor = choose_grouped_amax_blocks_per_tensor(max_elts_per_tensor);
  const dim3 grid(blocks_per_tensor, num_tensors);
  const dim3 block(grouped_amax_kernel_threads);

  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
      grouped_amax_kernel_v2<kNvec, InputType, ShapeRepresentation::SAME_BOTH_DIMS>
          <<<grid, block, 0, stream>>>(input, amax, num_tensors, first_logical_dim,
                                       last_logical_dim, offsets_ptr, noop_ptr);
      break;
    case ShapeRepresentation::VARYING_FIRST_DIM:
      grouped_amax_kernel_v2<kNvec, InputType, ShapeRepresentation::VARYING_FIRST_DIM>
          <<<grid, block, 0, stream>>>(input, amax, num_tensors, first_logical_dim,
                                       last_logical_dim, offsets_ptr, noop_ptr);
      break;
    case ShapeRepresentation::VARYING_LAST_DIM:
      grouped_amax_kernel_v2<kNvec, InputType, ShapeRepresentation::VARYING_LAST_DIM>
          <<<grid, block, 0, stream>>>(input, amax, num_tensors, first_logical_dim,
                                       last_logical_dim, offsets_ptr, noop_ptr);
      break;
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      grouped_amax_kernel_v2<kNvec, InputType, ShapeRepresentation::VARYING_BOTH_DIMS>
          <<<grid, block, 0, stream>>>(input, amax, num_tensors, first_logical_dim,
                                       last_logical_dim, offsets_ptr, noop_ptr);
      break;
  }
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
      input.data.dtype, IType, constexpr int nvec = 32 / sizeof(IType); launch_amax_kernel<nvec>(
          reinterpret_cast<const IType *>(input.data.dptr), amax_ptr, input.data.numel(), noop_ptr,
          stream););  // NOLINT(*)
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
             "Grouped amax input must be unquantized, but got dtype=",
             to_string(input.data.dtype));
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
      launch_grouped_amax_kernel(reinterpret_cast<const IType *>(input.data.dptr), amax_ptr,
                                 output.num_tensors, output.logical_shape.data[0],
                                 output.logical_shape.data[1], shape_rep, offsets_ptr,
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
                                         const NVTEQuantizationConfig config,
                                         cudaStream_t stream) {
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
