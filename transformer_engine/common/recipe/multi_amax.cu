/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <algorithm>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "../util/vectorized_pointwise.h"
#include "recipe_common.cuh"

namespace transformer_engine {
namespace {

constexpr int multi_amax_kernel_threads = 512;
// Per-launch capacity.  kMaxTensorsPerBatch * ~40 bytes per slot keeps the args
// struct within the 4KB kernel parameter limit with comfortable headroom.
constexpr int kMaxTensorsPerBatch = 64;

struct MultiAmaxArgs {
  const void *input_list[kMaxTensorsPerBatch];
  void *output_rowwise_amax_list[kMaxTensorsPerBatch];
  void *output_columnwise_amax_list[kMaxTensorsPerBatch];
  size_t input_numel[kMaxTensorsPerBatch];
  size_t num_aligned_elements[kMaxTensorsPerBatch];
  int num_tensors;
};

// Zero out every output amax slot (rowwise + columnwise, deduped) in a single launch.
// Respects the noop_ptr contract shared with the single-tensor amax path.
__launch_bounds__(multi_amax_kernel_threads) __global__
    void MultiZeroAmaxKernel(MultiAmaxArgs args, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < args.num_tensors; tid += stride) {
    float *rw = static_cast<float *>(args.output_rowwise_amax_list[tid]);
    float *cw = static_cast<float *>(args.output_columnwise_amax_list[tid]);
    if (rw != nullptr) {
      *rw = 0.0f;
    }
    if (cw != nullptr && cw != rw) {
      *cw = 0.0f;
    }
  }
}

// Per-tensor amax with one block-strip per tensor.  blockIdx.y selects the
// tensor; blockIdx.x is the work chunk within that tensor.  Each block
// vector-loads the tensor, reduces across threads, and atomicMaxFloats the
// result into BOTH output amax slots (rowwise + columnwise, deduped).  This
// subsumes the per-expert D2D copy that the single-tensor path does after the
// amax kernel.
template <int nvec, bool aligned, typename InputType>
__launch_bounds__(multi_amax_kernel_threads) __global__
    void MultiAmaxKernel(MultiAmaxArgs args, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const int t_idx = blockIdx.y;
  if (t_idx >= args.num_tensors) {
    return;
  }

  const InputType *input = static_cast<const InputType *>(args.input_list[t_idx]);
  const size_t N = args.input_numel[t_idx];
  if (N == 0) {
    return;
  }
  const size_t M = args.num_aligned_elements[t_idx];

  VectorizedLoader<InputType, nvec, aligned> loader(input, N);
  InputType max = InputType{0.f};
  const int warp_id = threadIdx.x / THREADS_PER_WARP;

  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const InputType val = static_cast<InputType>(loader.separate()[i]);
      __builtin_assume(max >= InputType{0.f});
      if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
        max = __hmax(__habs(val), max);
#else
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

  // Reduce amax over block.
  max = reduce_max<multi_amax_kernel_threads / THREADS_PER_WARP>(max, warp_id);
  if (threadIdx.x == 0) {
    float *rw = static_cast<float *>(args.output_rowwise_amax_list[t_idx]);
    float *cw = static_cast<float *>(args.output_columnwise_amax_list[t_idx]);
    if (rw != nullptr) {
      atomicMaxFloat(rw, static_cast<float>(max));
    }
    if (cw != nullptr && cw != rw) {
      atomicMaxFloat(cw, static_cast<float>(max));
    }
  }
}

template <typename InputType>
void launch_multi_amax_batch(const MultiAmaxArgs &args, size_t max_numel, Alignment align,
                             const float *noop_ptr, cudaStream_t stream) {
  // Zero all amax outputs in one launch.
  {
    constexpr int threads = multi_amax_kernel_threads;
    const int num_blocks = std::max(1, DIVUP(args.num_tensors, threads));
    MultiZeroAmaxKernel<<<num_blocks, threads, 0, stream>>>(args, noop_ptr);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }

  if (max_numel == 0) {
    return;
  }

  // Grid: y = tensor index, x = work chunks within the largest tensor.  Blocks
  // that exceed a shorter tensor's aligned element count bail out via the
  // bounds check inside the kernel.
  constexpr int nvec = 32 / sizeof(InputType);
  constexpr size_t threads = multi_amax_kernel_threads;
  const size_t max_aligned = (max_numel + nvec - 1) / nvec;
  size_t num_blocks_x = DIVUP(max_aligned, threads);
  constexpr size_t max_blocks = 65535;
  num_blocks_x = std::min(num_blocks_x, max_blocks);
  num_blocks_x = std::max<size_t>(num_blocks_x, 1);
  dim3 grid(num_blocks_x, static_cast<unsigned int>(args.num_tensors), 1);

  switch (align) {
    case Alignment::SAME_ALIGNED:
      MultiAmaxKernel<nvec, true, InputType>
          <<<grid, threads, 0, stream>>>(args, noop_ptr);
      break;
    case Alignment::SAME_UNALIGNED:
      MultiAmaxKernel<nvec, false, InputType>
          <<<grid, threads, 0, stream>>>(args, noop_ptr);
      break;
    case Alignment::DIFFERENT:
      // Heterogeneous alignment across tensors — fall back to nvec=1, aligned=true path
      // which is safe for any pointer alignment.
      MultiAmaxKernel<1, true, InputType>
          <<<grid, threads, 0, stream>>>(args, noop_ptr);
      break;
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Fill one MultiAmaxArgs batch from a slice of the full input/output list.
// Returns (max_numel in this batch, worst-case alignment across the batch).
template <typename InputType>
std::pair<size_t, Alignment> build_batch_args(const std::vector<Tensor *> &inputs,
                                              const std::vector<Tensor *> &outputs, size_t start,
                                              size_t count, MultiAmaxArgs &args) {
  constexpr int nvec = 32 / sizeof(InputType);
  size_t max_numel = 0;
  // SAME_ALIGNED is the most optimistic; degrade to SAME_UNALIGNED if any
  // tensor is merely same-layout but unaligned, to DIFFERENT if alignment
  // varies across tensors.
  Alignment batch_align = Alignment::SAME_ALIGNED;
  for (size_t i = 0; i < count; ++i) {
    const Tensor &inp = *inputs[start + i];
    Tensor &out = *outputs[start + i];
    const size_t N = inp.data.numel();
    void *rw_ptr = out.amax.dptr;
    void *cw_ptr = out.columnwise_amax.dptr;

    args.input_list[i] = inp.data.dptr;
    args.output_rowwise_amax_list[i] = rw_ptr;
    args.output_columnwise_amax_list[i] = cw_ptr;
    args.input_numel[i] = N;
    args.num_aligned_elements[i] = get_num_aligned_elements(inp.data.dptr, N, nvec,
                                                            sizeof(InputType));
    max_numel = std::max(max_numel, N);

    // Fold this tensor's alignment into the batch decision.  CheckAlignment on a
    // single pointer yields SAME_ALIGNED or SAME_UNALIGNED; mixing the two across
    // tensors means heterogeneous — switch to the DIFFERENT fall-back.
    if (N > 0) {
      Alignment a = CheckAlignment(N, nvec, static_cast<const InputType *>(inp.data.dptr));
      if (batch_align == Alignment::SAME_ALIGNED && a == Alignment::SAME_UNALIGNED) {
        batch_align = Alignment::SAME_UNALIGNED;
      } else if (batch_align == Alignment::SAME_UNALIGNED && a == Alignment::SAME_ALIGNED) {
        batch_align = Alignment::SAME_UNALIGNED;
      } else if (a == Alignment::DIFFERENT) {
        batch_align = Alignment::DIFFERENT;
      }
    }
  }
  args.num_tensors = static_cast<int>(count);
  return {max_numel, batch_align};
}

void multi_compute_amax_impl(const NVTETensor *inputs_, NVTETensor *outputs_, size_t num_tensors,
                             const NVTEQuantizationConfig config_, cudaStream_t stream) {
  if (num_tensors == 0) {
    return;
  }
  NVTE_CHECK(inputs_ != nullptr, "nvte_multi_compute_amax: inputs is NULL");
  NVTE_CHECK(outputs_ != nullptr, "nvte_multi_compute_amax: outputs is NULL");

  // Convert, validate, collect into plain vectors.
  std::vector<Tensor *> inputs(num_tensors);
  std::vector<Tensor *> outputs(num_tensors);
  DType input_dtype;
  for (size_t i = 0; i < num_tensors; ++i) {
    inputs[i] = convertNVTETensorCheck(inputs_[i]);
    outputs[i] = convertNVTETensorCheck(outputs_[i]);
    const auto &inp = *inputs[i];
    auto &out = *outputs[i];
    NVTE_CHECK(inp.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
               "nvte_multi_compute_amax: input[", i,
               "] must be unquantized, got scaling_mode=", to_string(inp.scaling_mode));
    NVTE_CHECK(!is_fp8_dtype(inp.data.dtype),
               "nvte_multi_compute_amax: input[", i,
               "] must be unquantized, got dtype=", to_string(inp.data.dtype));
    if (i == 0) {
      input_dtype = inp.data.dtype;
    } else {
      NVTE_CHECK(inp.data.dtype == input_dtype,
                 "nvte_multi_compute_amax: all inputs must share dtype; input[0]=",
                 to_string(input_dtype), ", input[", i, "]=", to_string(inp.data.dtype));
    }
    NVTE_CHECK(out.scaling_mode == NVTE_DELAYED_TENSOR_SCALING ||
                   out.scaling_mode == NVTE_NVFP4_1D_SCALING,
               "nvte_multi_compute_amax: output[", i, "] must be FP8 per-tensor or NVFP4 1D");
    NVTE_CHECK(out.amax.dptr != nullptr || out.columnwise_amax.dptr != nullptr,
               "nvte_multi_compute_amax: output[", i, "] has no amax buffer");
  }

  const float *noop_ptr = nullptr;
  if (config_ != nullptr) {
    const QuantizationConfig *config_cpp = reinterpret_cast<const QuantizationConfig *>(config_);
    const NVTETensor noop = config_cpp->noop_tensor;
    noop_ptr = reinterpret_cast<float *>(
        (noop != nullptr ? convertNVTETensorCheck(noop)->data.dptr : nullptr));
  }

  // Chunk across kMaxTensorsPerBatch launches (single launch in the common 8-expert case).
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input_dtype, IType, {
    for (size_t start = 0; start < num_tensors; start += kMaxTensorsPerBatch) {
      const size_t count = std::min<size_t>(kMaxTensorsPerBatch, num_tensors - start);
      MultiAmaxArgs args = {};
      auto [max_numel, batch_align] = build_batch_args<IType>(inputs, outputs, start, count, args);
      launch_multi_amax_batch<IType>(args, max_numel, batch_align, noop_ptr, stream);
    }
  });  // NOLINT(*)
}

}  // anonymous namespace
}  // namespace transformer_engine

void nvte_multi_compute_amax(const NVTETensor *inputs, NVTETensor *outputs, size_t num_tensors,
                             const NVTEQuantizationConfig config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_compute_amax);
  transformer_engine::multi_compute_amax_impl(inputs, outputs, num_tensors, config, stream);
}
