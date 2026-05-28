/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_amax_fp8.cuh
 *  \brief CUDA kernels to compute per-group amax values for grouped FP8
 *         tensor scaling.
 *
 *  These kernels are logically part of the grouped FP8 quantization stack
 *  (alongside group_quantize_fp8.cuh), but they are kept in a separate
 *  header so that only the single translation unit that actually launches
 *  them (recipe/current_scaling.cu) takes the compile-time hit. Pulling
 *  them into group_quantize_fp8.cuh forces every TU that transitively
 *  includes the quantize dispatcher (cast.cu, the activation .cu files,
 *  etc.) to recompile whenever this code changes, which defeats ccache.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_AMAX_FP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_AMAX_FP8_CUH_

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>

#include "../../common.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_amax_kernel {

constexpr int GROUPED_AMAX_KERNEL_THREADS = 256;
// Target block-per-tensor multiplier so that, even with few groups, we cover
// enough SMs on H100/GB200-class GPUs (132 SMs) to be HBM-bandwidth bound.
// blocks_per_tensor is bounded by both work and a hard cap to avoid launching
// far more blocks than there is work for.
constexpr int GROUPED_AMAX_BLOCKS_PER_TENSOR_CAP = 64;
constexpr size_t GROUPED_AMAX_MIN_ELTS_PER_BLOCK = 8 * 1024;  // ~16KB of bf16

// Zero per-tensor amax buffer so the main kernel can use atomicMax updates.
__launch_bounds__(GROUPED_AMAX_KERNEL_THREADS) __global__
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
// tensor) and means we never read past the active region into the unused
// tail of logical_shape (where logical_first_dim >= sum(first_dims)).
template <int NVEC, typename InputType, ShapeRepresentation SHAPE_REP>
__launch_bounds__(GROUPED_AMAX_KERNEL_THREADS) __global__
    void grouped_amax_kernel(const InputType *__restrict__ input, float *__restrict__ amax,
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
  constexpr int kWarps = GROUPED_AMAX_KERNEL_THREADS / THREADS_PER_WARP;
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

// Pick blocks-per-tensor so each block has at least ~GROUPED_AMAX_MIN_ELTS_PER_BLOCK elements
// of work. Capped at GROUPED_AMAX_BLOCKS_PER_TENSOR_CAP to avoid launching way more blocks
// than the GPU can resident.
inline size_t choose_grouped_amax_blocks_per_tensor(size_t max_elts_per_tensor) {
  if (max_elts_per_tensor == 0) return 1;
  size_t blocks = (max_elts_per_tensor + GROUPED_AMAX_MIN_ELTS_PER_BLOCK - 1) /
                  GROUPED_AMAX_MIN_ELTS_PER_BLOCK;
  if (blocks > static_cast<size_t>(GROUPED_AMAX_BLOCKS_PER_TENSOR_CAP)) {
    blocks = GROUPED_AMAX_BLOCKS_PER_TENSOR_CAP;
  }
  if (blocks == 0) blocks = 1;
  return blocks;
}

}  // namespace group_amax_kernel

template <typename InputType>
void launch_grouped_amax_kernel(const InputType *input, float *amax, const size_t num_tensors,
                                const size_t first_logical_dim, const size_t last_logical_dim,
                                const ShapeRepresentation shape_rep, const int64_t *offsets_ptr,
                                const int64_t *first_dims_ptr, const int64_t *last_dims_ptr,
                                const float *noop_ptr, cudaStream_t stream) {
  using namespace group_amax_kernel;
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
      (num_tensors + GROUPED_AMAX_KERNEL_THREADS - 1) / GROUPED_AMAX_KERNEL_THREADS;
  grouped_amax_zero_kernel<<<zero_blocks, GROUPED_AMAX_KERNEL_THREADS, 0, stream>>>(
      amax, num_tensors, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  constexpr int kVecBytes = 16;
  constexpr int kNvec = kVecBytes / sizeof(InputType);
  static_assert(kNvec >= 1, "Vector width must be at least 1");

  const size_t blocks_per_tensor = choose_grouped_amax_blocks_per_tensor(max_elts_per_tensor);
  const dim3 grid(blocks_per_tensor, num_tensors);
  const dim3 block(GROUPED_AMAX_KERNEL_THREADS);

  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
      grouped_amax_kernel<kNvec, InputType, ShapeRepresentation::SAME_BOTH_DIMS>
          <<<grid, block, 0, stream>>>(input, amax, num_tensors, first_logical_dim,
                                       last_logical_dim, offsets_ptr, noop_ptr);
      break;
    case ShapeRepresentation::VARYING_FIRST_DIM:
      grouped_amax_kernel<kNvec, InputType, ShapeRepresentation::VARYING_FIRST_DIM>
          <<<grid, block, 0, stream>>>(input, amax, num_tensors, first_logical_dim,
                                       last_logical_dim, offsets_ptr, noop_ptr);
      break;
    case ShapeRepresentation::VARYING_LAST_DIM:
      grouped_amax_kernel<kNvec, InputType, ShapeRepresentation::VARYING_LAST_DIM>
          <<<grid, block, 0, stream>>>(input, amax, num_tensors, first_logical_dim,
                                       last_logical_dim, offsets_ptr, noop_ptr);
      break;
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      grouped_amax_kernel<kNvec, InputType, ShapeRepresentation::VARYING_BOTH_DIMS>
          <<<grid, block, 0, stream>>>(input, amax, num_tensors, first_logical_dim,
                                       last_logical_dim, offsets_ptr, noop_ptr);
      break;
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_AMAX_FP8_CUH_
