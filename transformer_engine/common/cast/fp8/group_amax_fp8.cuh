/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_amax_fp8.cuh
 *  \brief CUDA kernels to compute per-group amax values for grouped FP8
 *         tensor scaling.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_AMAX_FP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_AMAX_FP8_CUH_

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../utils.cuh"
#include "../core/common.cuh"

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

// Flat vectorized amax kernel for imbalanced/varying shapes and uniform shapes alike.
//
// Grid: (num_blocks).
// Each block scans a flat chunk of elements within a specific tensor, or spanning across multiple tensors.
// Fully load-balanced!
template <int NVEC, typename InputType>
__launch_bounds__(GROUPED_AMAX_KERNEL_THREADS) __global__
    void grouped_amax_flat_kernel(const InputType *__restrict__ input, float *__restrict__ amax,
                                  const size_t num_tensors, const size_t first_logical_dim,
                                  const size_t last_logical_dim,
                                  const int64_t *__restrict__ offsets_ptr,
                                  const float *__restrict__ noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  // Dynamic shared memory layout:
  // s_offsets: int64_t[num_tensors + 1]
  extern __shared__ char s_mem[];
  int64_t *s_offsets = reinterpret_cast<int64_t *>(s_mem);

  const size_t tid = threadIdx.x;

  // 1. Copy or compute offsets to shared memory
  if (offsets_ptr != nullptr) {
    for (size_t i = tid; i <= num_tensors; i += blockDim.x) {
      s_offsets[i] = offsets_ptr[i];
    }
  } else {
    const size_t rows = first_logical_dim / num_tensors;
    const size_t numel_per_tensor = rows * last_logical_dim;
    for (size_t i = tid; i <= num_tensors; i += blockDim.x) {
      s_offsets[i] = i * numel_per_tensor;
    }
  }
  __syncthreads();

  const size_t total_active_elements = s_offsets[num_tensors];
  if (total_active_elements == 0) {
    return;
  }

  // Chunk based on the active elements and the grid size
  size_t block_chunk_size = (total_active_elements + gridDim.x - 1) / gridDim.x;
  // Align block_chunk_size to a multiple of NVEC to ensure perfect vector alignment
  block_chunk_size = DIVUP(block_chunk_size, static_cast<size_t>(NVEC)) * NVEC;
  //
  if (block_chunk_size < GROUPED_AMAX_MIN_ELTS_PER_BLOCK) {
    block_chunk_size = GROUPED_AMAX_MIN_ELTS_PER_BLOCK;
  }

  // Calculate this block's local work range
  const size_t start_elt = blockIdx.x * block_chunk_size;
  const size_t end_elt = start_elt + block_chunk_size < total_active_elements
                             ? start_elt + block_chunk_size
                             : total_active_elements;

  if (start_elt >= end_elt) {
    return;
  }

  constexpr int kWarps = GROUPED_AMAX_KERNEL_THREADS / THREADS_PER_WARP;
  __shared__ float s_block_amax[kWarps];

  using IVecT = Vec<InputType, NVEC>;

  size_t curr_elt = start_elt;
  size_t tensor_id = transformer_engine::dispatch::common::find_tensor_from_offsets(
      s_offsets, num_tensors, curr_elt);

  while (curr_elt < end_elt) {
    const size_t t_next_offset = s_offsets[tensor_id + 1];
    const size_t t_end = t_next_offset < end_elt ? t_next_offset : end_elt;

    const size_t numel = t_end - curr_elt;
    if (numel > 0) {
      const InputType *base = input + curr_elt;
      const size_t total_vecs = numel / NVEC;
      const size_t tail_start = total_vecs * NVEC;

      InputType thread_amax_val = 0.f;
      const bool aligned = (reinterpret_cast<uintptr_t>(base) % IVecT::BYTES) == 0;

      if (aligned) {
        InputType acc0 = 0.f;
        InputType acc1 = 0.f;
        InputType acc2 = 0.f;
        InputType acc3 = 0.f;
        size_t v = tid;
        // 4-way vectorized load and reduce
        for (; v + 3 * blockDim.x < total_vecs; v += 4 * blockDim.x) {
          IVecT vec0, vec1, vec2, vec3;
          vec0.load_from(base + v * NVEC);
          vec1.load_from(base + (v + blockDim.x) * NVEC);
          vec2.load_from(base + (v + 2 * blockDim.x) * NVEC);
          vec3.load_from(base + (v + 3 * blockDim.x) * NVEC);
#pragma unroll
          for (int i = 0; i < NVEC; ++i) {
            acc0 = max_val(acc0, abs_val(vec0.data.elt[i]));
            acc1 = max_val(acc1, abs_val(vec1.data.elt[i]));
            acc2 = max_val(acc2, abs_val(vec2.data.elt[i]));
            acc3 = max_val(acc3, abs_val(vec3.data.elt[i]));
          }
        }
        thread_amax_val = max_val(max_val(acc0, acc1), max_val(acc2, acc3));
        // 1-way vectorized load and reduce for the tail
        for (; v < total_vecs; v += blockDim.x) {
          IVecT vec;
          vec.load_from(base + v * NVEC);
#pragma unroll
          for (int i = 0; i < NVEC; ++i) {
            thread_amax_val = max_val(thread_amax_val, abs_val(vec.data.elt[i]));
          }
        }
      } else {
        for (size_t v = tid; v < total_vecs; v += blockDim.x) {
          const InputType *p = base + v * NVEC;
#pragma unroll
          for (int i = 0; i < NVEC; ++i) {
            thread_amax_val = max_val(thread_amax_val, abs_val(p[i]));
          }
        }
      }

      // Tail elements
      for (size_t i = tail_start + tid; i < numel; i += blockDim.x) {
        thread_amax_val = max_val(thread_amax_val, abs_val(base[i]));
      }

      // Warp-shuffle reduce within the block
      float thread_amax = static_cast<float>(thread_amax_val);
#pragma unroll
      for (int s = THREADS_PER_WARP / 2; s > 0; s >>= 1) {
        thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFFu, thread_amax, s));
      }

      const int warp_id = tid / THREADS_PER_WARP;
      const int lane = tid % THREADS_PER_WARP;
      if (lane == 0) {
        s_block_amax[warp_id] = thread_amax;
      }
      __syncthreads();
      // Reduce amax within the warp
      if (warp_id == 0) {
        float v = lane < kWarps ? s_block_amax[lane] : 0.0f;
#pragma unroll
        for (int s = kWarps / 2; s > 0; s >>= 1) {
          v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, s));
        }
        if (lane == 0) {
          atomicMaxFloat(&amax[tensor_id], v);
        }
      }
      __syncthreads();
      curr_elt = t_end;
    }

    if (curr_elt >= t_next_offset) {
      tensor_id++;
    }
  }
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
  (void)shape_rep;
  (void)first_dims_ptr;
  (void)last_dims_ptr;

  // Zero out the per-tensor amax buffer so atomicMaxFloat works.
  const size_t zero_blocks =
      (num_tensors + GROUPED_AMAX_KERNEL_THREADS - 1) / GROUPED_AMAX_KERNEL_THREADS;
  grouped_amax_zero_kernel<<<zero_blocks, GROUPED_AMAX_KERNEL_THREADS, 0, stream>>>(
      amax, num_tensors, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  constexpr int kVecBytes = 16;
  constexpr int kNvec = kVecBytes / sizeof(InputType);
  static_assert(kNvec >= 1, "Vector width must be at least 1");

  const int num_sms = ::transformer_engine::cuda::sm_count();
  const size_t grid_size = 8 * num_sms;
  const size_t shared_mem_bytes = (num_tensors + 1) * sizeof(int64_t);

  grouped_amax_flat_kernel<kNvec, InputType>
      <<<grid_size, GROUPED_AMAX_KERNEL_THREADS, shared_mem_bytes, stream>>>(
          input, amax, num_tensors, first_logical_dim, last_logical_dim, offsets_ptr, noop_ptr);

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_AMAX_FP8_CUH_
