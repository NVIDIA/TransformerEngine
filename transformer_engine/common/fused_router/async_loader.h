/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_ASYNC_LOADER_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_ASYNC_LOADER_H_

#include <cuda_pipeline.h>

#include <type_traits>

#include "../utils.cuh"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

// ============================================================================
// Persistent kernel grid size computation
// ============================================================================

// Compute a persistent grid size: min(total_blocks_needed, SMs * max_blocks_per_SM).
// `kernel_func` is a pointer to the __global__ function.
// `block_size` is kThreadsPerBlock.
// `shmem_bytes` is the dynamic shared memory per block.
// `total_blocks` is ceil(num_tokens / tokens_per_block).
template <typename KernelFunc>
inline size_t compute_persistent_grid(KernelFunc kernel_func, int block_size, size_t shmem_bytes,
                                      size_t total_blocks) {
  int blocks_per_sm = 0;
  NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_func,
                                                                block_size, shmem_bytes));
  if (blocks_per_sm <= 0) {
    return total_blocks;
  }
  int device_id;
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
  int num_sms;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));

  size_t max_resident = static_cast<size_t>(num_sms) * blocks_per_sm;
  return (total_blocks < max_resident) ? total_blocks : max_resident;
}

// ============================================================================
// Occupancy-aware double-buffer decision
// ============================================================================

// Decide whether to use 1 or 2 buffers based on shmem budget.
// `single_buf_shmem` is the per-buffer shmem for the async-loaded data.
// `other_shmem_bytes` is shmem for everything else (scratch, work buffers).
// Returns 1 or 2.  Ensures at least kMinBlocksPerSM blocks can co-reside.
inline int choose_num_buffers(size_t single_buf_shmem, size_t other_shmem_bytes) {
  constexpr int kMinBlocksPerSM = 4;

  size_t total_single = single_buf_shmem + other_shmem_bytes;
  size_t total_double = 2 * single_buf_shmem + other_shmem_bytes;

  int device_id;
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
  int max_smem;
  NVTE_CHECK_CUDA(
      cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));

  int blocks_double = (total_double > 0) ? static_cast<int>(max_smem / total_double) : 0;
  int blocks_single = (total_single > 0) ? static_cast<int>(max_smem / total_single) : 0;

  if (blocks_double >= kMinBlocksPerSM) return 2;
  if (blocks_single >= kMinBlocksPerSM) return 1;
  return (blocks_double >= blocks_single) ? 2 : 1;
}

// ============================================================================
// Vectorized global store/fill helpers (using Vec<> from utils.cuh)
// ============================================================================

template <typename T>
struct VecTraits {
  static constexpr int kVecSize = (sizeof(T) <= 16) ? (16 / sizeof(T)) : 1;
};

// Vectorized store: write `count` elements from shmem/registers to global memory.
template <typename T>
__device__ inline void vec_store_global(T *__restrict__ dst, const T *__restrict__ src, int count,
                                        int lane_id) {
  constexpr int kVecSize = VecTraits<T>::kVecSize;
  using VecType = typename BytesToType<sizeof(T) * kVecSize>::Type;

  bool aligned = (reinterpret_cast<uint64_t>(dst) % (sizeof(T) * kVecSize) == 0);
  int aligned_count = (count / kVecSize) * kVecSize;

  if (aligned && aligned_count > 0) {
    int vec_count = aligned_count / kVecSize;
    for (int vi = lane_id; vi < vec_count; vi += kThreadsPerWarp) {
      VecType v;
      T *v_elts = reinterpret_cast<T *>(&v);
#pragma unroll
      for (int e = 0; e < kVecSize; e++) {
        v_elts[e] = src[vi * kVecSize + e];
      }
      reinterpret_cast<VecType *>(dst)[vi] = v;
    }
    for (int i = aligned_count + lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = src[i];
    }
  } else {
    for (int i = lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = src[i];
    }
  }
}

// Vectorized fill: write `val` to `count` elements of global memory.
template <typename T>
__device__ inline void vec_fill_global(T *__restrict__ dst, T val, int count, int lane_id) {
  constexpr int kVecSize = VecTraits<T>::kVecSize;
  using VecType = typename BytesToType<sizeof(T) * kVecSize>::Type;

  bool aligned = (reinterpret_cast<uint64_t>(dst) % (sizeof(T) * kVecSize) == 0);
  int aligned_count = (count / kVecSize) * kVecSize;

  if (aligned && aligned_count > 0) {
    VecType v;
    T *v_elts = reinterpret_cast<T *>(&v);
#pragma unroll
    for (int e = 0; e < kVecSize; e++) {
      v_elts[e] = val;
    }
    int vec_count = aligned_count / kVecSize;
    for (int vi = lane_id; vi < vec_count; vi += kThreadsPerWarp) {
      reinterpret_cast<VecType *>(dst)[vi] = v;
    }
    for (int i = aligned_count + lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = val;
    }
  } else {
    for (int i = lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = val;
    }
  }
}

// ============================================================================
// cp.async wrappers — use hardware async copy on sm_80+, no-op on older archs.
// Always defined so callers don't need #if guards.
// ============================================================================

__device__ __forceinline__ void cp_async_16B(void *__restrict__ dst, const void *__restrict__ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __pipeline_memcpy_async(dst, src, 16);
#else
  // Scalar fallback — callers must not rely on this being async.
  *static_cast<int4 *>(dst) = *static_cast<const int4 *>(src);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __pipeline_commit();
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __pipeline_wait_prior(0);
#endif
}

// ============================================================================
// RawAsyncLoader<T> — double-buffered loader storing data in original type
//
// Enables cp.async for ALL data types (bf16, fp16, fp32) since no type
// conversion is needed during the copy.  The kernel reads from shmem and
// casts to CompType during compute.
// ============================================================================

template <typename T>
class RawAsyncLoader {
 public:
  // Shmem size calculation (usable from both host and device).
  static __host__ __device__ inline size_t shmem_bytes(int count, int num_warps, int num_buffers) {
    return static_cast<size_t>(num_buffers) * count * num_warps * sizeof(T);
  }

  // Device-side construction.
  __device__ RawAsyncLoader(T *buf_base, int warp_id, int count, int num_warps, int num_buffers)
      : phase_(0), double_buf_(num_buffers == 2) {
    int per_buffer = count * num_warps;
    buf_[0] = buf_base + warp_id * count;
    buf_[1] = (num_buffers == 2) ? buf_base + per_buffer + warp_id * count : buf_[0];
  }

  __device__ __forceinline__ T *current_buf() { return buf_[phase_]; }
  __device__ __forceinline__ T *next_buf() { return buf_[phase_ ^ 1]; }
  __device__ __forceinline__ void flip() {
    if (double_buf_) phase_ ^= 1;
  }

  // Async load into the NEXT buffer (for prefetching).
  __device__ void start_load(const T *__restrict__ src, int count, int lane_id) {
    raw_load(src, next_buf(), count, lane_id);
  }

  // Load into the CURRENT buffer (for the first load before the main loop).
  __device__ void load_current(const T *__restrict__ src, int count, int lane_id) {
    raw_load(src, current_buf(), count, lane_id);
  }

  // Wait for pending async loads to complete.
  __device__ __forceinline__ void wait() {
    cp_async_wait_all();
    __syncwarp();
  }

 private:
  T *buf_[2];
  int phase_;
  bool double_buf_;

  // Raw copy: global → shmem, no type conversion.
  // Uses 16-byte vectorised copies (cp.async on sm_80+, int4 on older archs)
  // when both pointers are 16-byte aligned, with a scalar tail / fallback.
  __device__ void raw_load(const T *__restrict__ src, T *__restrict__ dst, int count, int lane_id) {
    constexpr int kBytesPerCopy = 16;
    constexpr int kEltsPerCopy = kBytesPerCopy / sizeof(T);

    bool src_aligned = (reinterpret_cast<uint64_t>(src) % kBytesPerCopy == 0);
    bool dst_aligned = (reinterpret_cast<uint64_t>(dst) % kBytesPerCopy == 0);
    int aligned_count = (count / kEltsPerCopy) * kEltsPerCopy;

    if (src_aligned && dst_aligned && aligned_count > 0) {
      int vec_count = aligned_count / kEltsPerCopy;
      for (int vi = lane_id; vi < vec_count; vi += kThreadsPerWarp) {
        cp_async_16B(dst + vi * kEltsPerCopy, src + vi * kEltsPerCopy);
      }
      for (int i = aligned_count + lane_id; i < count; i += kThreadsPerWarp) {
        dst[i] = src[i];
      }
      cp_async_commit();
    } else {
      for (int i = lane_id; i < count; i += kThreadsPerWarp) {
        dst[i] = src[i];
      }
      cp_async_commit();  // No-op on sm_70; matches wait() expectation on sm_80+.
    }
  }
};

}  // namespace fused_router
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_FUSED_ROUTER_ASYNC_LOADER_H_
