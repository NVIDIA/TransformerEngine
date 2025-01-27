/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ptx.cuh
 *  \brief BW PTX
 */

#ifndef TRANSFORMER_ENGINE_PTX_CUH_
#define TRANSFORMER_ENGINE_PTX_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

namespace transformer_engine {
namespace ptx {

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, const uint32_t count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mbar_ptr), "r"(count) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
__device__ __forceinline__ void mbarrier_invalid(uint64_t* mbar) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(mbar_ptr) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(mbar_ptr) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, const uint32_t tx_count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(mbar_ptr), "r"(tx_count)
               : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// global -> shared::cluster
__device__ __forceinline__ void cp_async_bulk_tensor_1d_global_to_shared(
    uint64_t* dst_shmem, const uint64_t* src_global_ptr, const uint32_t size, uint64_t* mbar) {
  uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  // triggers async copy, i.e. the thread continues until wait() on mbarrier
  // barrier condition:
  // - leader must arrive (i.e. 1 thread as set above)
  // - TMA hardware substracts bytes from expect_tx counter, must reach zero
  asm volatile(
      "cp.async.bulk.shared::cta.global"
      ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst_shmem_ptr),
      "l"(src_global_ptr), "r"(size), "r"(mbar_ptr)
      : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// global -> shared::cluster
__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    uint64_t* dst_shmem, const uint64_t* tensor_map_ptr, const uint32_t offset_x,
    const uint32_t offset_y, uint64_t* mbar) {
  uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  // triggers async copy, i.e. the thread continues until wait() on mbarrier
  // barrier condition:
  // - leader must arrive (i.e. 1 thread as set above)
  // - TMA hardware substracts bytes from expect_tx counter, must reach zero
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_shmem_ptr),
      "l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(mbar_ptr)
      : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_1d_shared_to_global(uint64_t* dst_global_ptr,
                                                                         const uint64_t* src_shmem,
                                                                         const uint32_t size) {
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_global_ptr),
               "r"(src_shmem_ptr), "r"(size)
               : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_2d_shared_to_global(
    const uint64_t* tensor_map_ptr, const uint32_t offset_x, const uint32_t offset_y,
    uint64_t* src_shmem) {
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
                   tensor_map_ptr),
               "r"(offset_x), "r"(offset_y), "r"(src_shmem_ptr)
               : "memory");
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint32_t mbar_ptr, const uint32_t parity) {
  uint32_t waitComplete;
  asm volatile(
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(waitComplete)
      : "r"(mbar_ptr), "r"(parity)
      : "memory");
  return static_cast<bool>(waitComplete);
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, const uint32_t parity) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  while (!mbarrier_try_wait_parity(mbar_ptr, parity)) {
  }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
__device__ __forceinline__ void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
__device__ __forceinline__ void cp_async_bulk_wait_group() {
  asm volatile("cp.async.bulk.wait_group 0;");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
template <size_t W>
__device__ __forceinline__ void cp_async_bulk_wait_group_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;");
}

template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<0>() {
  asm volatile("cp.async.bulk.wait_group.read 0;");
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<1>() {
  asm volatile("cp.async.bulk.wait_group.read 1;");
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<2>() {
  asm volatile("cp.async.bulk.wait_group.read 2;");
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<4>() {
  asm volatile("cp.async.bulk.wait_group.read 4;");
}

// Proxy fence (bi-directional):
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async;"); }
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;");
}

#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

}  // namespace ptx
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PTX_CUH_
