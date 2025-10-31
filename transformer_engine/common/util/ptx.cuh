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

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif  // CUDA_VERSION >= 12080

#include "common/utils.cuh"

namespace transformer_engine {

namespace ptx {

template <int N>
struct ArchSpecific {
  constexpr static int id = N * 10;

  template <int CurrentArch, int ArchSpecific, int FamilySpecific>
  constexpr static bool compatible() {
    if constexpr (CurrentArch == id) {
      static_assert(ArchSpecific == CurrentArch,
                    "Compiled for the generic architecture, while utilizing arch-specific "
                    "features. Please compile for smXXXa architecture instead of smXXX "
                    "architecture.");
      return true;
    } else {
      return false;
    }
  }
};

template <int N>
struct FamilySpecific {
  constexpr static int id = N * 10;

  template <int CurrentArch, int ArchSpecific, int FamilySpecific>
  constexpr static bool compatible() {
    if constexpr ((CurrentArch / 100) == (id / 100)) {
      static_assert(FamilySpecific == CurrentArch,
                    "Compiled for the generic architecture, while utilizing family-specific "
                    "features. Please compile for smXXXf architecture instead of smXXX "
                    "architecture.");
      return true;
    } else {
      return false;
    }
  }
};

template <int Arch, int ArchSpecific, int FamilySpecific, class T, class... U>
constexpr bool is_supported_arch() {
  if constexpr (T::template compatible<Arch, ArchSpecific, FamilySpecific>()) {
    return true;
  } else if constexpr (sizeof...(U) != 0) {
    return is_supported_arch<Arch, ArchSpecific, FamilySpecific, U...>();
  } else {
    return false;
  }
}

#if CUDA_VERSION < 12090
#if __CUDA_ARCH_HAS_FEATURE__(SM90_ALL)
#define __CUDA_ARCH_SPECIFIC__ 900
#define __CUDA_ARCH_FAMILY_SPECIFIC__ 900
#endif
#if __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
#define __CUDA_ARCH_SPECIFIC__ 1000
#define __CUDA_ARCH_FAMILY_SPECIFIC__ 1000
#endif
#if __CUDA_ARCH_HAS_FEATURE__(SM101_ALL)
#define __CUDA_ARCH_SPECIFIC__ 1010
#define __CUDA_ARCH_FAMILY_SPECIFIC__ 1010
#endif
#if __CUDA_ARCH_HAS_FEATURE__(SM120_ALL)
#define __CUDA_ARCH_SPECIFIC__ 1200
#define __CUDA_ARCH_FAMILY_SPECIFIC__ 1200
#endif
#endif

#ifdef __CUDA_ARCH__
#define __NVTE_CURRENT_ARCH__ constexpr int current_arch = __CUDA_ARCH__;
#else
#define __NVTE_CURRENT_ARCH__ constexpr int current_arch = 0;
#endif

#ifdef __CUDA_ARCH_SPECIFIC__
#define __NVTE_ARCH_SPECIFIC__ constexpr int ArchSpecific = __CUDA_ARCH_SPECIFIC__;
#else
#define __NVTE_ARCH_SPECIFIC__ constexpr int ArchSpecific = 0;
#endif

#ifdef __CUDA_ARCH_FAMILY_SPECIFIC__
#define __NVTE_ARCH_FAMILY_SPECIFIC__ constexpr int FamilySpecific = __CUDA_ARCH_FAMILY_SPECIFIC__;
#else
#define __NVTE_ARCH_FAMILY_SPECIFIC__ constexpr int FamilySpecific = 0;
#endif

#define NVTE_CUDA_ARCH_MATCHES(...)                                                               \
  [&] {                                                                                           \
    __NVTE_CURRENT_ARCH__                                                                         \
    __NVTE_ARCH_SPECIFIC__                                                                        \
    __NVTE_ARCH_FAMILY_SPECIFIC__                                                                 \
    return transformer_engine::ptx::is_supported_arch<current_arch, ArchSpecific, FamilySpecific, \
                                                      __VA_ARGS__>();                             \
  }();

#define ARCH_BLACKWELL_FAMILY                                                \
  NVTE_CUDA_ARCH_MATCHES(ptx::FamilySpecific<100>, ptx::FamilySpecific<110>, \
                         ptx::FamilySpecific<120>)
#define ARCH_HAS_STOCHASTIC_ROUNDING \
  NVTE_CUDA_ARCH_MATCHES(ptx::ArchSpecific<100>, ptx::ArchSpecific<103>)

__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, const uint32_t count) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mbar_ptr), "r"(count) : "memory");
#else
  NVTE_DEVICE_ERROR("mbarrier_init is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mbarrier_invalid(uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(mbar_ptr) : "memory");
#else
  NVTE_DEVICE_ERROR("mbarrier_invalid is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(mbar_ptr) : "memory");
#else
  NVTE_DEVICE_ERROR("mbarrier_arrive is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mbarrier_arrive_relaxed_cta_shared_cta(uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0], 1;" ::"r"(mbar_ptr));
#else
  NVTE_DEVICE_ERROR("mbarrier_arrive_relaxed_cta_shared_cta is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mbarrier_arrive_release_cta_shared_cta(uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 1;" ::"r"(mbar_ptr));
#else
  NVTE_DEVICE_ERROR("mbarrier_arrive_release_cta_shared_cta is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *mbar, const uint32_t tx_count) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(mbar_ptr), "r"(tx_count)
               : "memory");
#else
  NVTE_DEVICE_ERROR("mbarrier_arrive_expect_tx is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cta_relaxed_shared_cta(
    uint64_t *mbar, const uint32_t tx_count) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_ptr),
               "r"(tx_count));
#else
  NVTE_DEVICE_ERROR(
      "mbarrier_arrive_expect_tx_cta_relaxed_shared_cta is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("fence.mbarrier_init.release.cluster;");
#else
  NVTE_DEVICE_ERROR("fence_mbarrier_init_release_cluster is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void cp_async_bulk_tensor_1d_global_to_shared(
    uint64_t *dst_shmem, const uint64_t *src_global_ptr, const uint32_t size, uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
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
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_tensor_1d_global_to_shared is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    uint64_t *dst_shmem, const uint64_t *tensor_map_ptr, const uint32_t offset_x,
    const uint32_t offset_y, uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
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
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_tensor_2d_global_to_shared is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint32_t mbar_ptr, const uint32_t parity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
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
#else
  NVTE_DEVICE_ERROR("mbarrier_try_wait_parity is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return true;
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t *mbar, const uint32_t parity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  while (!mbarrier_try_wait_parity(mbar_ptr, parity)) {
  }
#else
  NVTE_DEVICE_ERROR("mbarrier_wait_parity is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mbarrier_wait_parity_acquire_cta_shared_cta(uint64_t *mbar,
                                                                            uint32_t phase_parity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
      "{\n\t"
      ".reg .b64 r1; \n\t"
      ".reg .pred waitComplete; \n\t"  // predicate representing if barrier condition is met
      "WAIT: \n\t"                     // loop around barrier wait
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  waitComplete, [%0], %1; \n\t"
      "@waitComplete bra DONE; \n\t"  // mbarrier conditions are met
      "bra WAIT; \n\t"                // just a time-out, try again
      "DONE: \n\t"
      "}\n\t"
      :
      : "r"(mbar_ptr), "r"(phase_parity)
      : "memory");
#else
  NVTE_DEVICE_ERROR("mbarrier_wait_parity_acquire_cta_shared_cta is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mbarrier_wait_parity_relaxed_cta_shared_cta(uint64_t *mbar,
                                                                            uint32_t phase_parity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
      "{\n\t"
      ".reg .b64 r1; \n\t"
      ".reg .pred waitComplete; \n\t"  // predicate representing if barrier condition is met
      "WAIT: \n\t"                     // loop around barrier wait
      "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64  waitComplete, [%0], %1; \n\t"
      "@waitComplete bra DONE; \n\t"  // mbarrier conditions are met
      "bra WAIT; \n\t"                // just a time-out, try again
      "DONE: \n\t"
      "}\n\t"
      :
      : "r"(mbar_ptr), "r"(phase_parity)
      : "memory");
#else
  NVTE_DEVICE_ERROR("mbarrier_wait_parity_relaxed_cta_shared_cta is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void
clusterlaunchcontrol_try_cancel_async_shared_cta_mbarrier_complete_tx_bytes(
    uint64_t *mbar, __uint128_t *response_data_ptr) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
    uint32_t workID_response = __cvta_generic_to_shared(response_data_ptr);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::"
        "all.b128 "
        "[%0], [%1];" ::"r"(workID_response),
        "r"(mbar_ptr));
  } else {
    NVTE_DEVICE_ERROR(
        "Cluster Launch Control PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
}

__device__ __forceinline__ void get_cancelled_cta_2D_id(__uint128_t *response_data_ptr,
                                                        int32_t &ctaid_X, int32_t &ctaid_Y) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    uint32_t workID_response = __cvta_generic_to_shared(response_data_ptr);
    asm volatile(
        "{\n\t"
        ".reg .s32 x_ctaid; \n\t"
        ".reg .s32 y_ctaid; \n\t"
        "mov .s32 x_ctaid, -1; \n\t"
        "mov .s32 y_ctaid, -1; \n\t"
        ".reg.b128 try_cancel_response; \n\t"
        "ld.shared.b128 try_cancel_response, [%2]; \n\t"
        ".reg .pred P1; \n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P1, try_cancel_response; \n\t"
        "@P1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x_ctaid, y_ctaid, _, "
        "_}, try_cancel_response; \n\t"
        "mov .s32 %0, x_ctaid; \n\t"
        "mov .s32 %1, y_ctaid; \n\t"
        "}\n\t"
        : "=r"(ctaid_X), "=r"(ctaid_Y)
        : "r"(workID_response)
        : "memory");
  } else {
    NVTE_DEVICE_ERROR(
        "Cluster Launch Control PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
}

constexpr uint32_t FP32_MANTISSA_BITS = 23;
constexpr uint32_t FP32_EXPONENT_BIAS = 127;

__device__ __forceinline__ float exp2f_rcp(e8m0_t biased_exp) {
  return (biased_exp == 0) ? 1
                           : __int_as_float((254 - biased_exp)
                                            << FP32_MANTISSA_BITS);  // 127 - (biased_exp - 127)
}

__device__ __forceinline__ float exp2f(e8m0_t biased_exp) {
  return __int_as_float(biased_exp << FP32_MANTISSA_BITS);
}

__device__ __forceinline__ e8m0_t float_to_e8m0(float val) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    uint16_t out;
    asm volatile(
        "{\n"
        "cvt.rp.satfinite.ue8m0x2.f32  %0, 0.0, %1;\n"
        "}"
        : "=h"(out)
        : "f"(val));
    return *reinterpret_cast<e8m0_t *>(&out);
  } else {
    // TODO: nan/inf needs to be set for any value
    // of nan/inf in input not just amax.
    if (isnan(val)) {
      return 0xFF;
    }
    if (isinf(val)) {
      return 0xFE;
    }
    if (val == 0.0f) {
      return 0x00;
    }
    uint32_t val_u32 = *reinterpret_cast<uint32_t *>(&val);
    e8m0_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
    uint32_t mantissa = val_u32 & 0x7FFFFF;
    // Round up exponent and deal with satfinite.
    if ((mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)) {
      ++exponent;
    }
    return exponent;
  }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_1d_shared_to_global(uint64_t *dst_global_ptr,
                                                                         const uint64_t *src_shmem,
                                                                         const uint32_t size) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_global_ptr),
               "r"(src_shmem_ptr), "r"(size)
               : "memory");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_tensor_1d_shared_to_global is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_2d_shared_to_global(
    const uint64_t *tensor_map_ptr, const uint32_t offset_x, const uint32_t offset_y,
    uint64_t *src_shmem) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
                   tensor_map_ptr),
               "r"(offset_x), "r"(offset_y), "r"(src_shmem_ptr)
               : "memory");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_tensor_2d_shared_to_global is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

__device__ __forceinline__ void cp_async_bulk_wait_group() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group 0;");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_wait_group is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

template <size_t W>
__device__ __forceinline__ void cp_async_bulk_wait_group_read() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 0;");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_wait_group_read is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<0>() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 0;");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_wait_group_read is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<1>() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 1;");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_wait_group_read is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<2>() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 2;");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_wait_group_read is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<4>() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 4;");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_wait_group_read is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

__device__ __forceinline__ void cp_async_bulk_commit_group() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.commit_group;");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_commit_group is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// Proxy fence (bi-directional):
__device__ __forceinline__ void fence_proxy_async() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.async;");
#else
  NVTE_DEVICE_ERROR("fence_proxy_async is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.async.shared::cta;");
#else
  NVTE_DEVICE_ERROR("fence_proxy_async_shared_cta is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

template <typename T>
struct alignas(2 * sizeof(T)) FPx2 {
  T x;
  T y;
};

template <typename T>
struct FPx4 {
  T x1;
  T x2;
  T x3;
  T x4;
};

template <typename T>
struct Type2x {};

template <>
struct Type2x<float> {
  using type = float2;
};

template <>
struct Type2x<bf16> {
  using type = __nv_bfloat162;
};

template <>
struct Type2x<fp16> {
  using type = __half2;
};

using floatx2 = FPx2<float>;
using bf16x2 = FPx2<bf16>;
using fp16x2 = FPx2<fp16>;
using fp8e4m3x2 = FPx2<fp8e4m3>;
using fp8e5m2x2 = FPx2<fp8e5m2>;

using floatx4 = FPx4<float>;
using bf16x4 = FPx4<bf16>;
using fp16x4 = FPx4<fp16>;
using fp8e4m3x4 = FPx4<fp8e4m3>;
using fp8e5m2x4 = FPx4<fp8e5m2>;

static_assert(sizeof(floatx2) == 8);
static_assert(sizeof(bf16x2) == 4);
static_assert(sizeof(fp16x2) == 4);
static_assert(sizeof(fp8e4m3x2) == 2);
static_assert(sizeof(fp8e5m2x2) == 2);

#if FP4_TYPE_SUPPORTED
using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
static_assert(sizeof(fp4e2m1x2) == 1);
static_assert(sizeof(fp4e2m1x4) == 2);

// When converting to .e2m1x2 data formats, the destination operand d has .b8 type.
// When converting two .f32 inputs to .e2m1x2, each input is converted to the specified format,
// and the converted values are packed in the destination operand d such that the value
// converted from input a is stored in the upper 4 bits of d and the value converted
// from input b is stored in the lower 4 bits of d.

// SIMD like "Fused" cast + multiplication (x4)
template <typename Tx2>
__device__ __forceinline__ void mul_cvt_4x(fp4e2m1x4 &out, const Tx2 &in01, const Tx2 &in23,
                                           const float scale) {
  const float x0 = static_cast<float>(in01.x) * scale;
  const float x1 = static_cast<float>(in01.y) * scale;
  const float x2 = static_cast<float>(in23.x) * scale;
  const float x3 = static_cast<float>(in23.y) * scale;
  out = fp4e2m1x4(make_float4(x0, x1, x2, x3));
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b16 v0_bf16; \n\t"
        ".reg.b16 v1_bf16; \n\t"
        ".reg.b16 v2_bf16; \n\t"
        ".reg.b16 v3_bf16; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16} , %1; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %2; \n\t"  // mind the shuffled elements order
        "mul.f32x2 v23, v23, %2; \n\t"  // mind the shuffled elements order
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %3; \n\t"  // mind the shuffled elements order
        "}"
        : "=h"(out_4x)
        : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)), "r"(rbits));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_rn(const uint64_t in_4x,
                                                                    const float2 scale,
                                                                    const uint32_t rbits) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  uint32_t out_4x = 0;  // Only need 16 bit. Using 32 bit container for packing.
  if constexpr (is_blackwell) {
    // NOTE: rbits unused for rn.
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b16 v0_bf16; \n\t"
        ".reg.b16 v1_bf16; \n\t"
        ".reg.b16 v2_bf16; \n\t"
        ".reg.b16 v3_bf16; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        ".reg.b8 f0; \n\t"
        ".reg.b8 f1; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16} , %1; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %2; \n\t"  // mind the shuffled elements order
        "mul.f32x2 v23, v23, %2; \n\t"  // mind the shuffled elements order
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
        "mov.b32 %0, {f0, f1, f0, f1};\n\t"
        "}"
        : "=r"(out_4x)
        : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x(const uint64_t in_4x,
                                                            const float2 scale,
                                                            const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(in_4x, scale, rbits);
  } else {
    return mul_cvt_bf16_to_fp4_4x_with_rn(in_4x, scale, rbits);
  }
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        "mov.b64 {v0, v1} , %1; \n\t"
        "mov.b64 {v2, v3} , %2; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %3; \n\t"  // mind the shuffled elements order
        "mul.f32x2 v23, v23, %3; \n\t"  // mind the shuffled elements order
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %4; \n\t"  // mind the shuffled elements order
        "}"
        : "=h"(out_4x)
        : "l"(reinterpret_cast<const uint64_t &>(in01)),
          "l"(reinterpret_cast<const uint64_t &>(in23)),
          "l"(reinterpret_cast<const uint64_t &>(scale)), "r"(rbits));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_rn(const float2 in01,
                                                                    const float2 in23,
                                                                    const float2 scale,
                                                                    const uint32_t rbits) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  uint32_t out_4x = 0;  // Only need 16 bit. Using 32 bit container for packing.
  if constexpr (is_blackwell) {
    // NOTE: rbits unused for rn.
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        ".reg.b8 f0; \n\t"
        ".reg.b8 f1; \n\t"
        "mov.b64 {v0, v1} , %1; \n\t"
        "mov.b64 {v2, v3} , %2; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %3; \n\t"  // mind the shuffled elements order
        "mul.f32x2 v23, v23, %3; \n\t"  // mind the shuffled elements order
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
        "mov.b32 %0, {f0, f1, f0, f1};\n\t"
        "}"
        : "=r"(out_4x)
        : "l"(reinterpret_cast<const uint64_t &>(in01)),
          "l"(reinterpret_cast<const uint64_t &>(in23)),
          "l"(reinterpret_cast<const uint64_t &>(scale)));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x(const float2 in01, const float2 in23,
                                                            const float2 scale,
                                                            const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(in01, in23, scale, rbits);
  } else {
    return mul_cvt_fp32_to_fp4_4x_with_rn(in01, in23, scale, rbits);
  }
}
#endif  // FP4_TYPE_SUPPORTED

// SIMD like "Fused" cast + multiplication (x2)
__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const floatx2 &in,
                                           const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      "mul.f32x2 val_pair, %1, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair; \n\t"
      "cvt.rn.satfinite.e4m3x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "l"(reinterpret_cast<const uint64_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR("mul_cvt_2x is only supported on SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const floatx2 &in,
                                           const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      "mul.f32x2 val_pair, %1, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair; \n\t"
      "cvt.rn.satfinite.e5m2x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "l"(reinterpret_cast<const uint64_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR("mul_cvt_2x is only supported on SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const bf16x2 &in, const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair_before; \n\t"
      ".reg.b64 val_pair_after; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      ".reg.b16 val1_bf16; \n\t"
      ".reg.b16 val2_bf16; \n\t"
      "mov.b32 {val1_bf16, val2_bf16} , %1; \n\t"
      "cvt.f32.bf16 val1, val1_bf16; \n\t"
      "cvt.f32.bf16 val2, val2_bf16; \n\t"
      "mov.b64 val_pair_before, {val1,val2}; \n\t"
      "mul.f32x2 val_pair_after, val_pair_before, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair_after; \n\t"
      "cvt.rn.satfinite.e4m3x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "r"(reinterpret_cast<const uint32_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR("mul_cvt_2x is only supported on SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const bf16x2 &in, const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair_before; \n\t"
      ".reg.b64 val_pair_after; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      ".reg.b16 val1_bf16; \n\t"
      ".reg.b16 val2_bf16; \n\t"
      "mov.b32 {val1_bf16, val2_bf16} , %1; \n\t"
      "cvt.f32.bf16 val1, val1_bf16; \n\t"
      "cvt.f32.bf16 val2, val2_bf16; \n\t"
      "mov.b64 val_pair_before, {val1,val2}; \n\t"
      "mul.f32x2 val_pair_after, val_pair_before, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair_after; \n\t"
      "cvt.rn.satfinite.e5m2x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "r"(reinterpret_cast<const uint32_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR("mul_cvt_2x is only supported on SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const fp16x2 &in, const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair_before; \n\t"
      ".reg.b64 val_pair_after; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      ".reg.b16 val1_fp16; \n\t"
      ".reg.b16 val2_fp16; \n\t"
      "mov.b32 {val1_fp16, val2_fp16} , %1; \n\t"
      "cvt.f32.f16 val1, val1_fp16; \n\t"
      "cvt.f32.f16 val2, val2_fp16; \n\t"
      "mov.b64 val_pair_before, {val1,val2}; \n\t"
      "mul.f32x2 val_pair_after, val_pair_before, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair_after; \n\t"
      "cvt.rn.satfinite.e4m3x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "r"(reinterpret_cast<const uint32_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR("mul_cvt_2x is only supported on SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const fp16x2 &in, const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair_before; \n\t"
      ".reg.b64 val_pair_after; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      ".reg.b16 val1_fp16; \n\t"
      ".reg.b16 val2_fp16; \n\t"
      "mov.b32 {val1_fp16, val2_fp16} , %1; \n\t"
      "cvt.f32.f16 val1, val1_fp16; \n\t"
      "cvt.f32.f16 val2, val2_fp16; \n\t"
      "mov.b64 val_pair_before, {val1,val2}; \n\t"
      "mul.f32x2 val_pair_after, val_pair_before, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair_after; \n\t"
      "cvt.rn.satfinite.e5m2x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "r"(reinterpret_cast<const uint32_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR("mul_cvt_2x is only supported on SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void abs_max_2x(bf16x2 &dst, const bf16x2 &p1, const bf16x2 &p2) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;"
               : "=r"(reinterpret_cast<uint32_t &>(dst))
               : "r"(reinterpret_cast<const uint32_t &>(p1)),
                 "r"(reinterpret_cast<const uint32_t &>(p2)));
#else
  NVTE_DEVICE_ERROR("abs_max_2x is only supported on SM 8.9+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

__device__ __forceinline__ void abs_max_2x(fp16x2 &dst, const fp16x2 &p1, const fp16x2 &p2) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  asm volatile("max.xorsign.abs.f16x2 %0, %1, %2;"
               : "=r"(reinterpret_cast<uint32_t &>(dst))
               : "r"(reinterpret_cast<const uint32_t &>(p1)),
                 "r"(reinterpret_cast<const uint32_t &>(p2)));
#else
  NVTE_DEVICE_ERROR("abs_max_2x is only supported on SM 8.9+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

// Loads a pair of BF16/FP16 values from shared memory state space and converts them into float2
template <typename IType2>
__device__ __forceinline__ float2 ld_shared_cvt_f32x2(const IType2 *src_smem) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t src_smem_ptr = __cvta_generic_to_shared(src_smem);
  float2 dst;
  if constexpr (std::is_same<IType2, typename ptx::bf16x2>::value) {
    asm volatile(
        "{\n\t"
        ".reg.b32 in_2x; \n\t"
        "ld.shared.b32 in_2x, [%1]; \n\t"
        ".reg.b16 in1, in2; \n\t"
        "mov.b32 {in1, in2}, in_2x; \n\t"
        ".reg.f32 out1, out2; \n\t"
        "cvt.rn.f32.bf16 out1, in1; \n\t"
        "cvt.rn.f32.bf16 out2, in2; \n\t"
        "mov.b64 %0, {out1, out2}; \n"
        "}\n"
        : "=l"(reinterpret_cast<uint64_t &>(dst))
        : "r"(src_smem_ptr));
  } else if constexpr (std::is_same<IType2, typename ptx::fp16x2>::value) {
    asm volatile(
        "{\n\t"
        ".reg.b32 in_2x; \n\t"
        "ld.shared.b32 in_2x, [%1]; \n\t"
        ".reg.b16 in1, in2; \n\t"
        "mov.b32 {in1, in2}, in_2x; \n\t"
        ".reg.f32 out1, out2; \n\t"
        "cvt.f32.f16 out1, in1; \n\t"
        "cvt.f32.f16 out2, in2; \n\t"
        "mov.b64 %0, {out1, out2}; \n"
        "}\n"
        : "=l"(reinterpret_cast<uint64_t &>(dst))
        : "r"(src_smem_ptr));
  } else {
    NVTE_DEVICE_ERROR("Unsupported by the ld_shared_cvt_f32x2 type.");
  }
  return dst;
#else
  NVTE_DEVICE_ERROR("ld_shared_cvt_f32x2 is only supported on SM 9.0+.");
#endif
}

}  // namespace ptx

namespace {

template <int num_barriers, int THREADS_PER_BLOCK>
__forceinline__ __device__ void initialize_barriers(uint64_t *mbar, const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int iter = 0; iter < num_barriers; ++iter) {
      ptx::mbarrier_init(&mbar[iter], THREADS_PER_BLOCK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();
#else
  NVTE_DEVICE_ERROR("initialize_barriers is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <int num_barriers>
__forceinline__ __device__ void destroy_barriers(uint64_t *mbar, const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int iter = 0; iter < num_barriers; ++iter) {
      ptx::mbarrier_invalid(&mbar[iter]);
    }
  }
#else
  NVTE_DEVICE_ERROR("destroy_barriers is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__forceinline__ __device__ void copy_1d_to_shared(void *dst, const void *src,
                                                  const size_t num_bytes, uint64_t *barrier,
                                                  const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_1d_global_to_shared(reinterpret_cast<uint64_t *>(dst),
                                                  reinterpret_cast<const uint64_t *>(src),
                                                  num_bytes, barrier);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(barrier, num_bytes);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#else
  NVTE_DEVICE_ERROR("copy_1d_to_shared is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__forceinline__ __device__ void copy_2d_to_shared(void *dst, const void *src, const size_t chunk_X,
                                                  const size_t chunk_Y, const size_t num_bytes,
                                                  uint64_t *barrier, const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst),
                                                  reinterpret_cast<const uint64_t *>(src), chunk_X,
                                                  chunk_Y, barrier);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(barrier, num_bytes);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#else
  NVTE_DEVICE_ERROR("copy_2d_to_shared is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__forceinline__ __device__ void copy_2d_to_sharedx2(void *dst, const void *src,
                                                    const size_t chunk_X1, const size_t chunk_Y1,
                                                    void *dst2, const void *src2,
                                                    const size_t chunk_X2, const size_t chunk_Y2,
                                                    const size_t num_bytes, uint64_t *barrier,
                                                    const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst),
                                                  reinterpret_cast<const uint64_t *>(src), chunk_X1,
                                                  chunk_Y1, barrier);

    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst2),
                                                  reinterpret_cast<const uint64_t *>(src2),
                                                  chunk_X2, chunk_Y2, barrier);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(barrier, 2 * num_bytes);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#else
  NVTE_DEVICE_ERROR("copy_2d_to_sharedx2 is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__forceinline__ __device__ void copy_2d_to_sharedx3(
    void *dst, const void *src, const size_t chunk_X1, const size_t chunk_Y1, void *dst2,
    const void *src2, const size_t chunk_X2, const size_t chunk_Y2, void *dst3, const void *src3,
    const size_t chunk_X3, const size_t chunk_Y3, const size_t num_bytes, uint64_t *barrier,
    const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst),
                                                  reinterpret_cast<const uint64_t *>(src), chunk_X1,
                                                  chunk_Y1, barrier);

    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst2),
                                                  reinterpret_cast<const uint64_t *>(src2),
                                                  chunk_X2, chunk_Y2, barrier);

    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst3),
                                                  reinterpret_cast<const uint64_t *>(src3),
                                                  chunk_X3, chunk_Y3, barrier);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(barrier, 3 * num_bytes);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#else
  NVTE_DEVICE_ERROR("copy_2d_to_sharedx3 is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

}  // namespace
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PTX_CUH_
