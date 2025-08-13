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

__device__ __forceinline__ 
int32_t elect_one_sync(uint32_t mask = 0xFFFFFFFFu) {
    int32_t pred = 0;
    asm volatile (
        "{\n\t"
            ".reg .pred %px; \n"
            "elect.sync _|%px, %1; \n"
            "selp.b32 %0, 1, 0, %px; \n"
        "\n\t}"
        : "=r"(pred)
        : "r"(mask)
    );
    return pred;
}

__device__ __forceinline__ 
void numbered_barrier_sync(uint32_t num_threads, uint32_t barrier_id = 1u) {
    asm volatile ("bar.sync %0, %1;\n" :: "r"(barrier_id), "r"(num_threads));
}


// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, const uint32_t count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mbar_ptr), "r"(count) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
__device__ __forceinline__ void mbarrier_invalid(uint64_t *mbar) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(mbar_ptr) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive(uint64_t *mbar) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(mbar_ptr) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *mbar, const uint32_t tx_count) {
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
    uint64_t *dst_shmem, const uint64_t *src_global_ptr, const uint32_t size, uint64_t *mbar) {
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
    uint64_t *dst_shmem, const uint64_t *tensor_map_ptr, const uint32_t offset_x,
    const uint32_t offset_y, uint64_t *mbar) {
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

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t *mbar, const uint32_t parity) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  while (!mbarrier_try_wait_parity(mbar_ptr, parity)) {
  }
}

#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

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
#if ((__CUDA_ARCH_HAS_FEATURE__(SM100_ALL)) || (__CUDA_ARCH_HAS_FEATURE__(SM101_ALL)) || \
     (__CUDA_ARCH_HAS_FEATURE__(SM120_ALL)))
  uint16_t out;
  asm volatile(
      "{\n"
      "cvt.rp.satfinite.ue8m0x2.f32  %0, 0.0, %1;\n"
      "}"
      : "=h"(out)
      : "f"(val));
  return *reinterpret_cast<e8m0_t *>(&out);
#else
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
#endif
}

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_1d_shared_to_global(uint64_t *dst_global_ptr,
                                                                         const uint64_t *src_shmem,
                                                                         const uint32_t size) {
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_global_ptr),
               "r"(src_shmem_ptr), "r"(size)
               : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_2d_shared_to_global(
    const uint64_t *tensor_map_ptr, const uint32_t offset_x, const uint32_t offset_y,
    uint64_t *src_shmem) {
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
                   tensor_map_ptr),
               "r"(offset_x), "r"(offset_y), "r"(src_shmem_ptr)
               : "memory");
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

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
__device__ __forceinline__ void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

// Proxy fence (bi-directional):
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async;"); }

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;");
}

template <typename T>
struct alignas(2 * sizeof(T)) FPx2 {
  T x;
  T y;
};

using floatx2 = FPx2<float>;
using bf16x2 = FPx2<bf16>;
using fp16x2 = FPx2<fp16>;
using fp8e4m3x2 = FPx2<fp8e4m3>;
using fp8e5m2x2 = FPx2<fp8e5m2>;

static_assert(sizeof(floatx2) == 8);
static_assert(sizeof(bf16x2) == 4);
static_assert(sizeof(fp16x2) == 4);
static_assert(sizeof(fp8e4m3x2) == 2);
static_assert(sizeof(fp8e5m2x2) == 2);

template <typename T>
struct alignas(4 * sizeof(T)) FPx4 {
    T x, y, z, w;
};
using floatx4 = FPx4<float>;
using bf16x4 = FPx4<bf16>;
using fp16x4 = FPx4<fp16>;
using fp8e4m3x4 = FPx4<fp8e4m3>;
using fp8e5m2x4 = FPx4<fp8e5m2>;

static_assert(sizeof(floatx4) == 16);
static_assert(sizeof(bf16x4) == 8);
static_assert(sizeof(fp16x4) == 8);
static_assert(sizeof(fp8e4m3x4) == 4);
static_assert(sizeof(fp8e5m2x4) == 4);


// SIMD like "Fused" cast + multiplication (x2)
__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const floatx2 &in,
                                           const floatx2 &scale) {
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
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const floatx2 &in,
                                           const floatx2 &scale) {
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
}

__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const bf16x2 &in, const floatx2 &scale) {
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
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const bf16x2 &in, const floatx2 &scale) {
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
}

__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const fp16x2 &in, const floatx2 &scale) {
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
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const fp16x2 &in, const floatx2 &scale) {
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
}

__device__ __forceinline__ void abs_max_2x(bf16x2 &dst, const bf16x2 &p1, const bf16x2 &p2) {
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;"
               : "=r"(reinterpret_cast<uint32_t &>(dst))
               : "r"(reinterpret_cast<const uint32_t &>(p1)),
                 "r"(reinterpret_cast<const uint32_t &>(p2)));
}

__device__ __forceinline__ void abs_max_2x(fp16x2 &dst, const fp16x2 &p1, const fp16x2 &p2) {
  asm volatile("max.xorsign.abs.f16x2 %0, %1, %2;"
               : "=r"(reinterpret_cast<uint32_t &>(dst))
               : "r"(reinterpret_cast<const uint32_t &>(p1)),
                 "r"(reinterpret_cast<const uint32_t &>(p2)));
}

__device__ __forceinline__
bf16 get_amax(bf16 a, bf16 b) {
    bf16 r;
    asm volatile (
        "max.xorsign.abs.bf16 %0, %1, %2;"
        : "=h"(*reinterpret_cast<int16_t*>(&r))
        : "h"(*reinterpret_cast<int16_t*>(&a)),
          "h"(*reinterpret_cast<int16_t*>(&b))
    );
    return r;
}

__device__ __forceinline__
fp16 get_amax(fp16 a, fp16 b) {
    fp16 r;
    asm volatile (
        "max.xorsign.abs.f16 %0, %1, %2;"
        : "=h"(*reinterpret_cast<int16_t*>(&r))
        : "h"(*reinterpret_cast<int16_t*>(&a)),
          "h"(*reinterpret_cast<int16_t*>(&b))
    );
    return r;
}

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, const bf16x4 &in, const ptx::floatx2 &scale) {
    ptx::bf16x2 const * in2 = reinterpret_cast<ptx::bf16x2 const*>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "prmt.b32 val2, 0x0, %1, 0x7632;\n\t"
        "prmt.b32 val1, 0x0, %1, 0x5410;\n\t"
        "prmt.b32 val4, 0x0, %2, 0x7632;\n\t"
        "prmt.b32 val3, 0x0, %2, 0x5410;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %3, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, const bf16x4 &in, const floatx4 &scale) {
    ptx::bf16x2 const * in2 = reinterpret_cast<ptx::bf16x2 const*>(&in);
    ptx::floatx2 const * scale2 = reinterpret_cast<ptx::floatx2 const*>(&scale);
    asm volatile (
        "{\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "prmt.b32 val2, 0x0, %1, 0x7632;\n\t"
        "prmt.b32 val1, 0x0, %1, 0x5410;\n\t"
        "prmt.b32 val4, 0x0, %2, 0x7632;\n\t"
        "prmt.b32 val3, 0x0, %2, 0x5410;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %4, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[0])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[1])),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, const bf16x4 &in, const ptx::floatx2 &scale) {
    ptx::bf16x2 const * in2 = reinterpret_cast<ptx::bf16x2 const*>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "prmt.b32 val2, 0x0, %1, 0x7632;\n\t"
        "prmt.b32 val1, 0x0, %1, 0x5410;\n\t"
        "prmt.b32 val4, 0x0, %2, 0x7632;\n\t"
        "prmt.b32 val3, 0x0, %2, 0x5410;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %3, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, const bf16x4 &in, const floatx4 &scale) {
    ptx::bf16x2 const * in2 = reinterpret_cast<ptx::bf16x2 const*>(&in);
    ptx::floatx2 const * scale2 = reinterpret_cast<ptx::floatx2 const*>(&scale);
    asm volatile (
        "{\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "prmt.b32 val2, 0x0, %1, 0x7632;\n\t"
        "prmt.b32 val1, 0x0, %1, 0x5410;\n\t"
        "prmt.b32 val4, 0x0, %2, 0x7632;\n\t"
        "prmt.b32 val3, 0x0, %2, 0x5410;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %4, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[0])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[1])),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, const fp16x4 &in, const ptx::floatx2 &scale) {
    ptx::fp16x2 const * in2 = reinterpret_cast<ptx::fp16x2 const*>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b16 val1_f16;\n\t"
        ".reg.b16 val2_f16;\n\t"
        ".reg.b16 val3_f16;\n\t"
        ".reg.b16 val4_f16;\n\t"
        "mov.b32 {val1_f16, val2_f16}, %1;\n\t"
        "mov.b32 {val3_f16, val4_f16}, %2;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "cvt.f32.f16 val1, val1_f16;\n\t"
        "cvt.f32.f16 val2, val2_f16;\n\t"
        "cvt.f32.f16 val3, val3_f16;\n\t"
        "cvt.f32.f16 val4, val4_f16;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %3, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, const fp16x4 &in, const floatx4 &scale) {
    ptx::fp16x2 const * in2 = reinterpret_cast<ptx::fp16x2 const*>(&in);
    ptx::floatx2 const * scale2 = reinterpret_cast<ptx::floatx2 const*>(&scale);
    asm volatile (
        "{\n\t"
        ".reg.b16 val1_f16;\n\t"
        ".reg.b16 val2_f16;\n\t"
        ".reg.b16 val3_f16;\n\t"
        ".reg.b16 val4_f16;\n\t"
        "mov.b32 {val1_f16, val2_f16}, %1;\n\t"
        "mov.b32 {val3_f16, val4_f16}, %2;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "cvt.f32.f16 val1, val1_f16;\n\t"
        "cvt.f32.f16 val2, val2_f16;\n\t"
        "cvt.f32.f16 val3, val3_f16;\n\t"
        "cvt.f32.f16 val4, val4_f16;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %4, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[0])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[1])),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, const fp16x4 &in, const ptx::floatx2 &scale) {
    ptx::fp16x2 const * in2 = reinterpret_cast<ptx::fp16x2 const*>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b16 val1_f16;\n\t"
        ".reg.b16 val2_f16;\n\t"
        ".reg.b16 val3_f16;\n\t"
        ".reg.b16 val4_f16;\n\t"
        "mov.b32 {val1_f16, val2_f16}, %1;\n\t"
        "mov.b32 {val3_f16, val4_f16}, %2;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "cvt.f32.f16 val1, val1_f16;\n\t"
        "cvt.f32.f16 val2, val2_f16;\n\t"
        "cvt.f32.f16 val3, val3_f16;\n\t"
        "cvt.f32.f16 val4, val4_f16;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %3, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, const fp16x4 &in, const floatx4 &scale) {
    ptx::fp16x2 const * in2 = reinterpret_cast<ptx::fp16x2 const*>(&in);
    ptx::floatx2 const * scale2 = reinterpret_cast<ptx::floatx2 const*>(&scale);
    asm volatile (
        "{\n\t"
        ".reg.b16 val1_f16;\n\t"
        ".reg.b16 val2_f16;\n\t"
        ".reg.b16 val3_f16;\n\t"
        ".reg.b16 val4_f16;\n\t"
        "mov.b32 {val1_f16, val2_f16}, %1;\n\t"
        "mov.b32 {val3_f16, val4_f16}, %2;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "cvt.f32.f16 val1, val1_f16;\n\t"
        "cvt.f32.f16 val2, val2_f16;\n\t"
        "cvt.f32.f16 val3, val3_f16;\n\t"
        "cvt.f32.f16 val4, val4_f16;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %4, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[0])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[1])),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, floatx4 const &in, const ptx::floatx2 &scale) {
    ptx::floatx2 const * in2 = reinterpret_cast<ptx::floatx2 const *>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        ".reg.b64 re1;\n\t"
        ".reg.b64 re2;\n\t"
        "fma.rn.f32x2 re1, %1, %3, zeros;\n\t"
        "fma.rn.f32x2 re2, %2, %3, zeros;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "mov.b64 {val1, val2}, re1;\n\t"
        "mov.b64 {val3, val4}, re2;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "l"(reinterpret_cast<uint64_t const&>(in2[0])),
          "l"(reinterpret_cast<uint64_t const&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, floatx4 const &in, const floatx4 &scale) {
    ptx::floatx2 const * in2 = reinterpret_cast<ptx::floatx2 const *>(&in);
    ptx::floatx2 const * scale2 = reinterpret_cast<ptx::floatx2 const*>(&scale);
    asm volatile (
        "{\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        ".reg.b64 re1;\n\t"
        ".reg.b64 re2;\n\t"
        "fma.rn.f32x2 re1, %1, %3, zeros;\n\t"
        "fma.rn.f32x2 re2, %2, %4, zeros;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "mov.b64 {val1, val2}, re1;\n\t"
        "mov.b64 {val3, val4}, re2;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "l"(reinterpret_cast<uint64_t const&>(in2[0])),
          "l"(reinterpret_cast<uint64_t const&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[0])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[1])),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, floatx4 const &in, const ptx::floatx2 &scale) {
    ptx::floatx2 const * in2 = reinterpret_cast<ptx::floatx2 const *>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        ".reg.b64 re1;\n\t"
        ".reg.b64 re2;\n\t"
        "fma.rn.f32x2 re1, %1, %3, zeros;\n\t"
        "fma.rn.f32x2 re2, %2, %3, zeros;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "mov.b64 {val1, val2}, re1;\n\t"
        "mov.b64 {val3, val4}, re2;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "l"(reinterpret_cast<uint64_t const&>(in2[0])),
          "l"(reinterpret_cast<uint64_t const&>(in2[1]))
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, floatx4 const &in, const floatx4 &scale) {
    ptx::floatx2 const * in2 = reinterpret_cast<ptx::floatx2 const*>(&in);
    ptx::floatx2 const * scale2 = reinterpret_cast<ptx::floatx2 const*>(&scale);
    asm volatile (
        "{\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        ".reg.b64 re1;\n\t"
        ".reg.b64 re2;\n\t"
        "fma.rn.f32x2 re1, %1, %3, zeros;\n\t"
        "fma.rn.f32x2 re2, %2, %4, zeros;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "mov.b64 {val1, val2}, re1;\n\t"
        "mov.b64 {val3, val4}, re2;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "l"(reinterpret_cast<uint64_t const&>(in2[0])),
          "l"(reinterpret_cast<uint64_t const&>(in2[1]))
          "l"(reinterpret_cast<const uint64_t&>(scale2[0])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[1])),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void abs_max_2x(float &dst, const float &p1, const float &p2, const float &p3) {
    asm volatile (
        "max.abs.f32 %0, %1, %2, %3;"
        : "=f"(dst)
        : "f"(p1), "f"(p2), "f"(p3)
    );
}

__device__ __forceinline__
ptx::floatx2 up_cast(const ptx::fp16x2 &in) {
    ptx::floatx2 out;
    asm volatile (
        "{\n\t"
        ".reg.b16 f16_1;\n\t"
        ".reg.b16 f16_2;\n\t"
        "mov.b32 {f16_1, f16_2}, %2;\n\t"
        "cvt.f32.f16 %0, f16_1;\n\t"
        "cvt.f32.f16 %1, f16_2;\n\t"
        "}\n\t"
        : "=f"(out.x), "=f"(out.y)
        : "r"(reinterpret_cast<int32_t const&>(in))
    );
    return out;
}

__device__ __forceinline__
floatx4 up_cast(const fp16x4 &in) {
    floatx4 out;
    asm volatile (
        "{\n\t"
        ".reg.b16 f16_1;\n\t"
        ".reg.b16 f16_2;\n\t"
        ".reg.b16 f16_3;\n\t"
        ".reg.b16 f16_4;\n\t"
        "mov.b64 {f16_1, f16_2, f16_3, f16_4}, %4;\n\t"
        "cvt.f32.f16 %0, f16_1;\n\t"
        "cvt.f32.f16 %1, f16_2;\n\t"
        "cvt.f32.f16 %2, f16_3;\n\t"
        "cvt.f32.f16 %3, f16_4;\n\t"
        "}\n\t"
        : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w)
        : "l"(reinterpret_cast<int64_t const&>(in))
    );
    return out;
}

__device__ __forceinline__
ptx::floatx2 up_cast(const ptx::bf16x2 &in) {
    ptx::floatx2 out;
    asm volatile (
        "{\n\t"
        "prmt.b32 %1, 0x0, %2, 0x7632;\n\t"
        "prmt.b32 %0, 0x0, %2, 0x5410;\n\t"
        "}\n\t"
        : "=r"(reinterpret_cast<int32_t&>(out.x)),
          "=r"(reinterpret_cast<int32_t&>(out.y))
        : "r"(reinterpret_cast<int32_t const&>(in))
    );
    return out;
}

__device__ __forceinline__
floatx4 up_cast(const bf16x4 &in) {
    floatx4 out;
    int32_t const *in2 = reinterpret_cast<int32_t const *>(&in);
    asm volatile (
        "{\n\t"
        "prmt.b32 %1, 0x0, %4, 0x7632;\n\t"
        "prmt.b32 %0, 0x0, %4, 0x5410;\n\t"
        "prmt.b32 %3, 0x0, %5, 0x7632;\n\t"
        "prmt.b32 %2, 0x0, %5, 0x5410;\n\t"
        "}\n\t"
        : "=r"(reinterpret_cast<int32_t&>(out.x)),
          "=r"(reinterpret_cast<int32_t&>(out.y)),
          "=r"(reinterpret_cast<int32_t&>(out.z)),
          "=r"(reinterpret_cast<int32_t&>(out.w))
        : "r"(in2[0]),
          "r"(in2[1])
    );
    return out;
}

#endif // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

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
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

}  // namespace
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PTX_CUH_
