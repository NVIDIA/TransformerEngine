/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dgelu_table.cuh
 *  \brief dgelu of BF16 inputs by table lookup, for kernels that fuse dGeLU.
 *
 *  A BF16 input takes only 2^16 values, so dgelu of it can be read from a table
 *  instead of computed. The table holds, as FP32, dgelu of every BF16 value whose
 *  magnitude lies in [2^-12, 8], both signs, exactly as activation_2x computes
 *  them, in 16 KB. Inputs outside that window go through a slower per-lane path
 *  that returns the same values:
 *    - below 2^-12, tanh of the argument rounds to the argument itself, and a
 *      short form reproduces the rounding of the full formula;
 *    - above 8, dgelu has saturated to the last tabulated value, until the
 *      formula overflows to NaN; from there on, and for Inf and NaN, the full
 *      formula runs.
 *
 *  Usage:
 *    - host, before the launch: ensure_table(stream);
 *    - device: load_table_async(smem, &barrier, threadIdx.x), then
 *      wait_table(&barrier) before the first lookup(x, smem).
 *  The table is a __device__ array in an anonymous namespace, so each
 *  translation unit that includes this header has its own copy per device, and
 *  its own ensure_table to build it.
 */

#ifndef TRANSFORMER_ENGINE_UTIL_DGELU_TABLE_CUH_
#define TRANSFORMER_ENGINE_UTIL_DGELU_TABLE_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "../common.h"
#include "math.h"
#include "packed_activation.cuh"
#include "ptx.cuh"

namespace transformer_engine {
namespace dgelu_table {

//! BF16 bits of 2^-12, the smallest tabulated magnitude.
constexpr uint32_t kLoBits = 0x3980u;
//! BF16 bits of 8.0, the largest tabulated magnitude.
constexpr uint32_t kHiBits = 0x4100u;
//! Tabulated magnitudes per sign.
constexpr int kEntriesPerSign = kHiBits - kLoBits + 1;
//! Entries between the two signs. byte_offsets turns the BF16 sign bit, 0x8000, into
//! a byte offset of 0x8000 >> 2, which is this many 4-byte entries.
constexpr int kSignStride = (0x8000 >> 2) / 4;
static_assert(kEntriesPerSign <= kSignStride, "The two signs' entries must not overlap.");
//! Table size in bytes: up to the last negative entry, rounded up to the 16-byte
//! granularity of the bulk copy.
constexpr int kBytes = ((kSignStride + kEntriesPerSign) * 4 + 15) / 16 * 16;
//! BF16 bits of the smallest magnitude at which dgelu overflows to NaN. Between
//! kHiBits and this, dgelu equals the last tabulated value.
constexpr uint32_t kOverflowBits = 0x6044u;

namespace {

__device__ __align__(16) unsigned char d_table[kBytes];

__device__ __forceinline__ float dgelu_scalar(const float x) {
  return activation_2x<Empty, dgelu<float, float>>({x, x}, {}).x;
}

__global__ void init_table_kernel() {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * kEntriesPerSign) return;
  const bool negative = i >= kEntriesPerSign;
  const uint32_t mag = kLoBits + (negative ? i - kEntriesPerSign : i);
  const uint32_t bits = (negative ? 0x8000u : 0u) | mag;
  const int entry = (mag - kLoBits) + (negative ? kSignStride : 0);
  reinterpret_cast<float *>(d_table)[entry] = dgelu_scalar(__uint_as_float(bits << 16));
}

/*! \brief Build this translation unit's table on the current device, once.
 *
 *  A build issued under stream capture only runs when the graph does, so it does
 *  not mark the device ready. Rebuilding is idempotent.
 */
inline void ensure_table(cudaStream_t stream) {
  static std::mutex mutex;
  static std::vector<bool> ready;
  int device;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  if (static_cast<size_t>(device) < ready.size() && ready[device]) return;
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  init_table_kernel<<<(2 * kEntriesPerSign + 255) / 256, 256, 0, stream>>>();
  NVTE_CHECK_CUDA(cudaGetLastError());
  if (capture == cudaStreamCaptureStatusNone) {
    // Other streams on this device will skip the build, so it must complete.
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (static_cast<size_t>(device) >= ready.size()) ready.resize(device + 1, false);
    ready[device] = true;
  }
}

/*! \brief Start copying the table into \p smem (kBytes, 16-byte aligned) with one bulk copy.
 *
 *  Called by every thread of the CTA; contains a __syncthreads.
 */
__device__ __forceinline__ void load_table_async(unsigned char *smem, uint64_t *barrier,
                                                 const int tid) {
  if (tid == 0) ptx::mbarrier_init(barrier, 1);
  __syncthreads();
  if (tid == 0) {
    ptx::mbarrier_arrive_expect_tx(barrier, kBytes);
    ptx::cp_async_bulk_tensor_1d_global_to_shared(reinterpret_cast<uint64_t *>(smem),
                                                  reinterpret_cast<const uint64_t *>(d_table),
                                                  kBytes, barrier);
  }
}

__device__ __forceinline__ void wait_table(uint64_t *barrier) {
  ptx::mbarrier_wait_parity(barrier, 0);
}

}  // namespace

/*! \brief Byte offsets of both halves of a BF16 pair into the table, low half in the low
 *         16 bits.
 *
 *  The magnitudes are clamped into the tabulated window. The sign bit becomes the
 *  kSignStride entries, so each offset stays below 0x10000 and the pair never carries.
 *
 *  \param[in,out] clamped  Accumulates every bit the clamp moved.
 */
__device__ __forceinline__ uint32_t byte_offsets(const uint32_t x, uint32_t &clamped) {
  constexpr uint32_t kLoBitsX2 = kLoBits * 0x10001u;
  constexpr uint32_t kHiBitsX2 = kHiBits * 0x10001u;
  const uint32_t mag = x & 0x7fff7fffu;
  const __nv_bfloat162 c = __hmin2(__hmax2(reinterpret_cast<const __nv_bfloat162 &>(mag),
                                           reinterpret_cast<const __nv_bfloat162 &>(kLoBitsX2)),
                                   reinterpret_cast<const __nv_bfloat162 &>(kHiBitsX2));
  const uint32_t c_bits = reinterpret_cast<const uint32_t &>(c);
  clamped |= c_bits ^ mag;
  return ((c_bits - kLoBitsX2) << 2) + ((x ^ mag) >> 2);
}

/*! \brief dgelu for |x| < 2^-12, where tanh(u) rounds to u. Rounds as the full formula. */
__device__ __forceinline__ float dgelu_tiny(const float x) {
  const float t = __fmul_rn(0.79788456f, x);
  const float f = __fmul_rn(__fmul_rn(0.5f, x), 0.79788456f);
  return __fadd_rn(f, __fmaf_rn(0.5f, t, 0.5f));
}

/*! \brief dgelu of the BF16 value \p bits, given the table entry its clamped magnitude read. */
__device__ __forceinline__ float dgelu_outside_window(const uint32_t bits, const float probe) {
  const uint32_t mag = bits & 0x7fffu;
  const float x = __uint_as_float(bits << 16);
  if (mag < kLoBits) return dgelu_tiny(x);
  if (mag >= kOverflowBits) return dgelu_scalar(x);
  return probe;
}

/*! \brief dgelu of both halves of the BF16 pair \p x, from the table in \p table. */
__device__ __forceinline__ ptx::floatx2 lookup(const uint32_t x,
                                               const unsigned char *__restrict__ table) {
  uint32_t clamped = 0u;
  const uint32_t offsets = byte_offsets(x, clamped);
  const ptx::floatx2 probe = {*reinterpret_cast<const float *>(table + (offsets & 0xffffu)),
                              *reinterpret_cast<const float *>(table + (offsets >> 16))};
  if (__builtin_expect(clamped == 0u, 1)) return probe;
  return {dgelu_outside_window(x & 0xffffu, probe.x), dgelu_outside_window(x >> 16, probe.y)};
}

}  // namespace dgelu_table
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_UTIL_DGELU_TABLE_CUH_
