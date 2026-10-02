/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file activation_table.cuh
 *  \brief util/math.h activations of BF16 inputs by table lookup.
 *
 *  A BF16 input takes only 2^16 values, so an activation of it can be read from
 *  a table instead of computed. The table for OP holds, as FP32, OP of every BF16
 *  value whose magnitude lies in Window<OP>, both signs, exactly as activation_2x
 *  computes them, in at most 16 KB. Past the top of the window OP has saturated,
 *  so up to Window<OP>::kSaturatedEndBits the top entry of the same sign is
 *  returned. Every other input (below the window, past the saturated range, Inf
 *  and NaN) is computed with activation_2x.
 *
 *  Usage:
 *    - host, before the launch: ensure_table<OP>(stream);
 *    - device: load_table_async<OP>(smem, &barrier, threadIdx.x), then
 *      wait_table(&barrier) before the first lookup<OP>(x, smem).
 *  The table is a __device__ variable template in an anonymous namespace, so
 *  each translation unit that looks up OP has its own copy per device, and its
 *  own ensure_table to build it; the others have none.
 */

#ifndef TRANSFORMER_ENGINE_UTIL_ACTIVATION_TABLE_CUH_
#define TRANSFORMER_ENGINE_UTIL_ACTIVATION_TABLE_CUH_

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
namespace activation_table {

//! Entries between the two signs. byte_offsets turns the BF16 sign bit, 0x8000, into a byte
//! offset of 0x8000 >> 2, which is this many 4-byte entries.
constexpr int kSignStride = (0x8000 >> 2) / 4;
//! Shared memory a table takes at most.
constexpr int kMaxBytes = 2 * kSignStride * 4;

/*! \brief Tabulated magnitudes [kLo, kHi] and saturated magnitudes (kHi, kSaturatedEnd), as
 *         BF16 bits.
 */
template <uint32_t kLo, uint32_t kHi, uint32_t kSaturatedEnd>
struct WindowBits {
  static constexpr uint32_t kLoBits = kLo;
  static constexpr uint32_t kHiBits = kHi;
  static constexpr uint32_t kSaturatedEndBits = kSaturatedEnd;
  static constexpr int kEntriesPerSign = kHi - kLo + 1;
  static_assert(kEntriesPerSign <= kSignStride, "The two signs' entries must not overlap.");
  //! Up to the last negative entry, rounded up to the 16-byte granularity of the bulk copy.
  static constexpr int kBytes = ((kSignStride + kEntriesPerSign) * 4 + 15) / 16 * 16;
};

template <float (*OP)(float, const Empty &)>
struct Window;

//! [2^-12, 8]. dgelu has saturated, to 1 and to 0, by |x| = 8, until its formula overflows to
//! NaN at 0x6044.
template <>
struct Window<dgelu<float, float>> : WindowBits<0x3980u, 0x4100u, 0x6044u> {};

//! [2^-9, 89]. dsilu has saturated to 1 by x = 16.75 and to 0 by x = -89, up to Inf.
template <>
struct Window<dsilu<float, float>> : WindowBits<0x3B00u, 0x42B2u, 0x7F80u> {};

namespace {

template <float (*OP)(float, const Empty &)>
__device__ __align__(16) unsigned char d_table[Window<OP>::kBytes];

template <float (*OP)(float, const Empty &)>
__global__ void init_table_kernel() {
  using W = Window<OP>;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * W::kEntriesPerSign) return;
  const bool negative = i >= W::kEntriesPerSign;
  const uint32_t mag = W::kLoBits + (negative ? i - W::kEntriesPerSign : i);
  const float x = __uint_as_float(((negative ? 0x8000u : 0u) | mag) << 16);
  const int entry = (mag - W::kLoBits) + (negative ? kSignStride : 0);
  reinterpret_cast<float *>(d_table<OP>)[entry] = activation_2x<Empty, OP>({x, x}, {}).x;
}

/*! \brief Build this translation unit's table for OP on the current device, once.
 *
 *  A build issued under stream capture only runs when the graph does, so it does
 *  not mark the device ready. Rebuilding is idempotent.
 */
template <float (*OP)(float, const Empty &)>
void ensure_table(cudaStream_t stream) {
  static std::mutex mutex;
  static std::vector<bool> ready;
  int device;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  if (static_cast<size_t>(device) < ready.size() && ready[device]) return;
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  init_table_kernel<OP><<<(2 * Window<OP>::kEntriesPerSign + 255) / 256, 256, 0, stream>>>();
  NVTE_CHECK_CUDA(cudaGetLastError());
  if (capture == cudaStreamCaptureStatusNone) {
    // Other streams on this device will skip the build, so it must complete.
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (static_cast<size_t>(device) >= ready.size()) ready.resize(device + 1, false);
    ready[device] = true;
  }
}

/*! \brief Start copying the table for OP into \p smem (16-byte aligned) with one bulk copy.
 *
 *  Called by every thread of the CTA; contains a __syncthreads.
 */
template <float (*OP)(float, const Empty &)>
__device__ __forceinline__ void load_table_async(unsigned char *smem, uint64_t *barrier,
                                                 const int tid) {
  if (tid == 0) {
    ptx::mbarrier_init(barrier, 1);
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if (tid == 0) {
    ptx::mbarrier_arrive_expect_tx(barrier, Window<OP>::kBytes);
    ptx::cp_async_bulk_tensor_1d_global_to_shared(reinterpret_cast<uint64_t *>(smem),
                                                  reinterpret_cast<const uint64_t *>(d_table<OP>),
                                                  Window<OP>::kBytes, barrier);
  }
}

}  // namespace

__device__ __forceinline__ void wait_table(uint64_t *barrier) {
  ptx::mbarrier_wait_parity(barrier, 0);
}

/*! \brief Byte offsets of both halves of a BF16 pair into the table for OP, low half in the
 *         low 16 bits.
 *
 *  The magnitudes are clamped into the window. The sign bit becomes the kSignStride
 *  entries, so each offset stays below 0x10000 and the pair never carries.
 *
 *  \param[in,out] clamped  Accumulates every bit the clamp moved.
 */
template <float (*OP)(float, const Empty &)>
__device__ __forceinline__ uint32_t byte_offsets(const uint32_t x, uint32_t &clamped) {
  constexpr uint32_t kLoBitsX2 = Window<OP>::kLoBits * 0x10001u;
  constexpr uint32_t kHiBitsX2 = Window<OP>::kHiBits * 0x10001u;
  const uint32_t mag = x & 0x7fff7fffu;
  const __nv_bfloat162 c = __hmin2(__hmax2(reinterpret_cast<const __nv_bfloat162 &>(mag),
                                           reinterpret_cast<const __nv_bfloat162 &>(kLoBitsX2)),
                                   reinterpret_cast<const __nv_bfloat162 &>(kHiBitsX2));
  const uint32_t c_bits = reinterpret_cast<const uint32_t &>(c);
  clamped |= c_bits ^ mag;
  return ((c_bits - kLoBitsX2) << 2) + ((x ^ mag) >> 2);
}

/*! \brief OP of both halves of the BF16 pair \p x, from the table for OP in \p table. */
template <float (*OP)(float, const Empty &)>
__device__ __forceinline__ ptx::floatx2 lookup(const uint32_t x,
                                               const unsigned char *__restrict__ table) {
  uint32_t clamped = 0u;
  const uint32_t offsets = byte_offsets<OP>(x, clamped);
  const ptx::floatx2 probe = {*reinterpret_cast<const float *>(table + (offsets & 0xffffu)),
                              *reinterpret_cast<const float *>(table + (offsets >> 16))};
  if (__builtin_expect(clamped == 0u, 1)) return probe;
  // Inside the window or the saturated range, the probe holds the value.
  const auto probed = [](const uint32_t bits) {
    const uint32_t mag = bits & 0x7fffu;
    return mag >= Window<OP>::kLoBits && mag < Window<OP>::kSaturatedEndBits;
  };
  const bool probed_lo = probed(x & 0xffffu), probed_hi = probed(x >> 16);
  if (probed_lo && probed_hi) return probe;
  const ptx::floatx2 computed =
      activation_2x<Empty, OP>({__uint_as_float(x << 16), __uint_as_float(x & 0xffff0000u)}, {});
  return {probed_lo ? probe.x : computed.x, probed_hi ? probe.y : computed.y};
}

}  // namespace activation_table
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_UTIL_ACTIVATION_TABLE_CUH_
