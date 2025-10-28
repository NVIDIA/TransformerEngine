/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CURANDDX_HPP_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CURANDDX_HPP_

namespace transformer_engine {
namespace curanddx {
namespace detail {

inline constexpr unsigned int philox4x32_w32_0 = 0x9E3779B9U;
inline constexpr unsigned int philox4x32_w32_1 = 0xBB67AE85U;
inline constexpr unsigned int philox4x32_m4x32_0 = 0xD2511F53U;
inline constexpr unsigned int philox4x32_m4x32_1 = 0xCD9E8D57U;

__forceinline__ __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                                  unsigned int* hip) {
  *hip = __umulhi(a, b);
  return a * b;
}

__forceinline__ __device__ uint4 single_round(uint4 ctr, uint2 key) {
  unsigned int hi0;
  unsigned int hi1;
  unsigned int lo0 = mulhilo32(philox4x32_m4x32_0, ctr.x, &hi0);
  unsigned int lo1 = mulhilo32(philox4x32_m4x32_1, ctr.z, &hi1);

  uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
  return ret;
}

template <unsigned int Rounds>
__forceinline__ __device__ uint4 multiple_rounds(uint4 c, uint2 k) {
  for (unsigned int i = 0; i < Rounds - 1; i++) {
    c = single_round(c, k);  // 1
    k.x += philox4x32_w32_0;
    k.y += philox4x32_w32_1;
  }
  return single_round(c, k);  // Rounds
}

template <unsigned int Rounds>
struct philox4x32_native_state {
  static constexpr unsigned int rounds = Rounds;

  uint4 ctr;
  uint2 key;

  __forceinline__ __device__ void philox_state_incr() {
    if (++ctr.x) return;
    if (++ctr.y) return;
    if (++ctr.z) return;
    ++ctr.w;
  }

  __forceinline__ __device__ void philox_state_incr(size_t n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);

    ctr.x += nlo;
    if (ctr.x < nlo) nhi++;

    ctr.y += nhi;
    if (nhi <= ctr.y) return;
    if (++ctr.z) return;
    ++ctr.w;
  }

  __forceinline__ __device__ void philox_state_incr_hi(size_t n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);

    ctr.z += nlo;
    if (ctr.z < nlo) nhi++;

    ctr.w += nhi;
  }

  // offset is the total # of 128bits generated with a single generate4() call
  __forceinline__ __device__ void skip_offset(size_t n) { philox_state_incr(n); }

  __forceinline__ __device__ void skip_subsequence(size_t n) { philox_state_incr_hi(n); }

  __forceinline__ __device__ void init(size_t seed, size_t subsequence, size_t offset) {
    ctr = make_uint4(0, 0, 0, 0);
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);

    skip_subsequence(subsequence);
    skip_offset(offset);
  }

  __forceinline__ __device__ uint4 generate4() {
    auto tmp = multiple_rounds<Rounds>(ctr, key);
    philox_state_incr();
    return tmp;
  }
};
}  // namespace detail
}  // namespace curanddx
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CURANDDX_HPP_
