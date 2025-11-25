/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file swizzle.cuh
 *  \brief CUDA kernels to swizzle.
 */

#ifndef TRANSFORMER_ENGINE_SPECIALIZED_SWIZZLE_CUH_
#define TRANSFORMER_ENGINE_SPECIALIZED_SWIZZLE_CUH_

#include <cmath>
#include <cstdint>

namespace transformer_engine {
namespace swz {

template <auto v>
struct C {
  using type = C<v>;
  static constexpr auto value = v;
  using value_type = decltype(v);

  __device__ __host__ __forceinline__ constexpr operator value_type() const noexcept {
    return value;
  }
};

template <class T, T v>
using constant = C<v>;

template <class T, typename Ts, Ts s>
__host__ __device__ __forceinline__ constexpr T shiftr(T x) {
  if constexpr (std::is_same_v<Ts, uint32_t>) {
    return x >> s;
  } else if constexpr (std::is_same_v<Ts, int32_t>) {
    if constexpr (s >= 0) {
      return x >> s;
    } else {
      return x << -s;
    }
  }
}

template <int32_t BBits, int32_t MBase, int32_t SShift>
struct Swizzle {
  static constexpr int32_t num_bits = BBits;   // number of rows
  static constexpr int32_t num_base = MBase;   // number of elements within a chunk
  static constexpr int32_t num_shft = SShift;  // number of columns, at the granularity of a chunk

  static_assert(num_base >= 0, "MBase must be non-negative");
  static_assert(num_bits >= 0, "BBits must be non-negative");
  static_assert(abs(num_shft) >= num_bits, "abs(SShift) must be greater than or equal to num_bits");

  using bit_mask = constant<int32_t, (1 << num_bits) - 1>;
  using yyy_mask =
      constant<int32_t, bit_mask{} << (num_base + std::max(decltype(num_shft){0}, num_shft))>;
  using zzz_mask =
      constant<int32_t, bit_mask{} << (num_base - std::min(decltype(num_shft){0}, num_shft))>;
  using msk_shft = constant<int32_t, num_shft>;
  static constexpr int32_t swz_code = int32_t(yyy_mask{} | zzz_mask{});

  template <class Offset>
  __host__ __device__ __forceinline__ constexpr static int32_t apply(Offset const &offset) {
    return offset ^
           shiftr<Offset, typename msk_shft::value_type, msk_shft::value>(offset & yyy_mask{});
  }

  __host__ __device__ __forceinline__ constexpr static int32_t swz(int32_t const &offset) {
    return apply(offset);
  }
};

struct Linear {
  template <class Offset>
  __host__ __device__ __forceinline__ constexpr static int32_t apply(Offset const &offset) {
    return offset;
  }

  __host__ __device__ __forceinline__ constexpr static int32_t swz(int32_t const &offset) {
    return offset;
  }
};

}  // namespace swz
}  // namespace transformer_engine

#endif  // #ifndef TRANSFORMER_ENGINE_SPECIALIZED_SWIZZLE_CUH_
