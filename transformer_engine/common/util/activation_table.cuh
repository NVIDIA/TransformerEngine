/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file activation_table.cuh
 *  \brief GeLU, SiLU and their derivatives of BF16 inputs, read from lookup tables.
 *
 *  A BF16 input takes only 2^16 values, so these activations can be read from a table
 *  generated once per device from the scalar util/math.h functions. TE builds each table
 *  on its first use and keeps it for the life of the process (device_table); kernels read
 *  them for BF16 inputs quantized to MXFP8. NVTE_ACTIVATION_TABLE=0 turns them off.
 *  Outside each table's window the result has a closed form, so every lookup is
 *  branch-free.
 *
 *  Forward (GeLU, SiLU): BF16 entries. Below the window the result is RN(x/2); above it
 *  it is x for x > 0 and -0 for x < 0. The forward result is truncated to BF16 by every
 *  caller, so BF16 entries lose nothing. One difference from util/math.h: -inf gives -0,
 *  where the formula gives NaN (-inf * 0).
 *
 *  Derivatives (dGeLU, dSiLU): FP32 entries, because callers multiply the derivative by
 *  the incoming gradient in FP32. Below the window the derivative is its linear Taylor
 *  term 0.5 + c*x, rounded once; it differs from util/math.h by at most 1 ulp (dGeLU) or
 *  2 ulp (dSiLU) there. With NVTE_ACTIVATION_TABLE_EXACT_DSILU=1 the dSiLU table extends
 *  down to 2^-25, below which the linear term equals util/math.h, so dSiLU matches
 *  util/math.h for every input (32 KB instead of 16 KB). Above the window the derivative has saturated, and the table's
 *  last entry of the same sign holds the saturated value. Inputs where util/math.h
 *  returns NaN (dGeLU from 0x6044 on, where 0.107*x*x overflows; +-inf; NaN) keep NaN.
 *
 *  The tables follow TE's tanh-based GeLU. An erf-based variant would need its own
 *  tables, a different forward window, and an FP32 derivative Phi(x) + x*phi(x).
 */

#ifndef TRANSFORMER_ENGINE_UTIL_ACTIVATION_TABLE_CUH_
#define TRANSFORMER_ENGINE_UTIL_ACTIVATION_TABLE_CUH_

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

#include "../common.h"
#include "math.h"
#include "system.h"

namespace transformer_engine {
namespace activation_table {

//! Activation read from a table.
enum Kind : int {
  kGelu = 0,   //!< GeLU (tanh form), BF16 entries
  kSilu = 1,   //!< SiLU, BF16 entries
  kDgelu = 2,  //!< GeLU derivative (tanh form), FP32 entries
  kDsilu = 3,  //!< SiLU derivative, FP32 entries
  kNumKinds = 4,
};

/*! \brief Forward table over the BF16 magnitudes [kLo, kHi].
 *
 *  Rows are indexed by magnitude bits from kLowSentinel to kHighSentinel, two entries per
 *  row: op(+m) then op(-m). The kLowSentinel row holds -inf, so max(entry, RN(x/2)) picks
 *  RN(x/2) below the window. The kHighSentinel row holds +inf and -0, so min(entry, upper)
 *  picks x above the window for x > 0 and -0 for x < 0.
 */
template <uint32_t kLo, uint32_t kHi>
struct ForwardLayout {
  static constexpr bool kForward = true;
  static constexpr uint32_t kLowSentinel = kLo - 1;
  static constexpr uint32_t kHighSentinel = kHi + 1;
  static constexpr uint32_t kRows = kHighSentinel - kLowSentinel + 1;
  //! Rounded up to 16 bytes, the granularity of a bulk copy into shared memory.
  static constexpr size_t kBytes = (kRows * 2 * sizeof(__nv_bfloat16) + 15) / 16 * 16;
};

/*! \brief Derivative table over the BF16 magnitudes [kLo, kHi].
 *
 *  Positive inputs at entries [0, n), negative ones at [kSignStride, kSignStride + n).
 *  A power-of-two stride turns the BF16 sign bit into a byte offset with one shift.
 */
template <uint32_t kLo, uint32_t kHi, uint32_t kSaturatedEnd, uint32_t kStride = 2048>
struct DerivativeLayout {
  static constexpr bool kForward = false;
  static constexpr uint32_t kLoBits = kLo;
  static constexpr uint32_t kHiBits = kHi;
  //! Magnitudes from here on (and NaN) give NaN, as util/math.h does.
  static constexpr uint32_t kSaturatedEndBits = kSaturatedEnd;
  static constexpr uint32_t kEntriesPerSign = kHi - kLo + 1;
  static constexpr uint32_t kSignStride = kStride;
  static_assert(kEntriesPerSign <= kSignStride, "The two signs' entries must not overlap.");
  //! Rounded up to 16 bytes, the granularity of a bulk copy into shared memory.
  static constexpr size_t kBytes = ((kSignStride + kEntriesPerSign) * sizeof(float) + 15) / 16 * 16;
  //! The sign bit, 0x8000, shifted right by this is the byte offset of the negative half.
  static constexpr int kSignShift = kStride == 2048 ? 2 : 1;
  static_assert(kStride == 2048 || kStride == 4096, "The packed lookup assumes these strides.");
};

template <Kind kKind>
struct Table;

//! gelu(x) is RN(x/2) below 0x3b4b, x from 2.75 and -0 from -5.1875.
template <>
struct Table<kGelu> : ForwardLayout<0x3b4b, 0x40a5> {
  __device__ static float op(const float x) { return gelu<float, float>(x, {}); }
};

//! silu(x) is RN(x/2) below 0x3bb6, x from 5.96875 and -0 from -89.
template <>
struct Table<kSilu> : ForwardLayout<0x3bb6, 0x42b1> {
  __device__ static float op(const float x) { return silu<float, float>(x, {}); }
};

//! [2^-12, 8]. dgelu has saturated, to 1 and to +0, by |x| = 8, until 0.107*x*x overflows at
//! 0x6044.
template <>
struct Table<kDgelu> : DerivativeLayout<0x3980, 0x4100, 0x6044> {
  __device__ static float op(const float x) { return dgelu<float, float>(x, {}); }
  //! dgelu(x) = 0.5 + sqrt(2/pi)*x + O(x^3).
  __device__ static float below_window(const float x) { return fmaf(0.79788456f, x, 0.5f); }
};

//! [2^-9, 89]. dsilu has saturated, to 1 and to +0, by |x| = 89, up to inf.
template <>
struct Table<kDsilu> : DerivativeLayout<0x3B00, 0x42B2, 0x7F80> {
  __device__ static float op(const float x) { return dsilu<float, float>(x, {}); }
  //! dsilu(x) = 0.5 + x/2 + O(x^3).
  __device__ static float below_window(const float x) { return fmaf(0.5f, x, 0.5f); }
};

//! dSiLU over [2^-25, 89]: below 2^-25 the linear term already equals util/math.h. Selected
//! with NVTE_ACTIVATION_TABLE_EXACT_DSILU=1.
struct ExactDsiluTable : DerivativeLayout<0x3300, 0x42B2, 0x7F80, 4096> {
  __device__ static float op(const float x) { return dsilu<float, float>(x, {}); }
  __device__ static float below_window(const float x) { return fmaf(0.5f, x, 0.5f); }
};

//! Table size in bytes, in the default layout.
constexpr size_t table_bytes(const Kind kind) {
  switch (kind) {
    case kGelu:
      return Table<kGelu>::kBytes;
    case kSilu:
      return Table<kSilu>::kBytes;
    case kDgelu:
      return Table<kDgelu>::kBytes;
    case kDsilu:
      return Table<kDsilu>::kBytes;
    default:
      return 0;
  }
}

//! Whether dSiLU tables are built in the exact layout (NVTE_ACTIVATION_TABLE_EXACT_DSILU=1).
inline bool use_exact_dsilu_table() {
  static const bool exact = getenv<bool>("NVTE_ACTIVATION_TABLE_EXACT_DSILU");
  return exact;
}

//! Size of the table that device_table builds for kind.
inline size_t built_table_bytes(const Kind kind) {
  if (kind == kDsilu && use_exact_dsilu_table()) {
    return ExactDsiluTable::kBytes;
  }
  return table_bytes(kind);
}

//! Which table layout a kernel reads.
enum TableUse : int {
  kNoTable = 0,          //!< Compute OP with util/math.h.
  kDefaultTable = 1,     //!< Table<kind>.
  kExactDsiluTable = 2,  //!< ExactDsiluTable.
};

//! An activation table passed to a kernel.
struct TableRef {
  const void *ptr = nullptr;
  TableUse use = kNoTable;
};

/*! \brief The current device's table for kind, built on first use.
 *
 *  Each table is built once per device and process, then kept until the process exits
 *  (under 64 KB per device for all of them). The first call may come while the caller's
 *  stream is being captured into a CUDA graph: the table is then allocated and built outside
 *  the capture, and is ready before this returns. Thread-safe.
 */
TableRef device_table(Kind kind);

//! Whether kernels read the tables (NVTE_ACTIVATION_TABLE, default 1). Read on every call.
inline bool tables_enabled() { return getenv<bool>("NVTE_ACTIVATION_TABLE", true); }

/*! \brief The table kind for OP, if OP is read from a table. */
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr bool kHasTable = false;
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr Kind kKindOf = kGelu;

template <>
constexpr bool kHasTable<Empty, gelu<float, float>> = true;
template <>
constexpr bool kHasTable<Empty, silu<float, float>> = true;
template <>
constexpr bool kHasTable<Empty, dgelu<float, float>> = true;
template <>
constexpr bool kHasTable<Empty, dsilu<float, float>> = true;
template <>
constexpr Kind kKindOf<Empty, silu<float, float>> = kSilu;
template <>
constexpr Kind kKindOf<Empty, dgelu<float, float>> = kDgelu;
template <>
constexpr Kind kKindOf<Empty, dsilu<float, float>> = kDsilu;

/*! \brief Forward activation of x, a BF16 value widened to FP32.
 *
 *  Branch-free: the magnitude is clamped to the table rows with __hmax/__hmin, which
 *  ignore NaN, so NaN lands on the -inf row; __hmin_nan/__hmax_nan then propagate it.
 */
template <typename T>
__device__ __forceinline__ float forward(const float x, const __nv_bfloat16 *__restrict__ table) {
  static_assert(T::kForward, "Not a forward table.");
  const __nv_bfloat16 a = __float2bfloat16_rn(x);
  const uint32_t sign = __float_as_uint(x) >> 31;
  const __nv_bfloat16 magnitude =
      __hmin(__hmax(__habs(a), __ushort_as_bfloat16(static_cast<uint16_t>(T::kLowSentinel))),
             __ushort_as_bfloat16(static_cast<uint16_t>(T::kHighSentinel)));
  const uint32_t row = __bfloat16_as_ushort(magnitude) - T::kLowSentinel;
  const __nv_bfloat16 entry = __ldg(table + 2 * row + sign);
  // RN(x/2), as util/math.h rounds x*0.5 below the window.
  const __nv_bfloat16 half = __hmul(a, __ushort_as_bfloat16(0x3f00));
  // a for a > 0, -0 for a < 0, NaN for NaN.
  const __nv_bfloat16 upper = __hmax_nan(a, __ushort_as_bfloat16(0x8000));
  return __bfloat162float(__hmin_nan(__hmax(entry, half), upper));
}

/*! \brief Derivative of x, a BF16 value widened to FP32. Branch-free. */
template <typename T>
__device__ __forceinline__ float derivative(const float x, const float *__restrict__ table) {
  static_assert(!T::kForward, "Not a derivative table.");
  const uint32_t bits = __float_as_uint(x) >> 16;
  const uint32_t magnitude = bits & 0x7fffu;
  const uint32_t clamped = min(max(magnitude, T::kLoBits), T::kHiBits);
  const uint32_t index = (clamped - T::kLoBits) + (bits >> 15) * T::kSignStride;
  const float entry = __ldg(table + index);
  const float value = magnitude < T::kLoBits ? T::below_window(x) : entry;
  return magnitude >= T::kSaturatedEndBits ? __uint_as_float(0x7fffffffu) : value;
}

/*! \brief Forward activation of a BF16 pair, low lane in the low half. Same results as
 *         forward() per lane; the table may be in shared memory.
 */
template <typename T>
__device__ __forceinline__ uint32_t forward_2x(const uint32_t x,
                                               const __nv_bfloat16 *__restrict__ table) {
  static_assert(T::kForward, "Not a forward table.");
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162 *>(&x);
  // Clamp one lane's magnitude to the table rows. The +0 bound of the high lane zeroes it,
  // so the word is exactly the clamped magnitude.
  const auto row = [](const __nv_bfloat162 lane) {
    constexpr uint32_t kLowBound = T::kLowSentinel * 0x10001u;
    constexpr uint32_t kHighBound = T::kHighSentinel;
    __nv_bfloat162 m =
        __hmax2(__habs2(lane), *reinterpret_cast<const __nv_bfloat162 *>(&kLowBound));
    m = __hmin2(m, *reinterpret_cast<const __nv_bfloat162 *>(&kHighBound));
    return *reinterpret_cast<const uint32_t *>(&m) - T::kLowSentinel;
  };
  const __nv_bfloat16 entry_lo = table[2 * row(__low2bfloat162(a)) + ((x >> 15) & 1u)];
  const __nv_bfloat16 entry_hi = table[2 * row(__high2bfloat162(a)) + (x >> 31)];
  constexpr uint32_t kHalf = 0x3f003f00u, kNegativeZero = 0x80008000u;
  const __nv_bfloat162 half = __hmul2(a, *reinterpret_cast<const __nv_bfloat162 *>(&kHalf));
  const __nv_bfloat162 upper =
      __hmax2_nan(a, *reinterpret_cast<const __nv_bfloat162 *>(&kNegativeZero));
  const __nv_bfloat162 r =
      __hmin2_nan(__hmax2(__halves2bfloat162(entry_lo, entry_hi), half), upper);
  return *reinterpret_cast<const uint32_t *>(&r);
}

/*! \brief Derivative of a BF16 pair, low lane in .x. Same results as derivative() per lane;
 *         the table may be in shared memory.
 *
 *  Both lanes' byte offsets are formed in one 32-bit word: the clamp runs on the BF16 pair,
 *  and each lane's offset stays below 0x10000, so the halves never carry into each other.
 */
template <typename T>
__device__ __forceinline__ float2 derivative_2x(const uint32_t x,
                                                const unsigned char *__restrict__ table) {
  static_assert(!T::kForward, "Not a derivative table.");
  static_assert((T::kSignStride + T::kEntriesPerSign) * sizeof(float) <= 0x10000,
                "Byte offsets must fit in 16 bits.");
  constexpr uint32_t kLo2 = T::kLoBits * 0x10001u, kHi2 = T::kHiBits * 0x10001u;
  const uint32_t magnitude = x & 0x7fff7fffu;
  const __nv_bfloat162 clamped =
      __hmin2(__hmax2(*reinterpret_cast<const __nv_bfloat162 *>(&magnitude),
                      *reinterpret_cast<const __nv_bfloat162 *>(&kLo2)),
              *reinterpret_cast<const __nv_bfloat162 *>(&kHi2));
  const uint32_t offsets = ((*reinterpret_cast<const uint32_t *>(&clamped) - kLo2) << 2) +
                           ((x ^ magnitude) >> T::kSignShift);
  const float entry_lo = *reinterpret_cast<const float *>(table + (offsets & 0xffffu));
  const float entry_hi = *reinterpret_cast<const float *>(table + (offsets >> 16));
  const float linear_lo = T::below_window(__uint_as_float(x << 16));
  const float linear_hi = T::below_window(__uint_as_float(x & 0xffff0000u));
  constexpr uint32_t kSaturatedEnd2 = T::kSaturatedEndBits * 0x10001u;
  float lo, hi;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // One BF16x2 compare per condition sets both lanes' predicates. The unordered >= also
  // holds for NaN, which gives NaN like the saturated end.
  asm("{\n\t"
      ".reg .pred below_lo, below_hi, nan_lo, nan_hi;\n\t"
      "setp.lt.bf16x2 below_lo|below_hi, %2, %3;\n\t"
      "setp.geu.bf16x2 nan_lo|nan_hi, %2, %4;\n\t"
      "selp.f32 %0, %5, %6, below_lo;\n\t"
      "selp.f32 %1, %7, %8, below_hi;\n\t"
      "selp.b32 %0, 0x7fffffff, %0, nan_lo;\n\t"
      "selp.b32 %1, 0x7fffffff, %1, nan_hi;\n\t"
      "}"
      : "=f"(lo), "=f"(hi)
      : "r"(magnitude), "r"(kLo2), "r"(kSaturatedEnd2), "f"(linear_lo), "f"(entry_lo),
        "f"(linear_hi), "f"(entry_hi));
#else
  const uint32_t magnitude_lo = magnitude & 0xffffu, magnitude_hi = magnitude >> 16;
  lo = magnitude_lo < T::kLoBits ? linear_lo : entry_lo;
  hi = magnitude_hi < T::kLoBits ? linear_hi : entry_hi;
  lo = magnitude_lo >= T::kSaturatedEndBits ? __uint_as_float(0x7fffffffu) : lo;
  hi = magnitude_hi >= T::kSaturatedEndBits ? __uint_as_float(0x7fffffffu) : hi;
#endif
  return make_float2(lo, hi);
}

/*! \brief OP(x), from the table given by kUse, or from util/math.h for kNoTable.
 *
 *  x must be a BF16 value widened to FP32 when reading a table.
 */
template <int kUse, typename ParamOP, float (*OP)(float, const ParamOP &)>
__device__ __forceinline__ float evaluate(const float x, const void *table) {
  if constexpr (kUse == kNoTable) {
    return OP(x, {});
  } else {
    static_assert(kHasTable<ParamOP, OP>, "OP has no table.");
    constexpr Kind kKind = kKindOf<ParamOP, OP>;
    if constexpr (kUse == kExactDsiluTable) {
      static_assert(kKind == kDsilu, "Only dSiLU has an exact table.");
      return derivative<ExactDsiluTable>(x, static_cast<const float *>(table));
    } else if constexpr (Table<kKind>::kForward) {
      return forward<Table<kKind>>(x, static_cast<const __nv_bfloat16 *>(table));
    } else {
      return derivative<Table<kKind>>(x, static_cast<const float *>(table));
    }
  }
}

}  // namespace activation_table
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_UTIL_ACTIVATION_TABLE_CUH_
