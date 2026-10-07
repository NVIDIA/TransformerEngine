/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The fused MXFP8 activation quantize must round exactly as if the activation were computed per
// element with util/math.h, truncated to the input type, and then quantized on its own. This
// compares the fused kernels bit for bit against that composition: the activation is computed on
// the GPU with the scalar util/math.h functions, and the result is quantized by the generic
// cast-only kernel (a noop tensor keeps the call off the specialized cast kernels). dbias is
// compared against a host sum in the order the kernel accumulates it.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"
#include "util/math.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class Mode { GeLU, DGeLU, DBiasDGeLU, SiLU, DSiLU, DBiasDSiLU };

bool is_dbias(const Mode m) { return m == Mode::DBiasDGeLU || m == Mode::DBiasDSiLU; }

const char *mode_name(const Mode m) {
  switch (m) {
    case Mode::GeLU:
      return "GeLU";
    case Mode::DGeLU:
      return "DGeLU";
    case Mode::DBiasDGeLU:
      return "DBiasDGeLU";
    case Mode::SiLU:
      return "SiLU";
    case Mode::DSiLU:
      return "DSiLU";
    default:
      return "DBiasDSiLU";
  }
}

// act: y = OP(x); dact: y = grad * dOP(x), as the fused kernels form it in FP32. z keeps the FP32
// value (dbias sums it before truncation), y the value truncated to the input type.
template <typename IType>
__global__ void scalar_activation_kernel(const Mode mode, const IType *x, const IType *grad,
                                         IType *y, float *z, const size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float xv = static_cast<float>(x[i]);
  const Empty e{};
  float v;
  switch (mode) {
    case Mode::GeLU:
      v = gelu<float, float>(xv, e);
      break;
    case Mode::SiLU:
      v = silu<float, float>(xv, e);
      break;
    case Mode::DGeLU:
    case Mode::DBiasDGeLU:
      v = static_cast<float>(grad[i]);
      v *= dgelu<float, float>(xv, e);
      break;
    default:
      v = static_cast<float>(grad[i]);
      v *= dsilu<float, float>(xv, e);
      break;
  }
  z[i] = v;
  y[i] = static_cast<IType>(v);
}

enum class Fill { Typical, OrderedInputs, Specials };

template <typename T>
T from_bits(const uint32_t bits) {
  if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<const float &>(bits);
  } else {
    const uint16_t half_bits = static_cast<uint16_t>(bits);
    return reinterpret_cast<const T &>(half_bits);
  }
}

// Enumerate the entire 16-bit format in numerical order: -Inf, negative finite values, -0,
// +0, positive finite values, +Inf, then both signs of NaN (which have no numerical order).
// For FP32, enumerate the BF16 exponent/mantissa prefixes and vary the low mantissa bits.
template <typename T>
uint32_t ordered_bits(const size_t i, const size_t n) {
  // The high 16 bits select a format encoding; the low 16 sample FP32 mantissas.
  const uint32_t position = static_cast<uint32_t>((static_cast<uint64_t>(i) << 32) / n);
  const uint32_t rank = position >> 16;
  const uint32_t pos_inf = std::is_same_v<T, fp16> ? 0x7c00u : 0x7f80u;
  const uint32_t non_nan_count = 2 * (pos_inf + 1);
  uint32_t high;
  if (rank <= pos_inf) {
    high = 0x8000u | (pos_inf - rank);
  } else if (rank < non_nan_count) {
    high = rank - pos_inf - 1;
  } else {
    const uint32_t nan_rank = rank - non_nan_count;
    const uint32_t nan_count_per_sign = 0x7fffu - pos_inf;
    high = nan_rank < nan_count_per_sign
               ? 0x8000u | (pos_inf + 1 + nan_rank)
               : pos_inf + 1 + nan_rank - nan_count_per_sign;
  }
  if constexpr (std::is_same_v<T, float>) {
    const uint32_t fraction = position & 0xffffu;
    const uint32_t low = (high & 0x8000u) ? 0xffffu - fraction : fraction;
    return (high << 16) | ((high & 0x7fffu) == pos_inf ? 0u : low);
  } else {
    return high;
  }
}

// Exactly representable edge cases in each input format, ordered except for the final NaNs.
template <typename T>
std::array<T, 14> special_values() {
  const float inf = Numeric_Traits<T>::artifInf;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float max_normal = Numeric_Traits<T>::maxNorm;
  const float min_normal = Numeric_Traits<T>::minNorm;
  const float min_subnormal = Numeric_Traits<T>::minSubnorm;
  return {static_cast<T>(-inf),           static_cast<T>(-max_normal),
          static_cast<T>(-1.0f),          static_cast<T>(-min_normal),
          static_cast<T>(-min_subnormal), static_cast<T>(-0.0f),
          static_cast<T>(0.0f),           static_cast<T>(min_subnormal),
          static_cast<T>(min_normal),     static_cast<T>(1.0f),
          static_cast<T>(max_normal),     static_cast<T>(inf),
          static_cast<T>(-nan),           static_cast<T>(nan)};
}

// Typical activations range over [-16, 16] and gradients over [-4, 4]. The ordered sweep
// covers every BF16/FP16 bit pattern, including subnormals, infinities, and NaNs. The special
// case set guarantees exact FP32 boundary values that a sampled mantissa sweep may miss.
template <typename T>
void fill_inputs(const Fill fill, Tensor *x, Tensor *grad) {
  const size_t n = product(x->rowwise_shape());
  T *xp = x->rowwise_cpu_dptr<T>();
  T *gp = grad->rowwise_cpu_dptr<T>();
  const auto specials = special_values<T>();
  for (size_t i = 0; i < n; ++i) {
    switch (fill) {
      case Fill::Typical:
        xp[i] = static_cast<T>(-16.0f + 32.0f * static_cast<float>(i) / (n - 1));
        gp[i] = static_cast<T>(-4.0f + 8.0f * static_cast<float>(i % 257) / 256.0f);
        break;
      case Fill::OrderedInputs:
        xp[i] = from_bits<T>(ordered_bits<T>(i, n));
        gp[i] = static_cast<T>(-4.0f + 8.0f * static_cast<float>(i % 257) / 256.0f);
        break;
      case Fill::Specials: {
        const size_t index = i * specials.size() / n;
        xp[i] = specials[index];
        gp[i] = specials[(index + 4) % specials.size()];
        break;
      }
    }
  }
  x->from_cpu();
  grad->from_cpu();
}

// dbias in the order the quantize kernel accumulates it: per 64-row tile, then over the tiles.
// With FP32 input, rowwise-only scaling and a fused dact, each of the tile's 32 threads first
// adds its two rows (r and r + 32), and the 32 thread sums are then added in order.
std::vector<float> reference_dbias(const std::vector<float> &z, const size_t rows,
                                   const size_t cols, const bool rowwise_reduction) {
  constexpr size_t kTileRows = 64;
  constexpr size_t kThreadsY = 32;
  const size_t tiles = (rows + kTileRows - 1) / kTileRows;
  auto at = [&](const size_t r, const size_t c) { return r < rows ? z[r * cols + c] : 0.0f; };
  std::vector<float> dbias(cols);
  for (size_t c = 0; c < cols; ++c) {
    float total = 0.0f;
    for (size_t t = 0; t < tiles; ++t) {
      const size_t r0 = t * kTileRows;
      float partial = 0.0f;
      if (rowwise_reduction) {
        for (size_t i = 0; i < kThreadsY; ++i) {
          float thread_sum = 0.0f;
          thread_sum += at(r0 + i, c);
          thread_sum += at(r0 + kThreadsY + i, c);
          partial += thread_sum;
        }
      } else {
        for (size_t r = r0; r < r0 + kTileRows; ++r) {
          partial += at(r, c);
        }
      }
      total += partial;
    }
    dbias[c] = total;
  }
  return dbias;
}

size_t count_byte_mismatches(const uint8_t *a, const uint8_t *b, const size_t n) {
  size_t m = 0;
  for (size_t i = 0; i < n; ++i) m += (a[i] != b[i]);
  return m;
}

template <typename IType>
void run_case(const Mode mode, const Fill fill, const size_t rows, const size_t cols,
              const bool rowwise, const bool colwise, const bool swizzled) {
  const DType itype = TypeInfo<IType>::dtype;
  const DType otype = DType::kFloat8E4M3;
  const std::vector<size_t> shape{rows, cols};
  const size_t n = rows * cols;

  Tensor x("x", shape, itype);
  Tensor grad("grad", shape, itype);
  fill_inputs<IType>(fill, &x, &grad);

  // Reference: scalar activation on the GPU, then the generic cast-only quantize.
  Tensor y("y", shape, itype);
  float *z_dev = nullptr;
  ASSERT_EQ(cudaMalloc(&z_dev, n * sizeof(float)), cudaSuccess);
  constexpr int block_size = 256;
  const size_t grid_size = divide_round_up(n, block_size);
  scalar_activation_kernel<IType>
      <<<grid_size, block_size>>>(mode, static_cast<const IType *>(x.rowwise_dptr()),
                                 static_cast<const IType *>(grad.rowwise_dptr()),
                                 static_cast<IType *>(y.rowwise_dptr()), z_dev, n);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::vector<float> z(n);
  ASSERT_EQ(cudaMemcpy(z.data(), z_dev, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  cudaFree(z_dev);

  Tensor ref("ref", shape, otype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);
  ref.set_with_gemm_swizzled_scales(swizzled);
  Tensor noop("noop", std::vector<size_t>{1}, DType::kFloat32);
  nvte_quantize_noop(y.data(), ref.data(), noop.data(), 0);

  // Fused.
  Tensor out("out", shape, otype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);
  out.set_with_gemm_swizzled_scales(swizzled);
  Tensor dbias("dbias", std::vector<size_t>{cols}, itype);
  Tensor workspace;
  switch (mode) {
    case Mode::GeLU:
      nvte_gelu(x.data(), out.data(), 0);
      break;
    case Mode::SiLU:
      nvte_silu(x.data(), out.data(), 0);
      break;
    case Mode::DGeLU:
      nvte_dgelu(grad.data(), x.data(), out.data(), 0);
      break;
    case Mode::DSiLU:
      nvte_dsilu(grad.data(), x.data(), out.data(), 0);
      break;
    default: {
      auto fn = mode == Mode::DBiasDGeLU ? &nvte_quantize_dbias_dgelu : &nvte_quantize_dbias_dsilu;
      fn(grad.data(), x.data(), out.data(), dbias.data(), workspace.data(), 0);
      workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
      fn(grad.data(), x.data(), out.data(), dbias.data(), workspace.data(), 0);
    }
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  out.to_cpu();
  ref.to_cpu();
  if (rowwise) {
    EXPECT_EQ(count_byte_mismatches(
                  reinterpret_cast<const uint8_t *>(out.rowwise_cpu_dptr<fp8e4m3>()),
                  reinterpret_cast<const uint8_t *>(ref.rowwise_cpu_dptr<fp8e4m3>()), n),
              0u)
        << "rowwise data";
    const size_t s = product(out.rowwise_scale_inv_shape());
    EXPECT_EQ(count_byte_mismatches(out.rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                    ref.rowwise_cpu_scale_inv_ptr<uint8_t>(), s),
              0u)
        << "rowwise scales";
  }
  if (colwise) {
    EXPECT_EQ(count_byte_mismatches(
                  reinterpret_cast<const uint8_t *>(out.columnwise_cpu_dptr<fp8e4m3>()),
                  reinterpret_cast<const uint8_t *>(ref.columnwise_cpu_dptr<fp8e4m3>()), n),
              0u)
        << "colwise data";
    const size_t s = product(out.columnwise_scale_inv_shape());
    EXPECT_EQ(count_byte_mismatches(out.columnwise_cpu_scale_inv_ptr<uint8_t>(),
                                    ref.columnwise_cpu_scale_inv_ptr<uint8_t>(), s),
              0u)
        << "colwise scales";
  }
  if (is_dbias(mode)) {
    const bool rowwise_reduction = std::is_same_v<IType, float> && !colwise;
    const std::vector<float> expected = reference_dbias(z, rows, cols, rowwise_reduction);
    dbias.to_cpu();
    const IType *got = dbias.rowwise_cpu_dptr<IType>();
    size_t mismatches = 0;
    for (size_t c = 0; c < cols; ++c) {
      const IType e = static_cast<IType>(expected[c]);
      mismatches += std::memcmp(&e, &got[c], sizeof(IType)) != 0 &&
                    !(std::isnan(static_cast<float>(e)) &&
                      std::isnan(static_cast<float>(got[c])));
    }
    EXPECT_EQ(mismatches, 0u) << "dbias";
  }
}

using Params = std::tuple<Mode, Fill, DType, int /*layout: 0 row, 1 col, 2 both*/,
                          bool /*swizzled*/, std::pair<size_t, size_t>>;

class CastMXFP8ActExactTestSuite : public ::testing::TestWithParam<Params> {};

}  // namespace

TEST_P(CastMXFP8ActExactTestSuite, MatchesScalarActivationThenCast) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 10) {
    GTEST_SKIP() << "This MXFP8 quantize kernel requires compute capability 10.0 or newer";
  }
  const auto [mode, fill, itype, layout, swizzled, dims] = GetParam();
  const bool rowwise = layout != 1;
  const bool colwise = layout != 0;
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
      itype, IType,
      run_case<IType>(mode, fill, dims.first, dims.second, rowwise, colwise, swizzled););
}

namespace {

std::string case_name(const testing::TestParamInfo<Params> &info) {
  const char *layouts[] = {"Rowwise", "Colwise", "Both"};
  const auto [mode, fill, itype, layout, swizzled, dims] = info.param;
  const char *fills[] = {"XTypical", "XOrderedInputs", "XSpecials"};
  return std::string(mode_name(mode)) + fills[static_cast<int>(fill)] +
         "X" + test::typeName(itype) + "X" + layouts[layout] +
         (swizzled ? "XSwizzled" : "XCompact") + "X" + std::to_string(dims.first) + "X" +
         std::to_string(dims.second);
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, CastMXFP8ActExactTestSuite,
    ::testing::Combine(::testing::Values(Mode::GeLU, Mode::DGeLU, Mode::DBiasDGeLU, Mode::SiLU,
                                         Mode::DSiLU, Mode::DBiasDSiLU),
                       ::testing::Values(Fill::Typical, Fill::OrderedInputs),
                       ::testing::Values(DType::kBFloat16, DType::kFloat16, DType::kFloat32),
                       ::testing::Values(0, 1, 2), ::testing::Bool(),
                       ::testing::Values(std::make_pair<size_t, size_t>(1024, 2048),
                                         std::make_pair<size_t, size_t>(544, 2080))),
    case_name);

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, CastMXFP8ActExactTestSuite,
    ::testing::Combine(::testing::Values(Mode::GeLU, Mode::DGeLU, Mode::DBiasDGeLU, Mode::SiLU,
                                         Mode::DSiLU, Mode::DBiasDSiLU),
                       ::testing::Values(Fill::Specials),
                       ::testing::Values(DType::kBFloat16, DType::kFloat16, DType::kFloat32),
                       ::testing::Values(0, 1, 2), ::testing::Bool(),
                       ::testing::Values(std::make_pair<size_t, size_t>(64, 128))),
    case_name);
