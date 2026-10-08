/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file rmsnorm_fwd_mxfp8.cu
 *  \brief RMSNorm forward fused with MXFP8 quantization.
 *
 *  Normalizes in FP32 and quantizes the FP32 result, row-wise and/or column-wise, with
 *  the scaling factors written in compact or GEMM-swizzled layout. The output is that of
 *  the MXFP8 quantizer applied to the FP32 normalized values, which is also what cuDNN's
 *  fused kernel produces; the two can differ only through the rounding of rsigma.
 *
 *  Two kernels: a pre-pass computes rsigma per row (one warp per row), then one CTA per
 *  32x128 tile normalizes the tile, quantizes it and writes its scaling factors. A tile
 *  only sees 128 columns of a row, so it cannot reduce the row itself; doing so with whole
 *  rows per CTA would limit the grid to rows / 32 CTAs.
 */

#include <cuda_runtime.h>

#include <limits>
#include <type_traits>

#include "../../cast/mxfp8/swizzle.cuh"
#include "../../common.h"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"
#include "../common.h"

namespace transformer_engine {
namespace normalization {
namespace {

constexpr int kScaleBlock = 32;  // MXFP8 scaling block size.
constexpr int kTileRows = 32;    // One column-wise scaling block.
constexpr int kTileCols = 128;   // Four row-wise scaling blocks.
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / THREADS_PER_WARP;
constexpr int kRowsPerWarp = kTileRows / kWarps;
constexpr int kColsPerLane = kTileCols / THREADS_PER_WARP;
constexpr int kLanesPerBlock = kScaleBlock / kColsPerLane;
constexpr int kRsigmaRowsPerCTA = 4;

static_assert(kRowsPerWarp == 8);
static_assert(kColsPerLane == 4);
static_assert(kLanesPerBlock == 8);

template <typename T, int N>
__device__ __forceinline__ void load_elements(T (&dst)[N], const T *src) {
  if constexpr (sizeof(T) * N == 16) {
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
  } else if constexpr (sizeof(T) * N == 8) {
    *reinterpret_cast<uint2 *>(dst) = *reinterpret_cast<const uint2 *>(src);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = src[i];
    }
  }
}

// rsigma = 1 / sqrt(mean(x^2) + epsilon), one warp per row.
template <typename IType, bool kAligned>
__global__ void __launch_bounds__(kRsigmaRowsPerCTA *THREADS_PER_WARP)
    rmsnorm_mxfp8_rsigma_kernel(const IType *const x, float *const rsigma, const int rows,
                                const int cols, const float epsilon) {
  const int row = blockIdx.x * kRsigmaRowsPerCTA + threadIdx.x / THREADS_PER_WARP;
  const int lane = threadIdx.x % THREADS_PER_WARP;
  if (row >= rows) {
    return;
  }
  const IType *const x_row = x + static_cast<size_t>(row) * cols;
  float sum = 0.0f;
  if constexpr (kAligned) {
    constexpr int kVec = 16 / sizeof(IType);
    for (int c = lane * kVec; c < cols; c += THREADS_PER_WARP * kVec) {
      IType values[kVec];
      load_elements(values, x_row + c);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        const float value = static_cast<float>(values[i]);
        sum += value * value;
      }
    }
  } else {
    for (int c = lane; c < cols; c += THREADS_PER_WARP) {
      const float value = static_cast<float>(x_row[c]);
      sum += value * value;
    }
  }
#pragma unroll
  for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) {
    rsigma[row] = 1.0f / sqrtf(sum / static_cast<float>(cols) + epsilon);
  }
}

// Four FP32 values times per-value multipliers, converted to four FP8 values.
template <typename OType>
__device__ __forceinline__ uint32_t quantize_4x(const float (&values)[kColsPerLane],
                                                const float (&multipliers)[kColsPerLane]) {
  using OTypex2 =
      std::conditional_t<std::is_same_v<OType, fp8e4m3>, ptx::fp8e4m3x2, ptx::fp8e5m2x2>;
  OTypex2 out[2];
  ptx::mul_cvt_2x(out[0], ptx::floatx2{values[0], values[1]},
                  ptx::floatx2{multipliers[0], multipliers[1]});
  ptx::mul_cvt_2x(out[1], ptx::floatx2{values[2], values[3]},
                  ptx::floatx2{multipliers[2], multipliers[3]});
  return static_cast<uint32_t>(reinterpret_cast<const uint16_t &>(out[0])) |
         (static_cast<uint32_t>(reinterpret_cast<const uint16_t &>(out[1])) << 16);
}

struct QuantizeArgs {
  void *rowwise_data;
  e8m0_t *rowwise_scale_inv;
  void *colwise_data;
  e8m0_t *colwise_scale_inv;
  // Compact layouts: row strides of the scaling-factor tensors. Swizzled layouts: number of
  // 4-wide scaling-factor tiles along the blocked dimension.
  size_t rowwise_scale_stride;
  size_t colwise_scale_stride;
};

// One CTA per 32x128 tile: warp w normalizes rows 8w..8w+7, lane l columns 4l..4l+3.
template <typename IType, typename WType, typename OType, bool kRowwise, bool kColwise,
          bool kSwizzled, bool kAligned>
__global__ void __launch_bounds__(kThreads)
    rmsnorm_mxfp8_quantize_kernel(const IType *const x, const WType *const gamma,
                                  const float *const rsigma, const QuantizeArgs args,
                                  const int cols, const bool zero_centered_gamma,
                                  const bool gamma_in_weight_dtype) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;
  constexpr float kMaxNormRcp = Quantized_Limits<OType>::max_norm_rcp;
  __shared__ float colwise_amax[kWarps][kTileCols];

  const int warp = threadIdx.x / THREADS_PER_WARP;
  const int lane = threadIdx.x % THREADS_PER_WARP;
  const int row_base = blockIdx.x * kTileRows;
  const int row0 = row_base + warp * kRowsPerWarp;
  const int col0 = blockIdx.y * kTileCols + lane * kColsPerLane;

  // Gamma as applied by the other normalization backends: with a zero-centered gamma, 1 is
  // added in FP32, or in the weight type if requested.
  float g[kColsPerLane];
  {
    WType w[kColsPerLane];
    if constexpr (kAligned) {
      load_elements(w, gamma + col0);
    } else {
#pragma unroll
      for (int i = 0; i < kColsPerLane; ++i) {
        w[i] = gamma[col0 + i];
      }
    }
#pragma unroll
    for (int i = 0; i < kColsPerLane; ++i) {
      float value = static_cast<float>(w[i]);
      if (zero_centered_gamma) {
        value = gamma_in_weight_dtype ? static_cast<float>(static_cast<WType>(value + 1.0f))
                                      : value + 1.0f;
      }
      g[i] = value;
    }
  }

  float y[kRowsPerWarp][kColsPerLane];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int row = row0 + r;
    const float rs = rsigma[row];
    IType values[kColsPerLane];
    const IType *const src = x + static_cast<size_t>(row) * cols + col0;
    if constexpr (kAligned) {
      load_elements(values, src);
    } else {
#pragma unroll
      for (int i = 0; i < kColsPerLane; ++i) {
        values[i] = src[i];
      }
    }
#pragma unroll
    for (int i = 0; i < kColsPerLane; ++i) {
      y[r][i] = (static_cast<float>(values[i]) * rs) * g[i];
    }
  }

  if constexpr (kRowwise) {
    OType *const data = reinterpret_cast<OType *>(args.rowwise_data);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int row = row0 + r;
      // Each 32-column scaling block spans 8 consecutive lanes.
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kColsPerLane; ++i) {
        amax = fmaxf(amax, fabsf(y[r][i]));
      }
#pragma unroll
      for (int offset = 1; offset < kLanesPerBlock; offset *= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
      }
      const e8m0_t biased_exponent = ptx::float_to_e8m0(amax * kMaxNormRcp);
      const float multiplier = ptx::exp2f_rcp<float>(biased_exponent);
      const float multipliers[kColsPerLane] = {multiplier, multiplier, multiplier, multiplier};
      const uint32_t quantized = quantize_4x<OType>(y[r], multipliers);
      OType *const dst = data + static_cast<size_t>(row) * cols + col0;
      if constexpr (kAligned) {
        *reinterpret_cast<uint32_t *>(dst) = quantized;
      } else {
#pragma unroll
        for (int i = 0; i < kColsPerLane; ++i) {
          reinterpret_cast<uint8_t *>(dst)[i] = static_cast<uint8_t>(quantized >> (8 * i));
        }
      }
      if (lane % kLanesPerBlock == 0) {
        const size_t scale_col = col0 / kScaleBlock;
        const size_t idx = kSwizzled
                               ? gemm_swizzled_scale_idx(row, scale_col, args.rowwise_scale_stride)
                               : static_cast<size_t>(row) * args.rowwise_scale_stride + scale_col;
        args.rowwise_scale_inv[idx] = biased_exponent;
      }
    }
  }

  if constexpr (kColwise) {
    // Each column's scaling block is the tile's 32 rows: reduce over this warp's 8 rows, then
    // over the 4 warps through shared memory.
#pragma unroll
    for (int i = 0; i < kColsPerLane; ++i) {
      float amax = 0.0f;
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
        amax = fmaxf(amax, fabsf(y[r][i]));
      }
      colwise_amax[warp][lane * kColsPerLane + i] = amax;
    }
    __syncthreads();
    e8m0_t biased_exponents[kColsPerLane];
    float multipliers[kColsPerLane];
#pragma unroll
    for (int i = 0; i < kColsPerLane; ++i) {
      float amax = colwise_amax[0][lane * kColsPerLane + i];
#pragma unroll
      for (int w = 1; w < kWarps; ++w) {
        amax = fmaxf(amax, colwise_amax[w][lane * kColsPerLane + i]);
      }
      biased_exponents[i] = ptx::float_to_e8m0(amax * kMaxNormRcp);
      multipliers[i] = ptx::exp2f_rcp<float>(biased_exponents[i]);
    }
    OType *const data = reinterpret_cast<OType *>(args.colwise_data);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const uint32_t quantized = quantize_4x<OType>(y[r], multipliers);
      OType *const dst = data + static_cast<size_t>(row0 + r) * cols + col0;
      if constexpr (kAligned) {
        *reinterpret_cast<uint32_t *>(dst) = quantized;
      } else {
#pragma unroll
        for (int i = 0; i < kColsPerLane; ++i) {
          reinterpret_cast<uint8_t *>(dst)[i] = static_cast<uint8_t>(quantized >> (8 * i));
        }
      }
    }
    if (warp == 0) {
      const size_t scale_row = row_base / kScaleBlock;
#pragma unroll
      for (int i = 0; i < kColsPerLane; ++i) {
        const size_t col = col0 + i;
        const size_t idx = kSwizzled
                               ? gemm_swizzled_scale_idx(col, scale_row, args.colwise_scale_stride)
                               : scale_row * args.colwise_scale_stride + col;
        args.colwise_scale_inv[idx] = biased_exponents[i];
      }
    }
  }
#else
  NVTE_DEVICE_THREAD0_ERROR("RMSNorm forward with MXFP8 output requires SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

bool is_aligned_to(const void *ptr, size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <typename IType, typename WType, typename OType>
void launch_rmsnorm_fwd_mxfp8(const Tensor &x, const Tensor &gamma, const float epsilon, Tensor *z,
                              Tensor *rsigma, const bool zero_centered_gamma,
                              const bool gamma_in_weight_dtype, cudaStream_t stream) {
  const auto [rows_size_t, cols_size_t] = x.flat_2d_dims();
  const int rows = static_cast<int>(rows_size_t);
  const int cols = static_cast<int>(cols_size_t);
  const bool rowwise = z->has_data();
  const bool colwise = z->has_columnwise_data();
  const bool swizzled = z->with_gemm_swizzled_scales;

  QuantizeArgs args{};
  if (rowwise) {
    args.rowwise_data = z->data.dptr;
    args.rowwise_scale_inv = reinterpret_cast<e8m0_t *>(z->scale_inv.dptr);
    const size_t scale_cols = z->scale_inv.shape.back();
    args.rowwise_scale_stride = swizzled ? scale_cols / 4 : scale_cols;
  }
  if (colwise) {
    args.colwise_data = z->columnwise_data.dptr;
    args.colwise_scale_inv = reinterpret_cast<e8m0_t *>(z->columnwise_scale_inv.dptr);
    args.colwise_scale_stride =
        swizzled ? z->columnwise_scale_inv.shape.front() / 4 : z->columnwise_scale_inv.shape.back();
  }

  const IType *const x_ptr = reinterpret_cast<const IType *>(x.data.dptr);
  const WType *const gamma_ptr = reinterpret_cast<const WType *>(gamma.data.dptr);
  float *const rsigma_ptr = reinterpret_cast<float *>(rsigma->data.dptr);

  // Vector accesses: 16 bytes of a row of x in the pre-pass, 4 elements of x and gamma and 4
  // bytes of each output in the main kernel. Every row starts aligned since cols % 128 == 0.
  const bool aligned = is_aligned_to(x_ptr, 16) && is_aligned_to(gamma_ptr, 4 * sizeof(WType)) &&
                       (!rowwise || is_aligned_to(args.rowwise_data, 4)) &&
                       (!colwise || is_aligned_to(args.colwise_data, 4));

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      aligned, kAligned,
      rmsnorm_mxfp8_rsigma_kernel<IType, kAligned>
      <<<DIVUP(rows, kRsigmaRowsPerCTA), kRsigmaRowsPerCTA * THREADS_PER_WARP, 0, stream>>>(
          x_ptr, rsigma_ptr, rows, cols, epsilon););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());

  const dim3 grid(rows / kTileRows, cols / kTileCols);
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      aligned, kAligned,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          swizzled, kSwizzled,
          if (rowwise && colwise) {
            rmsnorm_mxfp8_quantize_kernel<IType, WType, OType, true, true, kSwizzled, kAligned>
                <<<grid, kThreads, 0, stream>>>(x_ptr, gamma_ptr, rsigma_ptr, args, cols,
                                                zero_centered_gamma, gamma_in_weight_dtype);
          } else if (rowwise) {
            rmsnorm_mxfp8_quantize_kernel<IType, WType, OType, true, false, kSwizzled, kAligned>
                <<<grid, kThreads, 0, stream>>>(x_ptr, gamma_ptr, rsigma_ptr, args, cols,
                                                zero_centered_gamma, gamma_in_weight_dtype);
          } else {
            rmsnorm_mxfp8_quantize_kernel<IType, WType, OType, false, true, kSwizzled, kAligned>
                <<<grid, kThreads, 0, stream>>>(x_ptr, gamma_ptr, rsigma_ptr, args, cols,
                                                zero_centered_gamma, gamma_in_weight_dtype);
          }););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

bool use_te_rmsnorm_fwd_mxfp8(const Tensor &x, const Tensor &gamma, const Tensor &z) {
  if (!is_mxfp8_scaling(z.scaling_mode) || use_cudnn_norm_fwd_mxfp8() ||
      !is_supported_by_CC_100()) {
    return false;
  }
  const bool rowwise = z.has_data();
  const bool colwise = z.has_columnwise_data();
  if (!rowwise && !colwise) {
    return false;
  }
  if ((rowwise && z.scale_inv.shape.size() != 2) ||
      (colwise && z.columnwise_scale_inv.shape.size() != 2)) {
    return false;
  }
  const auto [rows, cols] = x.flat_2d_dims();
  // Full 128x128 tiles, so the scaling factors carry no padding in either layout.
  if (rows == 0 || cols == 0 || rows % 128 != 0 || cols % 128 != 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max()) || cols / kTileCols > 65535) {
    return false;
  }
  const auto is_supported_input = [](DType t) {
    return t == DType::kFloat32 || t == DType::kBFloat16 || t == DType::kFloat16;
  };
  if (!is_supported_input(x.data.dtype) || !is_supported_input(gamma.data.dtype)) {
    return false;
  }
  const auto is_supported_output = [](DType t) {
    return t == DType::kFloat8E4M3 || t == DType::kFloat8E5M2;
  };
  if ((rowwise && !is_supported_output(z.data.dtype)) ||
      (colwise && !is_supported_output(z.columnwise_data.dtype)) ||
      (rowwise && colwise && z.data.dtype != z.columnwise_data.dtype)) {
    return false;
  }
  return true;
}

void rmsnorm_fwd_mxfp8(const Tensor &x, const Tensor &gamma, const float epsilon, Tensor *z,
                       Tensor *rsigma, const bool zero_centered_gamma, cudaStream_t stream) {
  const DType otype = z->has_data() ? z->data.dtype : z->columnwise_data.dtype;
  const bool gamma_in_weight_dtype = use_zero_centered_gamma_in_weight_dtype();
  TRANSFORMER_ENGINE_TYPE_SWITCH_FLOAT(
      x.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FLOAT(
          gamma.data.dtype, WType,
          TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
              otype, OType,
              launch_rmsnorm_fwd_mxfp8<IType, WType, OType>(
                  x, gamma, epsilon, z, rsigma, zero_centered_gamma, gamma_in_weight_dtype,
                  stream););););  // NOLINT(*)
}

}  // namespace normalization
}  // namespace transformer_engine
