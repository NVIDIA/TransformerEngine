/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp8_spec.cuh
 *  \brief CUDA kernels to cast MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_SPECIALIZED_QUANTIZE_MXFP8_CUH_
#define TRANSFORMER_ENGINE_SPECIALIZED_QUANTIZE_MXFP8_CUH_

#include <cstdlib>

#include "../../../util/ptx.cuh"
#include "state_counter.cuh"
#include "swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

namespace ptx = transformer_engine::ptx;
namespace {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

#if defined(_ENABLE_MXFMA)
template <typename IType, typename OType>
struct _Quantized_Limits;

template <>
struct _Quantized_Limits<float, fp8e5m2> {
  static constexpr uint16_t max_norm_rcp{0};
};

template <>
struct _Quantized_Limits<float, fp8e4m3> {
  static constexpr uint16_t max_norm_rcp{0};
};

template <>
struct _Quantized_Limits<fp16, fp8e5m2> {
  static constexpr uint16_t max_norm_rcp{0x125};
};

template <>
struct _Quantized_Limits<bf16, fp8e5m2> {
  static constexpr uint16_t max_norm_rcp{0x3792};
};

template <>
struct _Quantized_Limits<fp16, fp8e4m3> {
  static constexpr uint16_t max_norm_rcp{0x1892};
};

template <>
struct _Quantized_Limits<bf16, fp8e4m3> {
  static constexpr uint16_t max_norm_rcp{0x3b12};
};
#endif  // #if defined(_ENABLE_MXFMA)

template <typename OType, typename IType>
__device__ __forceinline__ e8m0_t to_e8m0(IType amax) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (defined _ENABLE_MXFMA)
  constexpr uint16_t max_norm_rcp = _Quantized_Limits<IType, OType>::max_norm_rcp;

  float amax_fp32;
  if constexpr (std::is_same_v<IType, fp16>) {
    ptx::fma_f32_f16(amax_fp32, reinterpret_cast<uint16_t &>(amax), max_norm_rcp);
  } else if constexpr (std::is_same_v<IType, bf16>) {
    ptx::fma_f32_bf16(amax_fp32, reinterpret_cast<uint16_t &>(amax), max_norm_rcp);
  } else {
    amax_fp32 = 0.0f;
    __trap();
  }
  return ptx::float_to_e8m0(amax_fp32);
#else
  if constexpr (std::is_same_v<IType, float>) {
    return ptx::float_to_e8m0(__fmaf_ieee_rn(amax, Quantized_Limits<OType>::max_norm_rcp, 0.0f));
  } else {
    float amax_fp32 = static_cast<float>(amax);
    return ptx::float_to_e8m0(
        __fmaf_ieee_rn(amax_fp32, Quantized_Limits<OType>::max_norm_rcp, 0.0f));
  }
#endif
}

#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}  // anonymous namespace

inline bool is_cast_only_enabled() {
  static bool enabled = []() {
    const char *env = std::getenv("ENABLE_CAST_ONLY");
    return env != nullptr && (env[0] == '1');
  }();
  return enabled;

  //  // FIXME: when finish debugging, remove this
  //  const char* env = std::getenv("ENABLE_CAST_ONLY");
  //  return env != nullptr && (env[0] == '1');
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename IType, typename OType>
inline bool hasSpec() {
  return false;
}

// IType could be [fp16, bf16]
// OType could be [fp8e5m2, fp8e4m3]
template <>
inline bool hasSpec<false, false, false, fp16, fp8e5m2>() {
  return is_cast_only_enabled();
}
template <>
inline bool hasSpec<false, false, false, fp16, fp8e4m3>() {
  return is_cast_only_enabled();
}
template <>
inline bool hasSpec<false, false, false, bf16, fp8e5m2>() {
  return is_cast_only_enabled();
}
template <>
inline bool hasSpec<false, false, false, bf16, fp8e4m3>() {
  return is_cast_only_enabled();
}

template <int32_t _M, int32_t _N>
struct Layout {
  static constexpr int32_t M = _M;  // row
  static constexpr int32_t N = _N;  // col
  static constexpr int32_t num = M * N;
};

template <typename IType, typename OType, bool rowwise, bool colwise>
struct CastTraits;

// 1x32
template <typename _IType, typename _OType>
struct CastTraits<_IType, _OType, /*rowwise=*/true, /*colwise=*/false> {
  static constexpr bool isRowwise = true;
  static constexpr bool isColwise = false;
  using IType = _IType;
  using OType = _OType;

  static constexpr int32_t chunkElems = 32;
  using threadLayout = Layout<1, 32>;
  static constexpr int32_t numThreadsPerChunk = 1;
  static constexpr int32_t warpDimM = threadLayout::M;
  static constexpr int32_t warpDimN = threadLayout::N * chunkElems;
  using inputUnitType = uint4;
  static constexpr int32_t numUnitsPerChunk = chunkElems * sizeof(IType) / sizeof(inputUnitType);
  using outputUnitType = uint4;
  static constexpr int32_t numOutUnitsPerChunk =
      chunkElems * sizeof(OType) / sizeof(outputUnitType);

  using warpLayout = Layout<4, 1>;
  static constexpr int32_t blockIterDimM = warpLayout::M * warpDimM;
  static constexpr int32_t blockIterDimN = warpLayout::N * warpDimN;

  using iterLayout = Layout<1, 1>;
  static constexpr int32_t blockDimM = iterLayout::M * blockIterDimM;
  static constexpr int32_t blockDimN = iterLayout::N * blockIterDimN;

  static constexpr int32_t numStages = 1;
  static constexpr int32_t numPrefetch = numStages - 1;

  static constexpr bool _use_cvt_4x = true;
  static constexpr bool _cache_rowwise_scale_in_smem = true;

  static constexpr int32_t numThreads = warpLayout::num * 32;

  static constexpr size_t smem_rowwise_scale =
      _cache_rowwise_scale_in_smem ? (blockDimM * (blockDimN / chunkElems) * sizeof(e8m0_t)) : 0ul;
  static constexpr size_t smem = smem_rowwise_scale;
};

// 1x32
template <typename CastTraits,
          std::enable_if_t<CastTraits::isRowwise && !CastTraits::isColwise, int> = 0>
__global__ void quantize_mxfp8_kernel_cast_only(typename CastTraits::IType *__restrict__ input,
                                                typename CastTraits::OType *__restrict__ output,
                                                e8m0_t *__restrict__ scales_rowwise, int32_t rows,
                                                int32_t cols, int32_t scale_stride_rowwise,
                                                int32_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using IType = typename CastTraits::IType;
  using OType = typename CastTraits::OType;
  using inputUnitType = typename CastTraits::inputUnitType;
  using outputUnitType = typename CastTraits::outputUnitType;

  using IType2 = typename ptx::FPx2<IType>;
  constexpr int32_t numItersIType2 = sizeof(inputUnitType) / sizeof(IType2);
  using OType2 = typename ptx::FPx2<OType>;

  e8m0_t *sRowwiseScale = nullptr;
  if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
    extern __shared__ char smem[];
    sRowwiseScale = reinterpret_cast<e8m0_t *>(smem);
  }

  int2 block_coords;
  block_coords.y = blockIdx.y * CastTraits::blockDimM + threadIdx.z * CastTraits::warpDimM +
                   (threadIdx.x / CastTraits::threadLayout::N);
  block_coords.x = blockIdx.x * CastTraits::blockDimN + threadIdx.y * CastTraits::warpDimN +
                   (threadIdx.x % CastTraits::threadLayout::N) * CastTraits::chunkElems;

  int32_t rowwise_scale_smem_base_offset;
  constexpr int32_t rowwise_scale_stride_in_smem = CastTraits::blockDimN / CastTraits::chunkElems;
  if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
    rowwise_scale_smem_base_offset =
        threadIdx.z * CastTraits::warpDimM * rowwise_scale_stride_in_smem +
        threadIdx.y * (CastTraits::warpDimN / CastTraits::chunkElems) +
        (threadIdx.x / CastTraits::threadLayout::N) * rowwise_scale_stride_in_smem +
        (threadIdx.x % CastTraits::threadLayout::N);
  }

  inputUnitType rInput[CastTraits::numStages][CastTraits::numUnitsPerChunk];
// prologue
#pragma unroll
  for (int32_t iter = 0; iter < CastTraits::numPrefetch; iter++) {
    int32_t iter_m = iter / CastTraits::iterLayout::N;
    int32_t iter_n = iter % CastTraits::iterLayout::N;

    int2 coords;
    coords.y = block_coords.y + iter_m * CastTraits::blockIterDimM;
    coords.x = block_coords.x + iter_n * CastTraits::blockIterDimN;

    if (coords.y < rows && coords.x < cols) {
      size_t offset = coords.y * static_cast<size_t>(cols) + coords.x;
      inputUnitType *input_units = reinterpret_cast<inputUnitType *>(input + offset);

#pragma unroll
      for (int32_t i = 0; i < CastTraits::numUnitsPerChunk; i++) {
        rInput[iter][i] = input_units[i];
      }
    }
  }
// mainloop
#pragma unroll
  for (int32_t iter = CastTraits::numPrefetch; iter < CastTraits::iterLayout::num; iter++) {
    {
      // load data
      int32_t iter_m = iter / CastTraits::iterLayout::N;
      int32_t iter_n = iter % CastTraits::iterLayout::N;

      int2 coords;
      coords.y = block_coords.y + iter_m * CastTraits::blockIterDimM;
      coords.x = block_coords.x + iter_n * CastTraits::blockIterDimN;

      if (coords.y < rows && coords.x < cols) {
        size_t offset = coords.y * static_cast<size_t>(cols) + coords.x;
        inputUnitType *input_units = reinterpret_cast<inputUnitType *>(input + offset);

#pragma unroll
        for (int32_t i = 0; i < CastTraits::numUnitsPerChunk; i++) {
          rInput[iter % CastTraits::numStages][i] = input_units[i];
        }
      }
    }
    int32_t process_iter = iter - CastTraits::numPrefetch;
    int32_t iter_m = process_iter / CastTraits::iterLayout::N;
    int32_t iter_n = process_iter % CastTraits::iterLayout::N;
    int2 coords;
    coords.y = block_coords.y + iter_m * CastTraits::blockIterDimM;
    coords.x = block_coords.x + iter_n * CastTraits::blockIterDimN;
    if (coords.y >= rows || coords.x >= cols) {
      return;
    }

    if constexpr (std::is_same_v<IType, float>) {
      float thread_amax = 0.f;
      IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput[process_iter % CastTraits::numStages]);
#pragma unroll
      for (int32_t j = 0; j < numItersIType2 * CastTraits::numUnitsPerChunk; j++) {
        ptx::abs_max_2x(thread_amax, thread_amax, rInput2[j].x, rInput2[j].y);
      }
      e8m0_t biased_exponent = to_e8m0<OType>(thread_amax);
      if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
        int32_t rowwise_scale_offset =
            rowwise_scale_smem_base_offset +
            iter_m * CastTraits::blockIterDimM * rowwise_scale_stride_in_smem +
            iter_n * (CastTraits::blockIterDimN / CastTraits::chunkElems);
        sRowwiseScale[rowwise_scale_offset] = biased_exponent;
      } else {
        scales_rowwise[coords.y * static_cast<size_t>(scale_stride_rowwise) +
                       coords.x / CastTraits::chunkElems] = biased_exponent;
      }

      float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
      ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

      outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
      if constexpr (CastTraits::_use_cvt_4x) {
        using OType4 = ptx::FPx4<OType>;
        using IType4 = ptx::FPx4<IType>;
        IType4 *rInput4 = reinterpret_cast<IType4 *>(&rInput[process_iter % CastTraits::numStages]);
        OType4 *rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
#pragma unroll
        for (int32_t j = 0; j < CastTraits::chunkElems / 4; j++) {
          IType4 in = rInput4[j];
          OType4 out;
          ptx::mul_cvt_4x(out, in, block_scale_inverse_2x);
          rOutput4[j] = out;
        }
      } else {
        OType2 *rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
#pragma unroll
        for (int32_t j = 0; j < CastTraits::chunkElems / 2; j++) {
          IType2 in = rInput2[j];
          OType2 out;
          ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
          rOutput2[j] = out;
        }
      }
      outputUnitType *output_units =
          reinterpret_cast<outputUnitType *>(output + coords.y * cols + coords.x);
#pragma unroll
      for (int32_t j = 0; j < CastTraits::numOutUnitsPerChunk; j++) {
        output_units[j] = rOutput[j];
      }
    } else {
      IType2 thread_amax2{0.f, 0.f};
      IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput[process_iter % CastTraits::numStages]);
#pragma unroll
      for (int32_t j = 0; j < numItersIType2 * CastTraits::numUnitsPerChunk; j++) {
        ptx::abs_max_2x(thread_amax2, thread_amax2, rInput2[j]);
      }
      IType thread_amax = ptx::get_amax(thread_amax2.x, thread_amax2.y);
      e8m0_t biased_exponent = to_e8m0<OType>(thread_amax);
      if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
        int32_t rowwise_scale_offset =
            rowwise_scale_smem_base_offset +
            iter_m * CastTraits::blockIterDimM * rowwise_scale_stride_in_smem +
            iter_n * (CastTraits::blockIterDimN / CastTraits::chunkElems);
        sRowwiseScale[rowwise_scale_offset] = biased_exponent;
      } else {
        scales_rowwise[coords.y * static_cast<size_t>(scale_stride_rowwise) +
                       coords.x / CastTraits::chunkElems] = biased_exponent;
      }

      // scaling input
      float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
      ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

      outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
      if constexpr (CastTraits::_use_cvt_4x) {
        using OType4 = ptx::FPx4<OType>;
        using IType4 = ptx::FPx4<IType>;
        IType4 *rInput4 = reinterpret_cast<IType4 *>(&rInput[process_iter % CastTraits::numStages]);
        OType4 *rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
#pragma unroll
        for (int32_t i = 0; i < CastTraits::chunkElems / 4; i++) {
          IType4 in = rInput4[i];
          OType4 out;
          ptx::mul_cvt_4x(out, in, block_scale_inverse_2x);
          rOutput4[i] = out;
        }
      } else {
        OType2 *rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
#pragma unroll
        for (int32_t i = 0; i < CastTraits::chunkElems / 2; i++) {
          IType2 in = rInput2[i];
          OType2 out;
          ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
          rOutput2[i] = out;
        }
      }
      outputUnitType *output_units =
          reinterpret_cast<outputUnitType *>(output + coords.y * cols + coords.x);
#pragma unroll
      for (int32_t j = 0; j < CastTraits::numOutUnitsPerChunk; j++) {
        output_units[j] = rOutput[j];
      }
    }
  }

// epilogue
#pragma unroll
  for (int32_t iter = CastTraits::iterLayout::num;
       iter < CastTraits::iterLayout::num + CastTraits::numPrefetch; iter++) {
    int32_t process_iter = iter - CastTraits::numPrefetch;
    int32_t iter_m = process_iter / CastTraits::iterLayout::N;
    int32_t iter_n = process_iter % CastTraits::iterLayout::N;
    int2 coords;
    coords.y = block_coords.y + iter_m * CastTraits::blockIterDimM;
    coords.x = block_coords.x + iter_n * CastTraits::blockIterDimN;
    if (coords.y >= rows || coords.x >= cols) {
      return;
    }

    if constexpr (std::is_same_v<IType, float>) {
      float thread_amax = 0.f;
      IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput[process_iter % CastTraits::numStages]);
#pragma unroll
      for (int32_t j = 0; j < numItersIType2 * CastTraits::numUnitsPerChunk; j++) {
        ptx::abs_max_2x(thread_amax, thread_amax, rInput2[j].x, rInput2[j].y);
      }
      e8m0_t biased_exponent = to_e8m0<OType>(thread_amax);
      if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
        int32_t rowwise_scale_offset =
            rowwise_scale_smem_base_offset +
            iter_m * CastTraits::blockIterDimM * rowwise_scale_stride_in_smem +
            iter_n * (CastTraits::blockIterDimN / CastTraits::chunkElems);
        sRowwiseScale[rowwise_scale_offset] = biased_exponent;
      } else {
        scales_rowwise[coords.y * static_cast<size_t>(scale_stride_rowwise) +
                       coords.x / CastTraits::chunkElems] = biased_exponent;
      }

      float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
      ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

      outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
      if constexpr (CastTraits::_use_cvt_4x) {
        using OType4 = ptx::FPx4<OType>;
        using IType4 = ptx::FPx4<IType>;
        IType4 *rInput4 = reinterpret_cast<IType4 *>(&rInput[process_iter % CastTraits::numStages]);
        OType4 *rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
#pragma unroll
        for (int32_t j = 0; j < CastTraits::chunkElems / 4; j++) {
          IType4 in = rInput4[j];
          OType4 out;
          ptx::mul_cvt_4x(out, in, block_scale_inverse_2x);
          rOutput4[j] = out;
        }
      } else {
        OType2 *rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
#pragma unroll
        for (int32_t j = 0; j < CastTraits::chunkElems / 2; j++) {
          IType2 in = rInput2[j];
          OType2 out;
          ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
          rOutput2[j] = out;
        }
      }
      outputUnitType *output_units =
          reinterpret_cast<outputUnitType *>(output + coords.y * cols + coords.x);
#pragma unroll
      for (int32_t j = 0; j < CastTraits::numOutUnitsPerChunk; j++) {
        output_units[j] = rOutput[j];
      }
    } else {
      IType2 thread_amax2{0.f, 0.f};
      IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput[process_iter % CastTraits::numStages]);
#pragma unroll
      for (int32_t j = 0; j < numItersIType2 * CastTraits::numUnitsPerChunk; j++) {
        ptx::abs_max_2x(thread_amax2, thread_amax2, rInput2[j]);
      }
      IType thread_amax = ptx::get_amax(thread_amax2.x, thread_amax2.y);
      e8m0_t biased_exponent = to_e8m0<OType>(thread_amax);
      if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
        int32_t rowwise_scale_offset =
            rowwise_scale_smem_base_offset +
            iter_m * CastTraits::blockIterDimM * rowwise_scale_stride_in_smem +
            iter_n * (CastTraits::blockIterDimN / CastTraits::chunkElems);
        sRowwiseScale[rowwise_scale_offset] = biased_exponent;
      } else {
        scales_rowwise[coords.y * static_cast<size_t>(scale_stride_rowwise) +
                       coords.x / CastTraits::chunkElems] = biased_exponent;
      }

      // scaling input
      float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
      ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

      outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
      if constexpr (CastTraits::_use_cvt_4x) {
        using OType4 = ptx::FPx4<OType>;
        using IType4 = ptx::FPx4<IType>;
        IType4 *rInput4 = reinterpret_cast<IType4 *>(&rInput[process_iter % CastTraits::numStages]);
        OType4 *rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
#pragma unroll
        for (int32_t i = 0; i < CastTraits::chunkElems / 4; i++) {
          IType4 in = rInput4[i];
          OType4 out;
          ptx::mul_cvt_4x(out, in, block_scale_inverse_2x);
          rOutput4[i] = out;
        }
      } else {
        OType2 *rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
#pragma unroll
        for (int32_t i = 0; i < CastTraits::chunkElems / 2; i++) {
          IType2 in = rInput2[i];
          OType2 out;
          ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
          rOutput2[i] = out;
        }
      }
      outputUnitType *output_units =
          reinterpret_cast<outputUnitType *>(output + coords.y * cols + coords.x);
#pragma unroll
      for (int32_t j = 0; j < CastTraits::numOutUnitsPerChunk; j++) {
        output_units[j] = rOutput[j];
      }
    }
  }

  if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
    __syncthreads();

    int32_t warpId = threadIdx.z * CastTraits::warpLayout::N + threadIdx.y;

    block_coords.y = blockIdx.y * CastTraits::blockDimM;
    block_coords.x = blockIdx.x * CastTraits::blockDimN;

    constexpr int32_t stride_in_smem = CastTraits::blockDimN / CastTraits::chunkElems;
    using PreferredDataType = std::conditional_t<
        stride_in_smem % 16 == 0, uint4,
        std::conditional_t<
            stride_in_smem % 8 == 0, uint2,
            std::conditional_t<stride_in_smem % 4 == 0, uint32_t,
                               std::conditional_t<stride_in_smem % 2 == 0, uint16_t, uint8_t>>>>;

    int2 end_coords;
    end_coords.y = std::min(block_coords.y + CastTraits::blockDimM, rows);
    end_coords.x = std::min((block_coords.x + CastTraits::blockDimN) / CastTraits::chunkElems,
                            scale_stride_rowwise);
    int2 valid_coords;
    valid_coords.y = end_coords.y - block_coords.y;
    valid_coords.x = end_coords.x - (block_coords.x / CastTraits::chunkElems);

    if (scale_stride_rowwise % sizeof(PreferredDataType) != 0) {
      using DataType = int32_t;
      constexpr int32_t num_elems_per_group = sizeof(DataType) / sizeof(e8m0_t);
      constexpr int32_t num_groups_per_row_in_smem = stride_in_smem / num_elems_per_group;

      int32_t num_threads_per_row = (valid_coords.x / num_elems_per_group);
      int32_t gmem_stride_in_group = scale_stride_rowwise / num_elems_per_group;

      DataType *sScales = reinterpret_cast<DataType *>(sRowwiseScale);
      DataType *gScales =
          reinterpret_cast<DataType *>(scales_rowwise + block_coords.y * scale_stride_rowwise +
                                       block_coords.x / CastTraits::chunkElems);

      for (int32_t i = threadIdx.x + warpId * 32; i < (valid_coords.y * num_threads_per_row);
           i += CastTraits::warpLayout::num * 32) {
        int32_t row = i / num_threads_per_row;
        int32_t col = i % num_threads_per_row;
        gScales[row * gmem_stride_in_group + col] = sScales[row * num_groups_per_row_in_smem + col];
      }
    } else {
      using DataType = PreferredDataType;
      constexpr int32_t num_elems_per_group = sizeof(DataType) / sizeof(e8m0_t);
      constexpr int32_t num_groups_per_row_in_smem = stride_in_smem / num_elems_per_group;

      int32_t num_threads_per_row = (valid_coords.x / num_elems_per_group);
      int32_t gmem_stride_in_group = scale_stride_rowwise / num_elems_per_group;

      DataType *sScales = reinterpret_cast<DataType *>(sRowwiseScale);
      DataType *gScales =
          reinterpret_cast<DataType *>(scales_rowwise + block_coords.y * scale_stride_rowwise +
                                       block_coords.x / CastTraits::chunkElems);

      for (int32_t i = threadIdx.x + warpId * 32; i < (valid_coords.y * num_threads_per_row);
           i += CastTraits::warpLayout::num * 32) {
        int32_t row = i / num_threads_per_row;
        int32_t col = i % num_threads_per_row;
        gScales[row * gmem_stride_in_group + col] = sScales[row * num_groups_per_row_in_smem + col];
      }
    }
  }

#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

enum class ColwiseReduceMax : int32_t {
  Atom = 0,
  Red = 1,  // it's actually the same to Atom
  RedAsync = 2,
  Redux = 3,
  Num = 4
};

// 32x32
template <typename _IType, typename _OType>
struct CastTraits<_IType, _OType, /*rowwise=*/true, /*colwise=*/true> {
  static constexpr bool isRowwise = true;
  static constexpr bool isColwise = true;
  using IType = _IType;
  using OType = _OType;

  static constexpr int32_t rowChunkElems = 32;
  static constexpr int32_t colChunkElems = 32;

  using rowThreadLayout = Layout<32, 1>;                                   // 32x1
  using colThreadLayout = Layout<rowThreadLayout::N, rowThreadLayout::M>;  // 1x32
  static_assert(rowThreadLayout::num == colThreadLayout::num,
                "rowThreadLayout::num must be equal to colThreadLayout::num");
  static_assert(rowThreadLayout::num == 32, "rowThreadLayout::num must be 32");

  using rowWarpDim = Layout<rowThreadLayout::M, rowThreadLayout::N * rowChunkElems>;
  using colWarpDim = Layout<colThreadLayout::M * colChunkElems, colThreadLayout::N>;
  using warpDim =
      Layout<std::max(rowWarpDim::M, colWarpDim::M), std::max(rowWarpDim::N, colWarpDim::N)>;

  static constexpr bool _tma_swizzle = true;
  using warpLayout = Layout<1, 2>;
  static_assert(_tma_swizzle ? (warpLayout::N == 2) : true);
  static constexpr CUtensorMapSwizzle input_swizzle_pattern =
      _tma_swizzle ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
                   : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;

  static constexpr CUtensorMapSwizzle output_swizzle_pattern =
      _tma_swizzle ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
                   : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;

  using blockIterDim = Layout<warpLayout::M * warpDim::M, warpLayout::N * warpDim::N>;

  using iterLayout = Layout<1, 4>;
  using blockDIM = Layout<iterLayout::M * blockIterDim::M, iterLayout::N * blockIterDim::N>;

  static constexpr int32_t numStages = 2;

  using inputUnitType = uint4;
  static constexpr int32_t rowNumElemsPerUnit = sizeof(inputUnitType) / sizeof(IType);
  static constexpr int32_t rowNumUnitsPerChunk = rowChunkElems / rowNumElemsPerUnit;
  // TODO: set condition for float
  using inputElemSwz = std::conditional_t<_tma_swizzle, swz::Swizzle<3, 3, 3>, swz::Linear>;
  using inputUnitSwz = std::conditional_t<_tma_swizzle, swz::Swizzle<3, 0, 3>, swz::Linear>;

  using colIndexSwz = swz::Swizzle<5, 0, 5>;

  using rowOutputUnitType = uint4;
  static constexpr int32_t rowNumOutUnitsPerChunk =
      rowChunkElems * sizeof(OType) / sizeof(rowOutputUnitType);
  static constexpr int32_t rowOutNumElemsPerUnit = sizeof(rowOutputUnitType) / sizeof(OType);

  using rowOutputChunkSwz = std::conditional_t<_tma_swizzle, swz::Swizzle<2, 0, 3>, swz::Linear>;
  using colOutputSwz = std::conditional_t<_tma_swizzle, swz::Swizzle<2, 4, 3>, swz::Linear>;

  static constexpr bool _use_cvt_4x = true;
  static constexpr bool _use_warp_specialization = false;
  static constexpr bool _need_wait_group = iterLayout::num > numStages;
  static constexpr bool _reuse_input_out_smem = false;
  static_assert(_reuse_input_out_smem == false, "Just don't use it");
  static constexpr bool _cache_rowwise_scale_in_smem = true;

  static constexpr bool _colwise_source_coming_from_rowwise = true;
  static constexpr ColwiseReduceMax _colwise_reduce_max = ColwiseReduceMax::Redux;
  static_assert(_colwise_reduce_max != ColwiseReduceMax::RedAsync,
                "It requires aligned smem pointer");

  static constexpr int32_t numWarps = warpLayout::num + 2 * (int32_t)_use_warp_specialization;
  static constexpr int32_t numThreads = numWarps * 32;
  static_assert(numThreads <= 1024, "numThreads must be less than or equal to 1024");

  static constexpr size_t smemInputPerWarp = warpDim::num * sizeof(IType);
  static constexpr size_t smemInputPerBlock = smemInputPerWarp * warpLayout::num;

  static constexpr size_t smemRowwiseOutputPerWarp = warpDim::num * sizeof(OType);
  static constexpr size_t smemRowwiseOutputPerBlock = smemRowwiseOutputPerWarp * warpLayout::num;

  static constexpr size_t smemColwiseOutputPerWarp = warpDim::num * sizeof(OType);
  static constexpr size_t smemColwiseOutputPerBlock = smemColwiseOutputPerWarp * warpLayout::num;

  static constexpr size_t smemInput = smemInputPerBlock * numStages;
  static constexpr size_t smemRowwiseOutput = smemRowwiseOutputPerBlock * numStages;
  static constexpr size_t smemColwiseOutput = smemColwiseOutputPerBlock * numStages;

  static constexpr size_t smem_rowwise_scale =
      _cache_rowwise_scale_in_smem ? (blockDIM::M * (blockDIM::N / rowChunkElems) * sizeof(e8m0_t))
                                   : 0ul;

  using ColwiseReduceDataType = float;
  static constexpr bool _need_smem_for_colwise_reduce =
      _colwise_source_coming_from_rowwise;  // && _colwise_reduce_max != ColwiseReduceMax::Redux;
  static constexpr size_t smem_colwise_reduce =
      _need_smem_for_colwise_reduce ? 32 * warpLayout::num * sizeof(ColwiseReduceDataType) : 0ul;

  static constexpr size_t smem_alignment = _tma_swizzle ? 1024ul : 128ul;
  static constexpr size_t smem = _reuse_input_out_smem
                                     ? (std::max(smemInput, smemColwiseOutput) + smemRowwiseOutput +
                                        smem_alignment + smem_rowwise_scale + smem_colwise_reduce)
                                     : (smemInput + smemRowwiseOutput + smemColwiseOutput +
                                        smem_alignment + smem_rowwise_scale + smem_colwise_reduce);
};

__device__ __forceinline__ intptr_t align_to(intptr_t x, intptr_t align) {
  return (x + align - 1) & ~((align)-1);
}

// 32x32
template <typename CastTraits,
          std::enable_if_t<CastTraits::isRowwise && CastTraits::isColwise, int> = 0,
          std::enable_if_t<CastTraits::_use_warp_specialization, int> = 0>
// __launch_bounds__(CastTraits::numThreads)
__global__ void quantize_mxfp8_kernel_cast_only(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_rowwise_output,
    const __grid_constant__ CUtensorMap tensor_map_colwise_output,
    e8m0_t *__restrict__ scales_rowwise, e8m0_t *__restrict__ scales_colwise, int32_t rows,
    int32_t cols, int32_t scale_stride_rowwise, int32_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using IType = typename CastTraits::IType;
  using OType = typename CastTraits::OType;
  using inputUnitType = typename CastTraits::inputUnitType;
  using rowOutputUnitType = typename CastTraits::rowOutputUnitType;
  using ColwiseReduceDataType = typename CastTraits::ColwiseReduceDataType;

  using IType2 = typename ptx::FPx2<IType>;
  using OType2 = typename ptx::FPx2<OType>;
  constexpr int32_t numItersIType2 = sizeof(inputUnitType) / sizeof(IType2);

  int32_t warpId = threadIdx.y;
  int32_t leader = ptx::elect_one_sync();
  int2 block_coords;
  block_coords.y = blockIdx.y * CastTraits::blockDIM::M;
  block_coords.x = blockIdx.x * CastTraits::blockDIM::N;

  extern __shared__ char smem[];
  char *smemAligned = reinterpret_cast<char *>(
      align_to(reinterpret_cast<intptr_t>(smem), CastTraits::smem_alignment));

  IType *sInput = reinterpret_cast<IType *>(smemAligned);
  inputUnitType *sInputUnit = reinterpret_cast<inputUnitType *>(sInput);

  OType *sRowOutput =
      reinterpret_cast<OType *>(sInput + CastTraits::blockIterDim::num * CastTraits::numStages);
  rowOutputUnitType *sRowOutputUnit = reinterpret_cast<rowOutputUnitType *>(sRowOutput);

  OType *sColOutput =
      reinterpret_cast<OType *>(sRowOutput + CastTraits::blockIterDim::num * CastTraits::numStages);
  rowOutputUnitType *sColOutputUnit = reinterpret_cast<rowOutputUnitType *>(sColOutput);

  e8m0_t *sRowwiseScale = nullptr;
  ColwiseReduceDataType *sColwiseReduce = nullptr;
  if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
    sRowwiseScale = reinterpret_cast<e8m0_t *>(sColOutput + CastTraits::blockIterDim::num *
                                                                CastTraits::numStages);
    if constexpr (CastTraits::_need_smem_for_colwise_reduce) {
      sColwiseReduce = reinterpret_cast<ColwiseReduceDataType *>(
          sRowwiseScale + CastTraits::smem_rowwise_scale / sizeof(e8m0_t));
      sColwiseReduce += warpId * 32;
    }
  } else if constexpr (CastTraits::_need_smem_for_colwise_reduce) {
    sColwiseReduce = reinterpret_cast<ColwiseReduceDataType *>(
        sColOutput + CastTraits::blockIterDim::num * CastTraits::numStages);
    sColwiseReduce += warpId * 32;
  }

  // TODO: maybe we can assign a different barrier for each warp
  __shared__ uint64_t ldg_producer[CastTraits::numStages], ldg_consumer[CastTraits::numStages];
  __shared__ uint64_t stg_producer[CastTraits::numStages], stg_consumer[CastTraits::numStages];

  if (warpId == 0 && leader) {
#pragma unroll
    for (int32_t i = 0; i < CastTraits::numStages; i++) {
      ptx::mbarrier_init(&ldg_producer[i], 1);
      ptx::mbarrier_init(&ldg_consumer[i], CastTraits::warpLayout::num * 32);
      ptx::mbarrier_init(&stg_producer[i], CastTraits::warpLayout::num * 32);
      ptx::mbarrier_init(&stg_consumer[i], 1);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  __syncthreads();

  if (warpId == CastTraits::warpLayout::num) {
    if (leader) {
      PipeState<CastTraits::numStages, true> write_state;
#pragma unroll 1
      for (int32_t iter = 0; iter < CastTraits::iterLayout::num; iter++) {
        int32_t iter_m = iter / CastTraits::iterLayout::N;
        int32_t iter_n = iter % CastTraits::iterLayout::N;

        int2 coords;
        coords.y = block_coords.y + iter_m * CastTraits::blockIterDim::M;
        coords.x = block_coords.x + iter_n * CastTraits::blockIterDim::N;

        if (coords.x >= cols || coords.y >= rows) {
          break;
        }

        ptx::mbarrier_wait_parity(&ldg_consumer[write_state.index()], write_state.phase());

        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(sInput +
                                         write_state.index() * CastTraits::blockIterDim::num),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), static_cast<uint32_t>(coords.x),
            static_cast<uint32_t>(coords.y), &ldg_producer[write_state.index()]);
        ptx::mbarrier_arrive_expect_tx(&ldg_producer[write_state.index()],
                                       CastTraits::blockIterDim::num * sizeof(IType));
        write_state++;
      }
    }
  } else if (warpId == CastTraits::warpLayout::num + 1) {
    if (leader) {
      PipeState<CastTraits::numStages> read_state;

#pragma unroll 1
      for (int32_t iter = 0; iter < CastTraits::numStages - 1; iter++) {
        int32_t iter_m = iter / CastTraits::iterLayout::N;
        int32_t iter_n = iter % CastTraits::iterLayout::N;

        int2 coords;
        coords.y = block_coords.y + iter_m * CastTraits::blockIterDim::M;
        coords.x = block_coords.x + iter_n * CastTraits::blockIterDim::N;

        size_t gmem_offset =
            static_cast<size_t>(read_state.index()) * CastTraits::blockIterDim::num;

        if (coords.x >= cols || coords.y >= rows) {
          break;
        }

        ptx::mbarrier_wait_parity(&stg_producer[read_state.index()], read_state.phase());

        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_rowwise_output),
            static_cast<uint32_t>(coords.x), static_cast<uint32_t>(coords.y),
            reinterpret_cast<uint64_t *>(sRowOutput + gmem_offset));
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_colwise_output),
            static_cast<uint32_t>(coords.x), static_cast<uint32_t>(coords.y),
            reinterpret_cast<uint64_t *>(sColOutput + gmem_offset));
        ptx::cp_async_bulk_commit_group();
        read_state++;
      }

#pragma unroll 1
      for (int32_t iter = CastTraits::numStages - 1; iter < CastTraits::iterLayout::num; iter++) {
        int32_t iter_m = iter / CastTraits::iterLayout::N;
        int32_t iter_n = iter % CastTraits::iterLayout::N;

        int2 coords;
        coords.y = block_coords.y + iter_m * CastTraits::blockIterDim::M;
        coords.x = block_coords.x + iter_n * CastTraits::blockIterDim::N;

        size_t gmem_offset =
            static_cast<size_t>(read_state.index()) * CastTraits::blockIterDim::num;

        if (coords.x >= cols || coords.y >= rows) {
          break;
        }

        ptx::mbarrier_wait_parity(&stg_producer[read_state.index()], read_state.phase());
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_rowwise_output),
            static_cast<uint32_t>(coords.x), static_cast<uint32_t>(coords.y),
            reinterpret_cast<uint64_t *>(sRowOutput + gmem_offset));
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_colwise_output),
            static_cast<uint32_t>(coords.x), static_cast<uint32_t>(coords.y),
            reinterpret_cast<uint64_t *>(sColOutput + gmem_offset));
        ptx::cp_async_bulk_commit_group();
        read_state++;

        ptx::cp_async_bulk_wait_group_read<CastTraits::numStages - 1>();
        ptx::mbarrier_arrive_expect_tx(&stg_consumer[read_state.index()], 0u);
      }
    }
    ptx::cp_async_bulk_wait_group_read<0>();
  } else {
    PipeState<CastTraits::numStages> read_state;

    int2 warp_coords;
    warp_coords.y = (warpId / CastTraits::warpLayout::N) * CastTraits::warpDim::M;
    warp_coords.x = (warpId % CastTraits::warpLayout::N) * CastTraits::warpDim::N;

    int32_t warp_base_offset = warp_coords.y * CastTraits::blockIterDim::N + warp_coords.x;

    int32_t thread_base_offset =
        (threadIdx.x / CastTraits::rowThreadLayout::N) *
            (CastTraits::blockIterDim::N / CastTraits::rowNumElemsPerUnit) +
        (threadIdx.x % CastTraits::rowThreadLayout::N) *
            (CastTraits::rowChunkElems / CastTraits::rowNumElemsPerUnit);

    size_t rowwise_scale_base_offset =
        (block_coords.y + warp_coords.y + (threadIdx.x / CastTraits::rowThreadLayout::N)) *
            static_cast<size_t>(scale_stride_rowwise) +
        (block_coords.x + warp_coords.x +
         (threadIdx.x % CastTraits::rowThreadLayout::N) * CastTraits::rowChunkElems) /
            CastTraits::rowChunkElems;
    size_t colwise_scale_base_offset =
        ((block_coords.y + warp_coords.y +
          (threadIdx.x / CastTraits::colThreadLayout::N) * CastTraits::colChunkElems) /
         CastTraits::colChunkElems) *
            static_cast<size_t>(scale_stride_colwise) +
        (block_coords.x + warp_coords.x + (threadIdx.x % CastTraits::colThreadLayout::N));

    constexpr int32_t rowwise_scale_stride_in_smem =
        CastTraits::blockDIM::N / CastTraits::rowChunkElems;
    int32_t rowwise_scale_smem_base_offset =
        (warpId / CastTraits::warpLayout::N) * CastTraits::warpDim::M *
            rowwise_scale_stride_in_smem +
        (warpId % CastTraits::warpLayout::N) *
            (CastTraits::warpDim::N / CastTraits::rowChunkElems) +
        (threadIdx.x / CastTraits::rowThreadLayout::N) * rowwise_scale_stride_in_smem +
        (threadIdx.x % CastTraits::rowThreadLayout::N);

#pragma unroll 1
    for (int32_t iter = 0; iter < CastTraits::iterLayout::num; iter++) {
      int32_t iter_m = iter / CastTraits::iterLayout::N;
      int32_t iter_n = iter % CastTraits::iterLayout::N;

      if (block_coords.x + iter_n * CastTraits::blockIterDim::N >= cols ||
          block_coords.y + iter_m * CastTraits::blockIterDim::M >= rows) {
        break;
      }

      ptx::mbarrier_wait_parity(&ldg_producer[read_state.index()], read_state.phase());

      {
        int32_t warp_offset = warp_base_offset + read_state.index() * CastTraits::blockIterDim::num;
        static_assert(CastTraits::_colwise_source_coming_from_rowwise);
        if constexpr (CastTraits::_colwise_source_coming_from_rowwise) {
          if constexpr (CastTraits::_need_smem_for_colwise_reduce &&
                        CastTraits::_colwise_reduce_max != ColwiseReduceMax::Redux) {
            sColwiseReduce[threadIdx.x] = 0;
          }

          IType rInput[CastTraits::rowChunkElems];
          {
            inputUnitType *rInputUnit = reinterpret_cast<inputUnitType *>(rInput);
            int32_t base = thread_base_offset + warp_offset / CastTraits::rowNumElemsPerUnit;
#pragma unroll
            for (int32_t i = 0; i < CastTraits::rowNumUnitsPerChunk; i++) {
              rInputUnit[i] = sInputUnit[CastTraits::inputUnitSwz::swz(base + i)];
            }
            ptx::mbarrier_arrive_expect_tx(&ldg_consumer[read_state.index()], 0u);
          }

          if constexpr (std::is_same_v<IType, float>) {
          } else {
            static_assert(CastTraits::_colwise_reduce_max == ColwiseReduceMax::Redux,
                          "Only Redux is implemented");

            float row_scale_inverse;

            IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput);
            float2 *sColwiseReduce_2x = reinterpret_cast<float2 *>(sColwiseReduce);

            IType2 row_amax2{0.0f, 0.0f};
#pragma unroll
            for (int32_t i = 0; i < CastTraits::rowChunkElems / 2; i++) {
              ptx::abs_max_2x(row_amax2, row_amax2, rInput2[i]);

              float2 values = ptx::up_cast(rInput2[i]);

              float2 amaxs;
              ptx::reduce_sync_max_abs_f32(amaxs.x, values.x);
              ptx::reduce_sync_max_abs_f32(amaxs.y, values.y);
              if (leader) {
                sColwiseReduce_2x[i] = amaxs;
              }
            }
            {
              IType row_amax = ptx::get_amax(row_amax2.x, row_amax2.y);
              e8m0_t row_biased_exponent = to_e8m0<OType>(row_amax);
              row_scale_inverse = ptx::exp2f_rcp(row_biased_exponent);
              if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
                int32_t rowwise_scale_offset =
                    rowwise_scale_smem_base_offset +
                    iter_m * CastTraits::blockIterDim::M * rowwise_scale_stride_in_smem +
                    iter_n * (CastTraits::blockIterDim::N / CastTraits::rowChunkElems);
                sRowwiseScale[rowwise_scale_offset] = row_biased_exponent;
              } else {
                size_t rowwise_scale_offset =
                    rowwise_scale_base_offset +
                    iter_m * (CastTraits::blockIterDim::M) *
                        static_cast<size_t>(scale_stride_rowwise) +
                    iter_n * (CastTraits::blockIterDim::N / CastTraits::rowChunkElems);
                scales_rowwise[rowwise_scale_offset] = row_biased_exponent;
              }
            }
            {
              __syncwarp();
              float col_amax = sColwiseReduce[threadIdx.x];
              e8m0_t col_biased_exponent = to_e8m0<OType>(col_amax);
              float col_scale_inverse = ptx::exp2f_rcp(col_biased_exponent);
              sColwiseReduce[threadIdx.x] = col_scale_inverse;
              size_t colwise_scale_offset =
                  colwise_scale_base_offset +
                  iter_m * (CastTraits::blockIterDim::M / CastTraits::colChunkElems) *
                      static_cast<size_t>(scale_stride_colwise) +
                  iter_n * CastTraits::blockIterDim::N;
              scales_colwise[colwise_scale_offset] = col_biased_exponent;
              __syncwarp();
            }
            // rowwise & colwise scaling
            {
              rowOutputUnitType rRowOutputUnit[CastTraits::rowNumOutUnitsPerChunk];
              rowOutputUnitType rColOutputUnit[CastTraits::rowNumOutUnitsPerChunk];

              ptx::floatx2 row_scale_inverse_2{row_scale_inverse, row_scale_inverse};
              if constexpr (CastTraits::_use_cvt_4x) {
                using OType4 = ptx::FPx4<OType>;
                using IType4 = ptx::FPx4<IType>;

                ptx::floatx4 col_scale_inverse_4[2];
                ptx::floatx4 *sColwiseScale4x = reinterpret_cast<ptx::floatx4 *>(sColwiseReduce);
                col_scale_inverse_4[0] = sColwiseScale4x[0];

                IType4 *rInput4 = reinterpret_cast<IType4 *>(&rInput);
                OType4 *rRowOutput4 = reinterpret_cast<OType4 *>(&rRowOutputUnit);
                OType4 *rColOutput4 = reinterpret_cast<OType4 *>(&rColOutputUnit);
#pragma unroll
                for (int32_t i = 1; i < CastTraits::rowChunkElems / 4; i++) {
                  {
                    col_scale_inverse_4[i % 2] = sColwiseScale4x[i];
                  }

                  IType4 in = rInput4[i - 1];
                  ptx::floatx4 in_fp4 = ptx::up_cast(in);

                  OType4 row_out;
                  ptx::mul_cvt_4x(row_out, in_fp4, row_scale_inverse_2);
                  rRowOutput4[i - 1] = row_out;

                  OType4 col_out;
                  ptx::mul_cvt_4x(col_out, in_fp4, col_scale_inverse_4[(i - 1) % 2]);
                  rColOutput4[i - 1] = col_out;
                }
                {
                  constexpr int32_t i = (CastTraits::rowChunkElems / 4) - 1;
                  IType4 in = rInput4[i];
                  ptx::floatx4 in_fp4 = ptx::up_cast(in);

                  OType4 row_out;
                  ptx::mul_cvt_4x(row_out, in_fp4, row_scale_inverse_2);
                  rRowOutput4[i] = row_out;

                  OType4 col_out;
                  ptx::mul_cvt_4x(col_out, in_fp4, col_scale_inverse_4[i % 2]);
                  rColOutput4[i] = col_out;
                }
              } else {
                ptx::floatx2 col_scale_inverse_2[2];
                ptx::floatx2 *sColwiseScale2x = reinterpret_cast<ptx::floatx2 *>(sColwiseReduce);
                col_scale_inverse_2[0] = sColwiseScale2x[0];

                IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput);
                OType2 *rRowOutput2 = reinterpret_cast<OType2 *>(&rRowOutputUnit);
                OType2 *rColOutput2 = reinterpret_cast<OType2 *>(&rColOutputUnit);
#pragma unroll
                for (int32_t i = 1; i < CastTraits::rowChunkElems / 2; i++) {
                  {
                    col_scale_inverse_2[i % 2] = sColwiseScale2x[i];
                  }

                  IType2 in = rInput2[i - 1];
                  ptx::floatx2 in_fp2 = ptx::up_cast(in);

                  OType2 row_out;
                  mul_cvt_2x(row_out, in_fp2, row_scale_inverse_2);
                  rRowOutput2[i - 1] = row_out;

                  OType2 col_out;
                  mul_cvt_2x(col_out, in_fp2, col_scale_inverse_2[(i - 1) % 2]);
                  rColOutput2[i - 1] = col_out;
                }
                {
                  constexpr int32_t i = (CastTraits::rowChunkElems / 2) - 1;
                  IType2 in = rInput2[i];
                  ptx::floatx2 in_fp2 = ptx::up_cast(in);

                  OType2 row_out;
                  mul_cvt_2x(row_out, in_fp2, row_scale_inverse_2);
                  rRowOutput2[i] = row_out;

                  OType2 col_out;
                  mul_cvt_2x(col_out, in_fp2, col_scale_inverse_2[i % 2]);
                  rColOutput2[i] = col_out;
                }
              }
              {
                ptx::mbarrier_wait_parity(&stg_consumer[read_state.index()],
                                          read_state.phase() ^ 1);

                int32_t base = thread_base_offset / (CastTraits::rowOutNumElemsPerUnit /
                                                     CastTraits::rowNumElemsPerUnit) +
                               warp_offset / CastTraits::rowOutNumElemsPerUnit;
#pragma unroll
                for (int32_t i = 0; i < CastTraits::rowNumOutUnitsPerChunk; i++) {
                  int32_t offset = CastTraits::rowOutputChunkSwz::swz(base + i);
                  sRowOutputUnit[offset] = rRowOutputUnit[i];
                  sColOutputUnit[offset] = rColOutputUnit[i];
                }
              }
            }
          }
        }
      }
      ptx::fence_proxy_async_shared_cta();

      ptx::mbarrier_arrive_expect_tx(&stg_producer[read_state.index()], 0u);
      read_state++;
    }

    if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
      ptx::numbered_barrier_sync(CastTraits::warpLayout::num * 32, 0u);

      constexpr int32_t stride_in_smem = CastTraits::blockDIM::N / CastTraits::rowChunkElems;
      using PreferredDataType = std::conditional_t<
          stride_in_smem % 16 == 0, uint4,
          std::conditional_t<
              stride_in_smem % 8 == 0, uint2,
              std::conditional_t<stride_in_smem % 4 == 0, uint32_t,
                                 std::conditional_t<stride_in_smem % 2 == 0, uint16_t, uint8_t>>>>;

      int2 end_coords;
      end_coords.y = std::min(block_coords.y + CastTraits::blockDIM::M, rows);
      end_coords.x =
          std::min((block_coords.x + CastTraits::blockDIM::N) / CastTraits::rowChunkElems,
                   scale_stride_rowwise);
      int2 valid_coords;
      valid_coords.y = end_coords.y - block_coords.y;
      valid_coords.x = end_coords.x - (block_coords.x / CastTraits::rowChunkElems);

      if (scale_stride_rowwise % sizeof(PreferredDataType) != 0) {
        using DataType = int32_t;
        constexpr int32_t num_elems_per_group = sizeof(DataType) / sizeof(e8m0_t);
        constexpr int32_t num_groups_per_row_in_smem = stride_in_smem / num_elems_per_group;

        int32_t num_threads_per_row = (valid_coords.x / num_elems_per_group);
        int32_t gmem_stride_in_group = scale_stride_rowwise / num_elems_per_group;

        DataType *sScales = reinterpret_cast<DataType *>(sRowwiseScale);
        DataType *gScales =
            reinterpret_cast<DataType *>(scales_rowwise + block_coords.y * scale_stride_rowwise +
                                         block_coords.x / CastTraits::rowChunkElems);

        for (int32_t i = threadIdx.x + warpId * 32; i < (valid_coords.y * num_threads_per_row);
             i += CastTraits::warpLayout::num * 32) {
          int32_t row = i / num_threads_per_row;
          int32_t col = i % num_threads_per_row;
          gScales[row * gmem_stride_in_group + col] =
              sScales[row * num_groups_per_row_in_smem + col];
        }
      } else {
        using DataType = PreferredDataType;
        constexpr int32_t num_elems_per_group = sizeof(DataType) / sizeof(e8m0_t);
        constexpr int32_t num_groups_per_row_in_smem = stride_in_smem / num_elems_per_group;

        int32_t num_threads_per_row = (valid_coords.x / num_elems_per_group);
        int32_t gmem_stride_in_group = scale_stride_rowwise / num_elems_per_group;

        DataType *sScales = reinterpret_cast<DataType *>(sRowwiseScale);
        DataType *gScales =
            reinterpret_cast<DataType *>(scales_rowwise + block_coords.y * scale_stride_rowwise +
                                         block_coords.x / CastTraits::rowChunkElems);

        for (int32_t i = threadIdx.x + warpId * 32; i < (valid_coords.y * num_threads_per_row);
             i += CastTraits::warpLayout::num * 32) {
          int32_t row = i / num_threads_per_row;
          int32_t col = i % num_threads_per_row;
          gScales[row * gmem_stride_in_group + col] =
              sScales[row * num_groups_per_row_in_smem + col];
        }
      }
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <typename CastTraits,
          std::enable_if_t<CastTraits::isRowwise && CastTraits::isColwise, int> = 0,
          std::enable_if_t<!CastTraits::_use_warp_specialization, int> = 0>
__global__ void quantize_mxfp8_kernel_cast_only(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_rowwise_output,
    const __grid_constant__ CUtensorMap tensor_map_colwise_output, e8m0_t *scales_rowwise,
    e8m0_t *scales_colwise, int32_t rows, int32_t cols, int32_t scale_stride_rowwise,
    int32_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using IType = typename CastTraits::IType;
  using OType = typename CastTraits::OType;
  using inputUnitType = typename CastTraits::inputUnitType;
  using rowOutputUnitType = typename CastTraits::rowOutputUnitType;
  using ColwiseReduceDataType = typename CastTraits::ColwiseReduceDataType;

  using IType2 = typename ptx::FPx2<IType>;
  using OType2 = typename ptx::FPx2<OType>;

  int32_t warpId = threadIdx.y;
  int32_t leader = ptx::elect_one_sync();
  int2 block_coords;
  block_coords.y = blockIdx.y * CastTraits::blockDIM::M;
  block_coords.x = blockIdx.x * CastTraits::blockDIM::N;

  extern __shared__ char smem[];
  char *smemAligned = reinterpret_cast<char *>(
      align_to(reinterpret_cast<intptr_t>(smem), CastTraits::smem_alignment));
  IType *sInput = reinterpret_cast<IType *>(smemAligned);
  inputUnitType *sInputUnit = reinterpret_cast<inputUnitType *>(sInput);

  OType *sRowOutput =
      reinterpret_cast<OType *>(sInput + CastTraits::blockIterDim::num * CastTraits::numStages);
  rowOutputUnitType *sRowOutputUnit = reinterpret_cast<rowOutputUnitType *>(sRowOutput);

  // colwise output will reuse input buffer
  OType *sColOutput;
  e8m0_t *sRowwiseScale = nullptr;
  ColwiseReduceDataType *sColwiseReduce = nullptr;
  if constexpr (CastTraits::_reuse_input_out_smem) {
    sColOutput = reinterpret_cast<OType *>(sInput);
    if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
      sRowwiseScale = reinterpret_cast<e8m0_t *>(sRowOutput + CastTraits::blockIterDim::num *
                                                                  CastTraits::numStages);
      if constexpr (CastTraits::_need_smem_for_colwise_reduce) {
        sColwiseReduce = reinterpret_cast<ColwiseReduceDataType *>(
            sRowwiseScale + CastTraits::smem_rowwise_scale / sizeof(e8m0_t));
      }
    } else if constexpr (CastTraits::_need_smem_for_colwise_reduce) {
      sColwiseReduce = reinterpret_cast<ColwiseReduceDataType *>(
          sRowOutput + CastTraits::blockIterDim::num * CastTraits::numStages);
    }
  } else {
    sColOutput = reinterpret_cast<OType *>(sRowOutput +
                                           CastTraits::blockIterDim::num * CastTraits::numStages);
    if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
      sRowwiseScale = reinterpret_cast<e8m0_t *>(sColOutput + CastTraits::blockIterDim::num *
                                                                  CastTraits::numStages);
      if constexpr (CastTraits::_need_smem_for_colwise_reduce) {
        sColwiseReduce = reinterpret_cast<ColwiseReduceDataType *>(
            sRowwiseScale + CastTraits::smem_rowwise_scale / sizeof(e8m0_t));
      }
    } else if constexpr (CastTraits::_need_smem_for_colwise_reduce) {
      sColwiseReduce = reinterpret_cast<ColwiseReduceDataType *>(
          sColOutput + CastTraits::blockIterDim::num * CastTraits::numStages);
    }
  }
  rowOutputUnitType *sColOutputUnit = reinterpret_cast<rowOutputUnitType *>(sColOutput);

  if constexpr (CastTraits::_need_smem_for_colwise_reduce) {
    sColwiseReduce += warpId * 32;
  }

  __shared__ uint64_t producer[CastTraits::numStages];
  uint64_t *colwise_reduce_barrier = nullptr;
  if constexpr (CastTraits::_colwise_source_coming_from_rowwise &&
                CastTraits::_colwise_reduce_max == ColwiseReduceMax::RedAsync) {
    __shared__ uint64_t colwise_reduce_bar[CastTraits::warpLayout::num];
    colwise_reduce_barrier = &colwise_reduce_bar[warpId];
  }

  if (leader) {
    if (warpId == 0) {
#pragma unroll
      for (int32_t i = 0; i < CastTraits::numStages; i++) {
        ptx::mbarrier_init(&producer[i], 1);
      }
    }
    if constexpr (CastTraits::_colwise_source_coming_from_rowwise &&
                  CastTraits::_colwise_reduce_max == ColwiseReduceMax::RedAsync) {
      ptx::mbarrier_init(colwise_reduce_barrier, 32);
    }

    ptx::fence_mbarrier_init_release_cluster();
  }
  __syncthreads();

  PipeState<CastTraits::numStages> states;

  int2 warp_coords;
  warp_coords.y = (warpId / CastTraits::warpLayout::N) * CastTraits::warpDim::M;
  warp_coords.x = (warpId % CastTraits::warpLayout::N) * CastTraits::warpDim::N;

  int32_t warp_base_offset = warp_coords.y * CastTraits::blockIterDim::N + warp_coords.x;

  int32_t thread_base_offset = (threadIdx.x / CastTraits::rowThreadLayout::N) *
                                   (CastTraits::blockIterDim::N / CastTraits::rowNumElemsPerUnit) +
                               (threadIdx.x % CastTraits::rowThreadLayout::N) *
                                   (CastTraits::rowChunkElems / CastTraits::rowNumElemsPerUnit);

  size_t rowwise_scale_base_offset =
      (block_coords.y + warp_coords.y + (threadIdx.x / CastTraits::rowThreadLayout::N)) *
          static_cast<size_t>(scale_stride_rowwise) +
      (block_coords.x + warp_coords.x +
       (threadIdx.x % CastTraits::rowThreadLayout::N) * CastTraits::rowChunkElems) /
          CastTraits::rowChunkElems;
  size_t colwise_scale_base_offset =
      ((block_coords.y + warp_coords.y +
        (threadIdx.x / CastTraits::colThreadLayout::N) * CastTraits::colChunkElems) /
       CastTraits::colChunkElems) *
          static_cast<size_t>(scale_stride_colwise) +
      (block_coords.x + warp_coords.x + (threadIdx.x % CastTraits::colThreadLayout::N));

  constexpr int32_t rowwise_scale_stride_in_smem =
      CastTraits::blockDIM::N / CastTraits::rowChunkElems;
  int32_t rowwise_scale_smem_base_offset =
      (warpId / CastTraits::warpLayout::N) * CastTraits::warpDim::M * rowwise_scale_stride_in_smem +
      (warpId % CastTraits::warpLayout::N) * (CastTraits::warpDim::N / CastTraits::rowChunkElems) +
      (threadIdx.x / CastTraits::rowThreadLayout::N) * rowwise_scale_stride_in_smem +
      (threadIdx.x % CastTraits::rowThreadLayout::N);

  if (warpId == 0 && leader) {
#pragma unroll 1
    for (int32_t iter = 0; iter < CastTraits::numStages - 1; iter++) {
      int32_t iter_m = iter / CastTraits::iterLayout::N;
      int32_t iter_n = iter % CastTraits::iterLayout::N;
      int2 coords;
      coords.y = block_coords.y + iter_m * CastTraits::blockIterDim::M;
      coords.x = block_coords.x + iter_n * CastTraits::blockIterDim::N;
      if (coords.x >= cols || coords.y >= rows) {
        break;
      }

      ptx::cp_async_bulk_tensor_2d_global_to_shared(
          reinterpret_cast<uint64_t *>(sInput + iter * CastTraits::blockIterDim::num),
          reinterpret_cast<const uint64_t *>(&tensor_map_input), static_cast<uint32_t>(coords.x),
          static_cast<uint32_t>(coords.y), &producer[iter]);
      ptx::mbarrier_arrive_expect_tx(&producer[iter],
                                     CastTraits::blockIterDim::num * sizeof(IType));
    }
  }
#pragma unroll 1
  for (int32_t iter = 0; iter < CastTraits::iterLayout::num; iter++) {
    {
      int32_t next = iter + (CastTraits::numStages - 1);
      int32_t next_stage = next % CastTraits::numStages;
      int32_t iter_m = next / CastTraits::iterLayout::N;
      int32_t iter_n = next % CastTraits::iterLayout::N;
      int2 coords;
      coords.y = block_coords.y + iter_m * CastTraits::blockIterDim::M;
      coords.x = block_coords.x + iter_n * CastTraits::blockIterDim::N;
      if (coords.x < cols && coords.y < rows) {
        if (warpId == 0 && leader) {
          if constexpr (CastTraits::_need_wait_group) {
            ptx::cp_async_bulk_wait_group_read<CastTraits::numStages - 1>();
          }

          ptx::cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(sInput + next_stage * CastTraits::blockIterDim::num),
              reinterpret_cast<const uint64_t *>(&tensor_map_input),
              static_cast<uint32_t>(coords.x), static_cast<uint32_t>(coords.y),
              &producer[next_stage]);
          ptx::mbarrier_arrive_expect_tx(&producer[next_stage],
                                         CastTraits::blockIterDim::num * sizeof(IType));
        }
      }
    }

    int32_t iter_m = iter / CastTraits::iterLayout::N;
    int32_t iter_n = iter % CastTraits::iterLayout::N;

    int2 coords;
    coords.y = block_coords.y + iter_m * CastTraits::blockIterDim::M;
    coords.x = block_coords.x + iter_n * CastTraits::blockIterDim::N;

    if (coords.x >= cols || coords.y >= rows) {
      break;
    }

    ptx::mbarrier_wait_parity(&producer[states.index()], states.phase());

    int32_t warp_offset = warp_base_offset + states.index() * CastTraits::blockIterDim::num;
    static_assert(CastTraits::_colwise_source_coming_from_rowwise);
    if constexpr (CastTraits::_colwise_source_coming_from_rowwise) {
      if constexpr (CastTraits::_need_smem_for_colwise_reduce &&
                    CastTraits::_colwise_reduce_max != ColwiseReduceMax::Redux) {
        sColwiseReduce[threadIdx.x] = 0.0f;
      }

      IType rInput[CastTraits::rowChunkElems];
      {
        inputUnitType *rInputUnit = reinterpret_cast<inputUnitType *>(rInput);
        int32_t base = thread_base_offset + warp_offset / CastTraits::rowNumElemsPerUnit;
#pragma unroll
        for (int32_t i = 0; i < CastTraits::rowNumUnitsPerChunk; i++) {
          rInputUnit[i] = sInputUnit[CastTraits::inputUnitSwz::swz(base + i)];
        }
      }

      if constexpr (std::is_same_v<IType, float>) {
        if constexpr (CastTraits::_colwise_reduce_max == ColwiseReduceMax::Atom ||
                      CastTraits::_colwise_reduce_max == ColwiseReduceMax::Red) {
        } else if constexpr (CastTraits::_colwise_reduce_max == ColwiseReduceMax::RedAsync) {
        } else if constexpr (CastTraits::_colwise_reduce_max == ColwiseReduceMax::Redux) {
        }
      } else {
        float row_scale_inverse;
        static_assert(CastTraits::_colwise_reduce_max == ColwiseReduceMax::Redux);
        if constexpr (CastTraits::_colwise_reduce_max == ColwiseReduceMax::Redux) {
          IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput);
          float2 *sColwiseReduce_2x = reinterpret_cast<float2 *>(sColwiseReduce);

          IType2 row_amax2{0.0f, 0.0f};
#pragma unroll
          for (int32_t i = 0; i < CastTraits::rowChunkElems / 2; i++) {
            ptx::abs_max_2x(row_amax2, row_amax2, rInput2[i]);

            ptx::floatx2 values = ptx::up_cast(rInput2[i]);

            float2 amaxs;
            ptx::reduce_sync_max_abs_f32(amaxs.x, values.x);
            ptx::reduce_sync_max_abs_f32(amaxs.y, values.y);

            if (leader) {
              sColwiseReduce_2x[i] = amaxs;
            }
          }

          {
            IType row_amax = ptx::get_amax(row_amax2.x, row_amax2.y);
            e8m0_t row_biased_exponent = to_e8m0<OType>(row_amax);
            row_scale_inverse = ptx::exp2f_rcp(row_biased_exponent);
            if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
              int32_t rowwise_scale_offset =
                  rowwise_scale_smem_base_offset +
                  iter_m * CastTraits::blockIterDim::M * rowwise_scale_stride_in_smem +
                  iter_n * (CastTraits::blockIterDim::N / CastTraits::rowChunkElems);
              sRowwiseScale[rowwise_scale_offset] = row_biased_exponent;
            } else {
              size_t rowwise_scale_offset =
                  rowwise_scale_base_offset +
                  iter_m * (CastTraits::blockIterDim::M) *
                      static_cast<size_t>(scale_stride_rowwise) +
                  iter_n * (CastTraits::blockIterDim::N / CastTraits::rowChunkElems);
              scales_rowwise[rowwise_scale_offset] = row_biased_exponent;
            }
          }
          {
            __syncwarp();
            float col_amax = sColwiseReduce[threadIdx.x];
            e8m0_t col_biased_exponent = to_e8m0<OType>(col_amax);
            float col_scale_inverse = ptx::exp2f_rcp(col_biased_exponent);
            sColwiseReduce[threadIdx.x] = col_scale_inverse;
            size_t colwise_scale_offset =
                colwise_scale_base_offset +
                iter_m * (CastTraits::blockIterDim::M / CastTraits::colChunkElems) *
                    static_cast<size_t>(scale_stride_colwise) +
                iter_n * CastTraits::blockIterDim::N;
            scales_colwise[colwise_scale_offset] = col_biased_exponent;
            __syncwarp();
          }
        }
        // row & colwise
        {
          rowOutputUnitType rRowOutputUnit[CastTraits::rowNumOutUnitsPerChunk];
          rowOutputUnitType rColOutputUnit[CastTraits::rowNumOutUnitsPerChunk];

          ptx::floatx2 row_scale_inverse_2{row_scale_inverse, row_scale_inverse};
          if constexpr (CastTraits::_use_cvt_4x) {
            using OType4 = ptx::FPx4<OType>;
            using IType4 = ptx::FPx4<IType>;

            ptx::floatx4 col_scale_inverse_4[2];
            ptx::floatx4 *sColwiseScale4x = reinterpret_cast<ptx::floatx4 *>(sColwiseReduce);
            col_scale_inverse_4[0] = sColwiseScale4x[0];

            IType4 *rInput4 = reinterpret_cast<IType4 *>(&rInput);
            OType4 *rRowOutput4 = reinterpret_cast<OType4 *>(&rRowOutputUnit);
            OType4 *rColOutput4 = reinterpret_cast<OType4 *>(&rColOutputUnit);
#pragma unroll
            for (int32_t i = 1; i < CastTraits::rowChunkElems / 4; i++) {
              {
                col_scale_inverse_4[i % 2] = sColwiseScale4x[i];
              }

              IType4 in = rInput4[i - 1];
              ptx::floatx4 in_fp4 = ptx::up_cast(in);

              OType4 row_out;
              ptx::mul_cvt_4x(row_out, in_fp4, row_scale_inverse_2);
              rRowOutput4[i - 1] = row_out;

              OType4 col_out;
              ptx::mul_cvt_4x(col_out, in_fp4, col_scale_inverse_4[(i - 1) % 2]);
              rColOutput4[i - 1] = col_out;
            }
            {
              constexpr int32_t i = (CastTraits::rowChunkElems / 4) - 1;
              IType4 in = rInput4[i];
              ptx::floatx4 in_fp4 = ptx::up_cast(in);

              OType4 row_out;
              ptx::mul_cvt_4x(row_out, in_fp4, row_scale_inverse_2);
              rRowOutput4[i] = row_out;

              OType4 col_out;
              ptx::mul_cvt_4x(col_out, in_fp4, col_scale_inverse_4[i % 2]);
              rColOutput4[i] = col_out;
            }
          } else {
            ptx::floatx2 col_scale_inverse_2[2];
            ptx::floatx2 *sColwiseScale2x = reinterpret_cast<ptx::floatx2 *>(sColwiseReduce);
            col_scale_inverse_2[0] = sColwiseScale2x[0];

            IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput);
            OType2 *rRowOutput2 = reinterpret_cast<OType2 *>(&rRowOutputUnit);
            OType2 *rColOutput2 = reinterpret_cast<OType2 *>(&rColOutputUnit);
#pragma unroll
            for (int32_t i = 1; i < CastTraits::rowChunkElems / 2; i++) {
              {
                col_scale_inverse_2[i % 2] = sColwiseScale2x[i];
              }

              IType2 in = rInput2[i - 1];
              ptx::floatx2 in_fp2 = ptx::up_cast(in);

              OType2 row_out;
              mul_cvt_2x(row_out, in_fp2, row_scale_inverse_2);
              rRowOutput2[i - 1] = row_out;

              OType2 col_out;
              mul_cvt_2x(col_out, in_fp2, col_scale_inverse_2[(i - 1) % 2]);
              rColOutput2[i - 1] = col_out;
            }
            {
              constexpr int32_t i = (CastTraits::rowChunkElems / 2) - 1;
              IType2 in = rInput2[i];
              ptx::floatx2 in_fp2 = ptx::up_cast(in);

              OType2 row_out;
              mul_cvt_2x(row_out, in_fp2, row_scale_inverse_2);
              rRowOutput2[i] = row_out;

              OType2 col_out;
              mul_cvt_2x(col_out, in_fp2, col_scale_inverse_2[i % 2]);
              rColOutput2[i] = col_out;
            }
          }

          {
            int32_t base = thread_base_offset / (CastTraits::rowOutNumElemsPerUnit /
                                                 CastTraits::rowNumElemsPerUnit) +
                           warp_offset / CastTraits::rowOutNumElemsPerUnit;
#pragma unroll
            for (int32_t i = 0; i < CastTraits::rowNumOutUnitsPerChunk; i++) {
              int32_t offset = CastTraits::rowOutputChunkSwz::swz(base + i);
              sRowOutputUnit[offset] = rRowOutputUnit[i];
              sColOutputUnit[offset] = rColOutputUnit[i];
            }
          }
        }
      }
    }
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    if (warpId == 0 && leader) {
      size_t gmem_offset = static_cast<size_t>(states.index()) * CastTraits::blockIterDim::num;
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_rowwise_output),
          static_cast<uint32_t>(coords.x), static_cast<uint32_t>(coords.y),
          reinterpret_cast<uint64_t *>(sRowOutput + gmem_offset));
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_colwise_output),
          static_cast<uint32_t>(coords.x), static_cast<uint32_t>(coords.y),
          reinterpret_cast<uint64_t *>(sColOutput + gmem_offset));
      ptx::cp_async_bulk_commit_group();
    }
    states++;
  }

  if constexpr (CastTraits::_cache_rowwise_scale_in_smem) {
    constexpr int32_t stride_in_smem = CastTraits::blockDIM::N / CastTraits::rowChunkElems;
    using PreferredDataType = std::conditional_t<
        stride_in_smem % 16 == 0, uint4,
        std::conditional_t<
            stride_in_smem % 8 == 0, uint2,
            std::conditional_t<stride_in_smem % 4 == 0, uint32_t,
                               std::conditional_t<stride_in_smem % 2 == 0, uint16_t, uint8_t>>>>;

    int2 end_coords;
    end_coords.y = std::min(block_coords.y + CastTraits::blockDIM::M, rows);
    end_coords.x = std::min((block_coords.x + CastTraits::blockDIM::N) / CastTraits::rowChunkElems,
                            scale_stride_rowwise);
    int2 valid_coords;
    valid_coords.y = end_coords.y - block_coords.y;
    valid_coords.x = end_coords.x - (block_coords.x / CastTraits::rowChunkElems);

    if (scale_stride_rowwise % sizeof(PreferredDataType) != 0) {
      using DataType = int32_t;
      constexpr int32_t num_elems_per_group = sizeof(DataType) / sizeof(e8m0_t);
      constexpr int32_t num_groups_per_row_in_smem = stride_in_smem / num_elems_per_group;

      int32_t num_threads_per_row = (valid_coords.x / num_elems_per_group);
      int32_t gmem_stride_in_group = scale_stride_rowwise / num_elems_per_group;

      DataType *sScales = reinterpret_cast<DataType *>(sRowwiseScale);
      DataType *gScales =
          reinterpret_cast<DataType *>(scales_rowwise + block_coords.y * scale_stride_rowwise +
                                       block_coords.x / CastTraits::rowChunkElems);

      for (int32_t i = threadIdx.x + warpId * 32; i < (valid_coords.y * num_threads_per_row);
           i += CastTraits::warpLayout::num * 32) {
        int32_t row = i / num_threads_per_row;
        int32_t col = i % num_threads_per_row;
        gScales[row * gmem_stride_in_group + col] = sScales[row * num_groups_per_row_in_smem + col];
      }
    } else {
      using DataType = PreferredDataType;
      constexpr int32_t num_elems_per_group = sizeof(DataType) / sizeof(e8m0_t);
      constexpr int32_t num_groups_per_row_in_smem = stride_in_smem / num_elems_per_group;

      int32_t num_threads_per_row = (valid_coords.x / num_elems_per_group);
      int32_t gmem_stride_in_group = scale_stride_rowwise / num_elems_per_group;

      DataType *sScales = reinterpret_cast<DataType *>(sRowwiseScale);
      DataType *gScales =
          reinterpret_cast<DataType *>(scales_rowwise + block_coords.y * scale_stride_rowwise +
                                       block_coords.x / CastTraits::rowChunkElems);

      for (int32_t i = threadIdx.x + warpId * 32; i < (valid_coords.y * num_threads_per_row);
           i += CastTraits::warpLayout::num * 32) {
        int32_t row = i / num_threads_per_row;
        int32_t col = i % num_threads_per_row;
        gScales[row * gmem_stride_in_group + col] = sScales[row * num_groups_per_row_in_smem + col];
      }
    }
  }

  ptx::cp_async_bulk_wait_group_read<0>();

#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // #ifndef TRANSFORMER_ENGINE_SPECIALIZED_QUANTIZE_MXFP8_CUH_
