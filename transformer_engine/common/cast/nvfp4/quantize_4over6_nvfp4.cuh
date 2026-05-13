/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_4over6_nvfp4.cuh
 *  \brief Dedicated kernels for NVFP4 4over6 quantization.
 *
 *  Four Over Six evaluates two TE-style NVFP4 encodings for every 1x16
 *  quantization group. The map-to-6 candidate uses the normal scale. The
 *  map-to-4 candidate expands the E4M3 block scale by 1.5x so FP4 value 4
 *  reaches the same range that FP4 value 6 reaches in the normal encoding.
 *  The selected candidate is the one with lower configured dequantization
 *  error; ties select map-to-6. The quantized candidates, dequantized values,
 *  and errors are kept in registers, matching the structure of the official
 *  Four Over Six implementation.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>

#include "../../common.h"
#include "../../util/math.h"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

#if FP4_TYPE_SUPPORTED

#define TRANSFORMER_ENGINE_NVFP4_4OVER6_ERR_MODE_SWITCH(ERR_MODE, ERR_MODE_CONST, ...) \
  switch (ERR_MODE) {                                                                  \
    case kNVTENVFP44Over6ErrMAE: {                                                     \
      constexpr NVTENVFP44Over6ErrMode ERR_MODE_CONST = kNVTENVFP44Over6ErrMAE;        \
      { __VA_ARGS__ }                                                                  \
    } break;                                                                           \
    case kNVTENVFP44Over6ErrMSE: {                                                     \
      constexpr NVTENVFP44Over6ErrMode ERR_MODE_CONST = kNVTENVFP44Over6ErrMSE;        \
      { __VA_ARGS__ }                                                                  \
    } break;                                                                           \
    default: {                                                                         \
      NVTE_ERROR("Unsupported NVFP4 4over6 error mode.");                              \
    }                                                                                  \
  }

#define TRANSFORMER_ENGINE_NVFP4_4OVER6_E4M3_MAX_SWITCH(E4M3_MAX_VALUE, E4M3_MAX_CONST, ...) \
  if ((E4M3_MAX_VALUE) == 256) {                                                             \
    constexpr int E4M3_MAX_CONST = 256;                                                      \
    { __VA_ARGS__ }                                                                          \
  } else {                                                                                   \
    NVTE_CHECK((E4M3_MAX_VALUE) == 448, "Unsupported NVFP4 E4M3 max.");                      \
    constexpr int E4M3_MAX_CONST = 448;                                                      \
    { __VA_ARGS__ }                                                                          \
  }

namespace quantize_4over6_kernel {

constexpr int kThreads = 128;
constexpr int kWarpThreads = 32;
constexpr int kGroupSize = 16;
constexpr int kTileRows = 128;
constexpr int kTileCols = 64;
constexpr int kTileColGroups = kTileCols / kGroupSize;
constexpr int kTileRowGroups = kTileRows / kGroupSize;
constexpr int kPipelineStages = 2;
constexpr int kStageRows = kTileRows / kPipelineStages;
constexpr int kStageRowGroups = kStageRows / kGroupSize;
constexpr int kElementsPerHalfGroup = 8;
constexpr int kPackedWordsPerGroup = 2;
static_assert(kTileRows == kPipelineStages * kStageRows);
static_assert(kStageRows % kGroupSize == 0);

template <NVTENVFP44Over6ErrMode kErrMode, bool kErrUseFastMath>
struct Config {
  static constexpr NVTENVFP44Over6ErrMode err_mode = kErrMode;
  static constexpr bool err_use_fast_math = kErrUseFastMath;
};

struct Candidate {
  uint32_t packed[kPackedWordsPerGroup];
  float err;
};

struct CandidatePair {
  Candidate map4;
  Candidate map6;
};

struct ScalePair {
  nvfp4_scale_t map4;
  nvfp4_scale_t map6;
  float inv_map4;
  float inv_map6;
};

template <NVTENVFP44Over6ErrMode kErrMode>
__device__ __forceinline__ float compute_error_rn(const float diff) {
  if constexpr (kErrMode == kNVTENVFP44Over6ErrMSE) {
    return __fmul_rn(diff, diff);
  } else if constexpr (kErrMode == kNVTENVFP44Over6ErrMAE) {
    return fabsf(diff);
  } else {
    NVTE_DEVICE_ERROR("Unsupported NVFP4 4over6 error mode.");
    return fabsf(diff);
  }
}

template <NVTENVFP44Over6ErrMode kErrMode>
__device__ __forceinline__ float compute_error(const float diff) {
  if constexpr (kErrMode == kNVTENVFP44Over6ErrMSE) {
    return diff * diff;
  } else if constexpr (kErrMode == kNVTENVFP44Over6ErrMAE) {
    return fabsf(diff);
  } else {
    NVTE_DEVICE_ERROR("Unsupported NVFP4 4over6 error mode.");
    return fabsf(diff);
  }
}

template <int E4M3_MAX>
__device__ __forceinline__ ScalePair compute_scale_pair(const float block_amax,
                                                        const float global_amax) {
  static_assert(E4M3_MAX == 448 || E4M3_MAX == 256, "Unsupported NVFP4 E4M3 max.");
  constexpr float fp4_max = detail::TypeExtrema<fp4e2m1>::max;  // 6.0f
  constexpr float fp8_max = detail::TypeExtrema<fp8e4m3>::max;  // 448.0f
  constexpr float expand_to_map4 = 1.5f;
  const float S_enc = core::compute_global_encode_scaling_factor_FP4<E4M3_MAX>(global_amax);
  const float base = block_amax / fp4_max * S_enc;

  ScalePair scales;
  scales.map4 = static_cast<nvfp4_scale_t>(fminf(base * expand_to_map4, fp8_max));
  scales.map6 = static_cast<nvfp4_scale_t>(fminf(base, fp8_max));

  const float S_dec = 1.0f / S_enc;
  scales.inv_map4 =
      fminf(1.0f / (static_cast<float>(scales.map4) * S_dec), detail::TypeExtrema<float>::max);
  scales.inv_map6 =
      fminf(1.0f / (static_cast<float>(scales.map6) * S_dec), detail::TypeExtrema<float>::max);
  return scales;
}

template <typename IType>
__device__ __forceinline__ float load_input(const IType *ptr, const size_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <typename IType>
__device__ __forceinline__ void load_row_group(const IType *tile, const int row,
                                               const int col_start, float (&x0)[8], float (&x1)[8],
                                               float *amax) {
  Vec<IType, kElementsPerHalfGroup> x0_vec;
  Vec<IType, kElementsPerHalfGroup> x1_vec;
  x0_vec.load_from(&tile[row * kTileCols + col_start]);
  x1_vec.load_from(&tile[row * kTileCols + col_start + kElementsPerHalfGroup]);

  *amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kElementsPerHalfGroup; ++i) {
    const float v0 = static_cast<float>(x0_vec.data.elt[i]);
    const float v1 = static_cast<float>(x1_vec.data.elt[i]);
    x0[i] = v0;
    x1[i] = v1;
    *amax = fmaxf(*amax, fabsf(v0));
    *amax = fmaxf(*amax, fabsf(v1));
  }
}

template <typename IType>
__device__ __forceinline__ void load_col_group(const IType *tile, const int row_start,
                                               const int col, float (&x0)[8], float (&x1)[8],
                                               float *amax) {
  *amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kElementsPerHalfGroup; ++i) {
    const float v0 = load_input(tile, (row_start + i) * kTileCols + col);
    const float v1 = load_input(tile, (row_start + i + kElementsPerHalfGroup) * kTileCols + col);
    x0[i] = v0;
    x1[i] = v1;
    *amax = fmaxf(*amax, fabsf(v0));
    *amax = fmaxf(*amax, fabsf(v1));
  }
}

template <typename Cfg, int E4M3_MAX, int SHIFT>
__device__ __forceinline__ void accumulate_dequant_error(const uint32_t dequant_bits, const float x,
                                                         const float sf, const float global_amax,
                                                         float *err) {
  constexpr float fp4_max = detail::TypeExtrema<fp4e2m1>::max;  // 6.0f
  constexpr float fp8_max = static_cast<float>(E4M3_MAX);
  constexpr float err_denom = fp4_max * fp8_max;
  const uint16_t half_bits = (dequant_bits >> SHIFT) & 0xFFFF;

  if constexpr (Cfg::err_use_fast_math) {
    const float dequant = __half2float(__ushort_as_half(half_bits));
    const float val = dequant * sf * global_amax / err_denom;
    const float diff = val - x;
    *err += compute_error<Cfg::err_mode>(diff);
  } else {
    const float dequant = __half2float(__ushort_as_half(half_bits));
    const float val = __fdiv_rn(__fmul_rn(__fmul_rn(dequant, sf), global_amax), err_denom);
    const float diff = __fsub_rn(val, x);
    *err = __fadd_rn(*err, compute_error_rn<Cfg::err_mode>(diff));
  }
}

template <typename Cfg, int E4M3_MAX>
__device__ __forceinline__ uint32_t cvt_fp32_to_fp4_8x_with_error(const float (&x)[8],
                                                                  const float block_scale_inverse,
                                                                  const nvfp4_scale_t sf,
                                                                  const float global_amax,
                                                                  float *err) {
  uint32_t out = 0;
  uint32_t out_dequant_1 = 0;
  uint32_t out_dequant_2 = 0;
  uint32_t out_dequant_3 = 0;
  uint32_t out_dequant_4 = 0;

  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %8, %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %10, %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %12, %11;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %4, byte3;\n"
        "}"
        : "=r"(out), "=r"(out_dequant_1), "=r"(out_dequant_2), "=r"(out_dequant_3),
          "=r"(out_dequant_4)
        : "f"(__fmul_rn(x[0], block_scale_inverse)), "f"(__fmul_rn(x[1], block_scale_inverse)),
          "f"(__fmul_rn(x[2], block_scale_inverse)), "f"(__fmul_rn(x[3], block_scale_inverse)),
          "f"(__fmul_rn(x[4], block_scale_inverse)), "f"(__fmul_rn(x[5], block_scale_inverse)),
          "f"(__fmul_rn(x[6], block_scale_inverse)), "f"(__fmul_rn(x[7], block_scale_inverse)));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }

  const float sf_float = static_cast<float>(sf);
  accumulate_dequant_error<Cfg, E4M3_MAX, 0>(out_dequant_1, x[0], sf_float, global_amax, err);
  accumulate_dequant_error<Cfg, E4M3_MAX, 16>(out_dequant_1, x[1], sf_float, global_amax, err);
  accumulate_dequant_error<Cfg, E4M3_MAX, 0>(out_dequant_2, x[2], sf_float, global_amax, err);
  accumulate_dequant_error<Cfg, E4M3_MAX, 16>(out_dequant_2, x[3], sf_float, global_amax, err);
  accumulate_dequant_error<Cfg, E4M3_MAX, 0>(out_dequant_3, x[4], sf_float, global_amax, err);
  accumulate_dequant_error<Cfg, E4M3_MAX, 16>(out_dequant_3, x[5], sf_float, global_amax, err);
  accumulate_dequant_error<Cfg, E4M3_MAX, 0>(out_dequant_4, x[6], sf_float, global_amax, err);
  accumulate_dequant_error<Cfg, E4M3_MAX, 16>(out_dequant_4, x[7], sf_float, global_amax, err);
  return out;
}

template <typename Cfg, int E4M3_MAX>
__device__ __forceinline__ CandidatePair make_candidates(const float (&x0)[8], const float (&x1)[8],
                                                         const ScalePair &scales,
                                                         const float global_amax) {
  CandidatePair candidates;
  candidates.map4.err = 0.0f;
  candidates.map6.err = 0.0f;
  candidates.map4.packed[0] = cvt_fp32_to_fp4_8x_with_error<Cfg, E4M3_MAX>(
      x0, scales.inv_map4, scales.map4, global_amax, &candidates.map4.err);
  candidates.map6.packed[0] = cvt_fp32_to_fp4_8x_with_error<Cfg, E4M3_MAX>(
      x0, scales.inv_map6, scales.map6, global_amax, &candidates.map6.err);
  candidates.map4.packed[1] = cvt_fp32_to_fp4_8x_with_error<Cfg, E4M3_MAX>(
      x1, scales.inv_map4, scales.map4, global_amax, &candidates.map4.err);
  candidates.map6.packed[1] = cvt_fp32_to_fp4_8x_with_error<Cfg, E4M3_MAX>(
      x1, scales.inv_map6, scales.map6, global_amax, &candidates.map6.err);
  return candidates;
}

__device__ __forceinline__ float reduce_group_sum_16(float value) {
  const int lane = threadIdx.x & (kWarpThreads - 1);
  const int group_base = lane & ~(kGroupSize - 1);
  const unsigned mask = 0xffffu << group_base;
#pragma unroll
  for (int offset = kGroupSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset, kGroupSize);
  }
  return __shfl_sync(mask, value, group_base, kWarpThreads);
}

__device__ __forceinline__ float reduce_group_max_16(float value) {
  const int lane = threadIdx.x & (kWarpThreads - 1);
  const int group_base = lane & ~(kGroupSize - 1);
  const unsigned mask = 0xffffu << group_base;
#pragma unroll
  for (int offset = kGroupSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(mask, value, offset, kGroupSize));
  }
  return __shfl_sync(mask, value, group_base, kWarpThreads);
}

__device__ __forceinline__ void store_packed_group(const uint32_t *packed, fp4e2m1x2 *dst) {
  const uint64_t packed64 =
      static_cast<uint64_t>(packed[0]) | (static_cast<uint64_t>(packed[1]) << 32);
  *reinterpret_cast<uint64_t *>(dst) = packed64;
}

__device__ __forceinline__ const uint32_t *select_packed(const CandidatePair &candidates,
                                                         const bool pick_map4) {
  if (pick_map4) {
    return candidates.map4.packed;
  }
  return candidates.map6.packed;
}

__device__ __forceinline__ nvfp4_scale_t select_scale(const ScalePair &scales,
                                                      const bool pick_map4) {
  if (pick_map4) {
    return scales.map4;
  }
  return scales.map6;
}

__device__ __forceinline__ void cp_async_cg_16(void *dst, const void *src) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const uint32_t dst_smem_ptr = __cvta_generic_to_shared(dst);
  const uint64_t src_gmem_ptr = reinterpret_cast<uint64_t>(src);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem_ptr),
               "l"(src_gmem_ptr));
#else
  NVTE_DEVICE_ERROR("cp.async is only supported on SM 8.0+.");
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#else
  NVTE_DEVICE_ERROR("cp.async is only supported on SM 8.0+.");
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#else
  NVTE_DEVICE_ERROR("cp.async is only supported on SM 8.0+.");
#endif
}

template <typename IType>
__device__ void load_stage_to_shared_async(const IType *input, IType *tile, const size_t rows,
                                           const size_t cols, const size_t stage_row,
                                           const size_t tile_col) {
  constexpr int vec_elems = 16 / sizeof(IType);
  constexpr int vecs_per_row = kTileCols / vec_elems;
  constexpr int vecs = kStageRows * vecs_per_row;
  using TileVec = Vec<IType, vec_elems>;

  for (int idx = threadIdx.x; idx < vecs; idx += blockDim.x) {
    const int local_row = idx / vecs_per_row;
    const int local_vec_col = idx - local_row * vecs_per_row;
    const int local_col = local_vec_col * vec_elems;
    const size_t global_row = stage_row + local_row;
    const size_t global_col = tile_col + local_col;
    IType *stage_ptr = &tile[local_row * kTileCols + local_col];

    if (global_row < rows && global_col + vec_elems <= cols) {
      cp_async_cg_16(stage_ptr, &input[global_row * cols + global_col]);
    } else {
      TileVec vec;
      vec.clear();
#pragma unroll
      for (int i = 0; i < vec_elems; ++i) {
        if (global_row < rows && global_col + i < cols) {
          vec.data.elt[i] = input[global_row * cols + global_col + i];
        }
      }
      vec.store_to(stage_ptr);
    }
  }
}

template <bool USE_2D_QUANTIZATION, bool ROW_SCALED_NVFP4, typename Cfg, int E4M3_MAX,
          typename IType>
__device__ void quantize_stage_rowwise(const IType *tile, fp4e2m1x2 *output, nvfp4_scale_t *scales,
                                       const float *amax, const size_t rows, const size_t cols,
                                       const size_t stage_row, const size_t tile_col,
                                       const size_t scale_stride) {
  constexpr int groups = kStageRows * kTileColGroups;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    const int local_row = group % kStageRows;
    const int local_col_group = group / kStageRows;
    const int local_col = local_col_group * kGroupSize;
    const size_t global_row = stage_row + local_row;
    const size_t global_col = tile_col + local_col;
    if (global_row >= rows || global_col >= cols) {
      continue;
    }

    float x0[8];
    float x1[8];
    float group_amax = 0.0f;
    load_row_group(tile, local_row, local_col, x0, x1, &group_amax);

    float block_amax = group_amax;
    if constexpr (USE_2D_QUANTIZATION) {
      block_amax = reduce_group_max_16(group_amax);
    }

    float global_amax = amax[0];
    if constexpr (ROW_SCALED_NVFP4) {
      global_amax = amax[global_row];
    }

    const ScalePair scale_pair = compute_scale_pair<E4M3_MAX>(block_amax, global_amax);
    CandidatePair candidates = make_candidates<Cfg, E4M3_MAX>(x0, x1, scale_pair, global_amax);

    float err_map4 = candidates.map4.err;
    float err_map6 = candidates.map6.err;
    if constexpr (USE_2D_QUANTIZATION) {
      err_map4 = reduce_group_sum_16(err_map4);
      err_map6 = reduce_group_sum_16(err_map6);
    }

    const bool pick_map4 = err_map4 < err_map6;
    const nvfp4_scale_t selected_scale = select_scale(scale_pair, pick_map4);
    const uint32_t *selected = select_packed(candidates, pick_map4);

    const size_t global_col_group = global_col / kGroupSize;
    scales[global_row * scale_stride + global_col_group] = selected_scale;
    store_packed_group(selected, &output[(global_row * cols + global_col) / 2]);
  }
}

template <bool USE_2D_QUANTIZATION, typename Cfg, int E4M3_MAX, typename IType>
__device__ void quantize_stage_colwise(const IType *tile, fp4e2m1x2 *output_t,
                                       nvfp4_scale_t *scales_t, const float *amax,
                                       const size_t rows, const size_t cols, const size_t stage_row,
                                       const size_t tile_col, const size_t scale_stride_t) {
  constexpr int groups = kStageRowGroups * kTileCols;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    const int local_row_group = group / kTileCols;
    const int local_col = group - local_row_group * kTileCols;
    const int local_row = local_row_group * kGroupSize;
    const size_t global_row = stage_row + local_row;
    const size_t global_col = tile_col + local_col;
    if (global_row >= rows || global_col >= cols) {
      continue;
    }

    float x0[8];
    float x1[8];
    float group_amax = 0.0f;
    load_col_group(tile, local_row, local_col, x0, x1, &group_amax);

    float block_amax = group_amax;
    if constexpr (USE_2D_QUANTIZATION) {
      block_amax = reduce_group_max_16(group_amax);
    }

    const float global_amax = amax[0];
    const ScalePair scale_pair = compute_scale_pair<E4M3_MAX>(block_amax, global_amax);
    CandidatePair candidates = make_candidates<Cfg, E4M3_MAX>(x0, x1, scale_pair, global_amax);

    float err_map4 = candidates.map4.err;
    float err_map6 = candidates.map6.err;
    if constexpr (USE_2D_QUANTIZATION) {
      err_map4 = reduce_group_sum_16(err_map4);
      err_map6 = reduce_group_sum_16(err_map6);
    }

    const bool pick_map4 = err_map4 < err_map6;
    const nvfp4_scale_t selected_scale = select_scale(scale_pair, pick_map4);
    const uint32_t *selected = select_packed(candidates, pick_map4);

    const size_t global_row_group = global_row / kGroupSize;
    scales_t[global_col * scale_stride_t + global_row_group] = selected_scale;
    store_packed_group(selected, &output_t[(global_col * rows + global_row) / 2]);
  }
}

template <bool USE_2D_QUANTIZATION, bool RETURN_IDENTITY, bool RETURN_TRANSPOSE,
          bool ROW_SCALED_NVFP4, typename Cfg, int E4M3_MAX, typename IType>
__global__ void __launch_bounds__(kThreads)
    quantize_4over6_kernel(const IType *input, fp4e2m1x2 *output, fp4e2m1x2 *output_t,
                           nvfp4_scale_t *scales, nvfp4_scale_t *scales_t,
                           const float *amax_rowwise, const float *amax_colwise, const size_t rows,
                           const size_t cols, const size_t scale_stride,
                           const size_t scale_stride_t, const float *noop) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  extern __shared__ char dynamic_shmem[];
  auto *tiles = reinterpret_cast<IType *>(dynamic_shmem);
  const size_t tile_col = blockIdx.x * kTileCols;
  const size_t tile_row = blockIdx.y * kTileRows;

  IType *stage_tiles[kPipelineStages] = {
      &tiles[0],
      &tiles[kStageRows * kTileCols],
  };

  load_stage_to_shared_async(input, stage_tiles[0], rows, cols, tile_row, tile_col);
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();

  for (int stage = 0; stage < kPipelineStages; ++stage) {
    const int next_stage = stage + 1;
    if (next_stage < kPipelineStages) {
      const size_t next_stage_row = tile_row + next_stage * kStageRows;
      load_stage_to_shared_async(input, stage_tiles[next_stage], rows, cols, next_stage_row,
                                 tile_col);
      cp_async_commit_group();
    }

    const size_t stage_row = tile_row + stage * kStageRows;
    IType *stage_tile = stage_tiles[stage];

    if constexpr (RETURN_IDENTITY) {
      quantize_stage_rowwise<USE_2D_QUANTIZATION, ROW_SCALED_NVFP4, Cfg, E4M3_MAX>(
          stage_tile, output, scales, amax_rowwise, rows, cols, stage_row, tile_col, scale_stride);
    }

    if constexpr (RETURN_TRANSPOSE) {
      const float *columnwise_amax = amax_colwise;
      if (columnwise_amax == nullptr) {
        columnwise_amax = amax_rowwise;
      }
      quantize_stage_colwise<USE_2D_QUANTIZATION, Cfg, E4M3_MAX>(
          stage_tile, output_t, scales_t, columnwise_amax, rows, cols, stage_row, tile_col,
          scale_stride_t);
    }

    if (next_stage < kPipelineStages) {
      cp_async_wait_group<0>();
      __syncthreads();
    }
  }
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool USE_2D_QUANTIZATION, typename Cfg, int E4M3_MAX, typename IType>
void launch_quantize_4over6(const Tensor &input, const Tensor *noop, Tensor *output,
                            cudaStream_t stream) {
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const bool row_scaled_nvfp4 = output->row_scaled_nvfp4;
  const bool return_identity = output->has_data();
  const bool return_transpose = output->has_columnwise_data();

  const auto *input_ptr = reinterpret_cast<const IType *>(input.data.dptr);
  auto *output_ptr = reinterpret_cast<fp4e2m1x2 *>(output->data.dptr);
  auto *output_t_ptr = reinterpret_cast<fp4e2m1x2 *>(output->columnwise_data.dptr);
  auto *scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  auto *scales_t_ptr = reinterpret_cast<nvfp4_scale_t *>(output->columnwise_scale_inv.dptr);
  const auto *amax_rowwise_ptr = reinterpret_cast<const float *>(output->amax.dptr);
  const auto *amax_colwise_ptr = reinterpret_cast<const float *>(output->columnwise_amax.dptr);
  const auto *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  const dim3 grid(DIVUP(cols, static_cast<size_t>(kTileCols)),
                  DIVUP(rows, static_cast<size_t>(kTileRows)));
  const dim3 block(kThreads);
  const size_t shmem = kPipelineStages * kStageRows * kTileCols * sizeof(IType);
  const size_t scale_stride = return_identity ? output->scale_inv.shape[1] : 0;
  const size_t scale_stride_t = return_transpose ? output->columnwise_scale_inv.shape[1] : 0;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(return_identity, RETURN_IDENTITY, {
    TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE, {
      TRANSFORMER_ENGINE_SWITCH_CONDITION(row_scaled_nvfp4, ROW_SCALED_NVFP4, {
        auto kernel = quantize_4over6_kernel<USE_2D_QUANTIZATION, RETURN_IDENTITY, RETURN_TRANSPOSE,
                                             ROW_SCALED_NVFP4, Cfg, E4M3_MAX, IType>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
        kernel<<<grid, block, shmem, stream>>>(input_ptr, output_ptr, output_t_ptr, scales_ptr,
                                               scales_t_ptr, amax_rowwise_ptr, amax_colwise_ptr,
                                               rows, cols, scale_stride, scale_stride_t, noop_ptr);
      });
    });
  });
}

}  // namespace quantize_4over6_kernel

#endif  // FP4_TYPE_SUPPORTED

template <bool use_2d_quantization>
void quantize_4over6(const Tensor &input, const Tensor *noop, Tensor *output,
                     const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_4over6_kernel;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(quant_config != nullptr && quant_config->nvfp4_4over6,
             "NVFP4 4over6 quantization requires an enabled quantization config.");
  NVTE_CHECK(!quant_config->stochastic_rounding,
             "NVFP4 4over6 quantization does not support stochastic rounding.");
  NVTE_CHECK(quant_config->nvfp4_e4m3_max == output->nvfp4_e4m3_max,
             "Tensor and quantization config have inconsistent options for NVFP4 4over6 "
             "E4M3 scale bound.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "NVFP4 4over6 output tensor must have rowwise or columnwise data.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales, "Output must have scales in compact format.");
  NVTE_CHECK(input.flat_last_dim() % kGroupSize == 0,
             "NVFP4 4over6 quantization requires columns divisible by ", kGroupSize, ".");
  NVTE_CHECK(!(output->has_columnwise_data() || use_2d_quantization) ||
                 input.flat_first_dim() % kGroupSize == 0,
             "NVFP4 4over6 columnwise or 2D quantization requires rows divisible by ", kGroupSize,
             ".");
  NVTE_CHECK(!output->row_scaled_nvfp4 || !use_2d_quantization,
             "Row-scaled NVFP4 quantization does not support 2D quantization.");
  NVTE_CHECK(!output->row_scaled_nvfp4 || !output->has_columnwise_data(),
             "Row-scaled NVFP4 quantization does not produce columnwise output.");
  NVTE_CHECK(!use_2d_quantization || output->has_data(),
             "NVFP4 4over6 2D quantization requires rowwise output.");

  if (output->has_data()) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
    NVTE_CHECK(output->amax.dptr != nullptr, "Rowwise amax tensor must be allocated.");
    NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  }
  if (output->has_columnwise_data()) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Transposed scaling tensor must be allocated.");
    NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
               "Transposed output must have FP4 type.");
    NVTE_CHECK(output->columnwise_amax.dptr != nullptr || output->amax.dptr != nullptr,
               "NVFP4 4over6 columnwise quantization requires columnwise amax or rowwise amax.");
  }

  TRANSFORMER_ENGINE_NVFP4_4OVER6_E4M3_MAX_SWITCH(
      quant_config->nvfp4_e4m3_max, E4M3_MAX,
      TRANSFORMER_ENGINE_NVFP4_4OVER6_ERR_MODE_SWITCH(
          quant_config->nvfp4_4over6_err_mode, ERR_MODE,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              quant_config->nvfp4_4over6_err_use_fast_math, ERR_USE_FAST_MATH, {
                using Cfg = quantize_4over6_kernel::Config<ERR_MODE, ERR_USE_FAST_MATH>;
                TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
                    input.dtype(), IType,
                    quantize_4over6_kernel::launch_quantize_4over6<use_2d_quantization, Cfg,
                                                                   E4M3_MAX, IType>(
                        input, noop, output, stream););
              });););

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
