/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nvfp4_transpose.cuh
 *  \brief CUDA kernels to cast to NVFP4 and transpose.
 */

#ifndef TRANSFORMER_ENGINE_NVFP4_TRANSPOSE_CUH_
#define TRANSFORMER_ENGINE_NVFP4_TRANSPOSE_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#if CUDA_VERSION > 12080
#include <cuda_fp4.h>
#endif  // CUDA_VERSION > 12080

#include <cfloat>

#include "../common.h"
#include "../utils.cuh"
#include "curanddx.hpp"
#include "math.h"
#include "ptx.cuh"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {

#if CUDA_VERSION > 12080
namespace nvfp4_transpose {

using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() + curanddx::PhiloxRounds<10>() +
                     curanddx::SM<800>() + curanddx::Thread());

using namespace ptx;
using nvfp4_scale_t = fp8e4m3;

constexpr size_t SCALE_DIM = 16;  // NVFP4 block (x16 elts)

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_NUM = 128;

constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;

constexpr size_t SCALES_PER_THREAD = 2 * (CHUNK_DIM_Y * CHUNK_DIM_X) / SCALE_DIM / THREADS_NUM;
constexpr size_t RNG_GENS_PER_THREAD =
    SCALES_PER_THREAD / 4;  // Each call generates 4x uint32_t random numbers

constexpr size_t TILE_DIM_Y = 32;
constexpr size_t TILE_DIM_X = 128;

// SHould this be SCALE_DIM or BLOCK_DIM? Both are 16, should work for both 1D and 2D
constexpr size_t SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;
constexpr size_t SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;  // 128 / 16 =  8

constexpr size_t TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;
constexpr size_t TILES_X = CHUNK_DIM_X / TILE_DIM_X;
constexpr size_t STAGES = TILES_Y * TILES_X;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t BUFF_DIM_Y = TILE_DIM_Y;
constexpr size_t BUFF_DIM_X = TILE_DIM_X;
constexpr size_t BUFF_SIZE = BUFF_DIM_Y * BUFF_DIM_X;
constexpr size_t BUFF_SIZE_TOTAL = BUFF_SIZE * BUFFS_NUM;

// Input buffer (BF16)
constexpr size_t BUFF_IN_DIM_Y = BUFF_DIM_Y;
constexpr size_t BUFF_IN_DIM_X = BUFF_DIM_X;
constexpr size_t BUFF_IN_SIZE = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;

// Output buffer (NVFP4)
constexpr size_t BUFF_OUT_DIM_Y = BUFF_DIM_Y;
constexpr size_t BUFF_OUT_DIM_X = (BUFF_DIM_X * 4) / 8;
constexpr size_t BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

// Output transpose buffer (NVFP4)
constexpr size_t BUFF_OUT_T_DIM_Y = BUFF_DIM_X;
constexpr size_t BUFF_OUT_T_DIM_X = (BUFF_DIM_Y * 4) / 8;
constexpr size_t BUFF_OUT_T_SIZE = BUFF_OUT_T_DIM_Y * BUFF_OUT_T_DIM_X;

// Manual swizzling parameters to reduce SHMEM bank conflicts
constexpr size_t PACK_SIZE = 8;
constexpr size_t WAVES = SCALE_DIM / PACK_SIZE;

constexpr size_t SCALING_FACTORS_PER_TILE_X = TILE_DIM_X / SCALE_DIM;
constexpr size_t THREADS_X_ROWWISE = SCALING_FACTORS_PER_TILE_X;       // 128 / 16 = 8
constexpr size_t THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;  // 128 / 8 = 16

constexpr size_t ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;  // 32/ 16 = 2
constexpr size_t ITERATIONS_TRANSPOSE = BUFF_IN_DIM_Y / SCALE_DIM;
constexpr size_t BUFF_OUT_IT_OFFSET = BUFF_OUT_T_DIM_X / ITERATIONS_TRANSPOSE;

static_assert(BUFF_DIM_Y >= SCALE_DIM &&
              "Number of buffer rows must be greater or equal to the size of the columwise "
              "scaling block\0");
static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y);
static_assert(BUFF_DIM_Y >= THREADS_Y_ROWWISE &&
              "Number of buffer rows must be greater or equal to the number of rowwise "
              "processing threads in Y dimension\0");

// Number of 4-bit elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;  // 256

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM;  // 8 = 128 / 16

// Compute per-block E4M3 encoding/decoding scaling factor
__device__ __forceinline__ nvfp4_scale_t compute_decoding_scaling_factor(const float block_amax,
                                                                         const float S_enc) {
  // constexpr float rcp_6f = 1.0f / 6.0f;
  // const float S_dec_b = block_amax * rcp_6f;
  // const nvfp4_scale_t S_dec_b_fp8 = static_cast<nvfp4_scale_t>(S_dec_b * S_enc);
  // return S_dec_b_fp8;
  // NOTE: Divide by 6.0f is not elegant and not efficient.
  // However, this is part of the emulation code to ensure exact match.
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f;
  const float S_dec_b = block_amax / fp4_max * S_enc;
  return static_cast<nvfp4_scale_t>(fminf(S_dec_b, TypeExtrema<float>::max));
}

// Compute the global encode scale factor for a given global amax
__device__ __forceinline__ float compute_global_encode_scaling_factor_FP4(const float global_amax) {
  using namespace detail;
  constexpr float fp8_max = TypeExtrema<fp8e4m3>::max;  // 448.0f;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale = fminf(global_encode_scale, TypeExtrema<float>::max);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

__device__ __forceinline__ uint32_t get_rbits(RNG &rng, uint4 &random_uint4, int &rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    curanddx::uniform_bits dist;
    random_uint4 = dist.generate4(rng);
  }
  // Treat uint4 as an array of 4x uint32_t elements for indexing
  const uint32_t *const rbits_arr = reinterpret_cast<uint32_t *>(&random_uint4);
  const uint32_t rbits = rbits_arr[rnd_idx++];
  return rbits;
}

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b16 v0_bf16; \n\t"
      ".reg.b16 v1_bf16; \n\t"
      ".reg.b16 v2_bf16; \n\t"
      ".reg.b16 v3_bf16; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16} , %1; \n\t"
      "cvt.f32.bf16 v0, v0_bf16; \n\t"
      "cvt.f32.bf16 v1, v1_bf16; \n\t"
      "cvt.f32.bf16 v2, v2_bf16; \n\t"
      "cvt.f32.bf16 v3, v3_bf16; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %2; \n\t"  // mind the shuffled elements order
      "mul.f32x2 v23, v23, %2; \n\t"  // mind the shuffled elements order
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %3; \n\t"  // mind the shuffled elements order
      "}"
      : "=h"(out_4x)
      : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)), "r"(rbits));
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_rn(const uint64_t in_4x,
                                                                    const float2 scale,
                                                                    const uint32_t rbits) {
  // NOTE: rbits unused for rn.
  uint32_t out_4x = 0;  // Only need 16 bit. Using 32 bit container for packing.
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b16 v0_bf16; \n\t"
      ".reg.b16 v1_bf16; \n\t"
      ".reg.b16 v2_bf16; \n\t"
      ".reg.b16 v3_bf16; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      ".reg.b8 f0; \n\t"
      ".reg.b8 f1; \n\t"
      "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16} , %1; \n\t"
      "cvt.f32.bf16 v0, v0_bf16; \n\t"
      "cvt.f32.bf16 v1, v1_bf16; \n\t"
      "cvt.f32.bf16 v2, v2_bf16; \n\t"
      "cvt.f32.bf16 v3, v3_bf16; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %2; \n\t"  // mind the shuffled elements order
      "mul.f32x2 v23, v23, %2; \n\t"  // mind the shuffled elements order
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x(const uint64_t in_4x,
                                                            const float2 scale,
                                                            const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(in_4x, scale, rbits);
  } else {
    return mul_cvt_bf16_to_fp4_4x_with_rn(in_4x, scale, rbits);
  }
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      "mov.b64 {v0, v1} , %1; \n\t"
      "mov.b64 {v2, v3} , %2; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %3; \n\t"  // mind the shuffled elements order
      "mul.f32x2 v23, v23, %3; \n\t"  // mind the shuffled elements order
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %4; \n\t"  // mind the shuffled elements order
      "}"
      : "=h"(out_4x)
      : "l"(reinterpret_cast<const uint64_t &>(in01)),
        "l"(reinterpret_cast<const uint64_t &>(in23)),
        "l"(reinterpret_cast<const uint64_t &>(scale)), "r"(rbits));
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_rn(const float2 in01,
                                                                    const float2 in23,
                                                                    const float2 scale,
                                                                    const uint32_t rbits) {
  // NOTE: rbits unused for rn.
  uint32_t out_4x = 0;  // Only need 16 bit. Using 32 bit container for packing.
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      ".reg.b8 f0; \n\t"
      ".reg.b8 f1; \n\t"
      "mov.b64 {v0, v1} , %1; \n\t"
      "mov.b64 {v2, v3} , %2; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %3; \n\t"  // mind the shuffled elements order
      "mul.f32x2 v23, v23, %3; \n\t"  // mind the shuffled elements order
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "l"(reinterpret_cast<const uint64_t &>(in01)),
        "l"(reinterpret_cast<const uint64_t &>(in23)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x(const float2 in01, const float2 in23,
                                                            const float2 scale,
                                                            const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(in01, in23, scale, rbits);
  } else {
    return mul_cvt_fp32_to_fp4_4x_with_rn(in01, in23, scale, rbits);
  }
}

#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, bool USE_STOCHASTIC_ROUNDING, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS_NUM)
    nvfp4_transpose_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                           const __grid_constant__ CUtensorMap tensor_map_output,
                           const __grid_constant__ CUtensorMap tensor_map_output_t,
                           nvfp4_scale_t *const scales_ptr, nvfp4_scale_t *const scales_t_ptr,
                           const float *noop, const float *const amax_rowwise_ptr,
                           const float *const amax_colwise_ptr, const size_t rows,
                           const size_t cols, const size_t scale_stride,
                           const size_t scale_stride_t, const size_t rng_seed,
                           const size_t rng_sequence) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool NO_ACTIVATIONS_NOT_FP32_INPUT =
      (!COMPUTE_ACTIVATIONS) && (!std::is_same_v<IType, float>);

  using IType2 = typename ptx::FPx2<IType>;

  if constexpr (!COMPUTE_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }

  const size_t rng_offest = threadIdx.x;
  RNG rng(rng_seed, rng_sequence, rng_offest);  // seed, sequence, offset
  curanddx::uniform_bits dist;
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? dist.generate4(rng) : uint4{0, 0, 0, 0};
  int rnd_idx =
      0;  // Index of the random number. It increments each time when used and resets to 0 if reaches 4x

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS;

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;

  const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y;

  const size_t chunk_rows = rows - block_offset_Y;

  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * SCALES_PER_CHUNK_X;
  const size_t scales_block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t scales_block_offset_X_t = blockIdx.y * SCALES_PER_CHUNK_Y;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const size_t tid_X_colwise = threadIdx.x;
  const size_t tid_Y_t = tid_X_colwise;
  // const size_t tid_X_t = 0;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const size_t row_base_colwise = block_offset_Y;
  const size_t col_base_colwise = block_offset_X + thread_offset_X_colwise;

  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_t = scales_block_offset_Y_t + tid_Y_t;
  const size_t scales_offset_X_t = scales_block_offset_X_t;

  const size_t SFs_per_row = cols / SCALE_DIM;

  const bool rowwise_scale_is_within_bounds_X = scales_offset_X_rowwise < SFs_per_row;
  const bool colwise_scale_is_within_bounds_Y = scales_offset_Y_t < cols;

  // Helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_mem_rowwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_colwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_rowwise_scales = 0;

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *out_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *out_t_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);

  nvfp4_scale_t *out_rowwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
  nvfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);
  IType *cached_act_sh = in_sh;  // in_sh is used as a cache buffer

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  // Compute a global encoding/decoding scaling factors for all S_dec_b
  const float S_enc_rowwise = (amax_rowwise_ptr == nullptr)
                                  ? 1.0f
                                  : compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);
  // NOTE: This is to match with how emulation code was written.
  const float S_dec_rowwise = 1.0 / S_enc_rowwise;

  const float S_enc_colwise = (amax_colwise_ptr == nullptr)
                                  ? S_enc_rowwise
                                  : compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);
  const float S_dec_colwise = 1.0 / S_enc_colwise;

  float thread_amax = 0.0f;

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, shmem_buff_size,
                    &mbar[0], is_master_thread);

#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    const size_t buff_offset_in = buff * BUFF_IN_SIZE;
    const size_t buff_offset_out = buff * BUFF_OUT_SIZE;
    const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

    if (next_stage < STAGES) {
      // Wait for TMA transfer to have finished reading shared memory.
      // I.e. the buffer is ready to be written to
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t next_buff_offset = next_buff * BUFF_IN_SIZE;

      copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                        global_offset_Y, shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;

    // COLWISE scaling
    if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
        const size_t in_thread_offset_Y = 0 + it * SCALE_DIM;
        const size_t in_thread_offset_X = thread_offset_X_colwise;

        const size_t out_t_thread_offset_Y = thread_offset_X_colwise;
        const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;

        const size_t shmem_offset_base_colwise_in =
            buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;
        const size_t shmem_offset_base_colwise_out_t =
            buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

        block_amax = 0.0f;
        float in_compute_colwise[SCALE_DIM];
        IType in_colwise_IType[SCALE_DIM];
        // 1. Read/Compute elements. Find NVFP4-block AMAX
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType block_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            in_colwise_IType[i] = in_sh[shmem_offset_colwise];
            block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
          }
          block_amax = static_cast<float>(block_amax_f16);
        } else {
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            float elt = static_cast<float>(in_sh[shmem_offset_colwise]);
            if constexpr (COMPUTE_ACTIVATIONS) {
              elt = OP(elt, {});
            }
            // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            // Cache computed activations to avoid computing them again in the 2nd pass along another dimension
            if constexpr (IS_CACHED_ACT_OP) {
              cached_act_sh[shmem_offset_colwise] = static_cast<IType>(elt);
            }
            if constexpr (COMPUTE_ACTIVATIONS) {
              const bool row_out_of_bounds_colwise =
                  (row_base_colwise + stage_offset_Y + i >= rows);
              const bool out_of_bounds = (col_out_of_bounds_colwise || row_out_of_bounds_colwise);
              if (!out_of_bounds) {
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
            } else {
              // If no activation, elt is 0 so we can safely do this
              block_amax = fmaxf(block_amax, fabsf(elt));
            }
            in_compute_colwise[i] = elt;
          }
        }
        // 2. Compute E4M3 scaling factor
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_colwise);

        // Store scaling factors through SHMEM
        const size_t scale_idx_sh =
            tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
        out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

        // Compute "correct" per-block encoding scaling factor
        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);  // S_enc_b_fp8
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

        // 3. Scale elements
        fp4e2m1x4 regs[SCALE_DIM / 4];

#pragma unroll
        for (int e = 0; e < SCALE_DIM / 4; ++e) {
          const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
          if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
            const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_colwise_IType[4 * e]);
            regs[e] = mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(elts, block_scale_inverse_2x,
                                                                      rbits);
          } else {
            const float2 in01 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e]);
            const float2 in23 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e + 2]);
            regs[e] = mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                in01, in23, block_scale_inverse_2x, rbits);
          }
        }

        const int group = thread_lane / 16;
        uint32_t val[2];
        uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

        // Helps reducing bank conflicts
        switch (group) {
          case 0:
            val[0] = regs_4x[0];
            val[1] = regs_4x[1];
            break;
          case 1:
            val[0] = regs_4x[1];
            val[1] = regs_4x[0];

            break;
        }
        uint32_t *out_t_data_sh_as_uint32_t =
            reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);
        out_t_data_sh_as_uint32_t[group] = val[0];            // idx1 = (group + 0) % 2;
        out_t_data_sh_as_uint32_t[(group + 1) & 1] = val[1];  // idx2 = (group + 1) % 2;
      }
    }

    // ROWWISE scaling
    {
      const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
        const size_t it_thread_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const size_t shmem_offset_base_rowwise_in =
            buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
        const size_t shmem_offset_base_rowwise_out =
            buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

        const size_t it_offset_Y = stage_offset_Y + it * THREADS_Y_ROWWISE;

        block_amax = 0.0f;
        float in_compute_rowwise[SCALE_DIM];
        Vec<IType, PACK_SIZE> in_cached[WAVES];

        // used as an IType container for BF16/FP16 --> NVFP4 CAST ONLY
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

        // 1. Read/Compute elements. Find NVFP4-block AMAX
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;
            // Load elements
            in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
              ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
            }
          }
          block_amax =
              static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
        } else if constexpr (IS_CACHED_ACT_OP) {
          // ensures that all writes to cache made in the section above are visible to all threads
          __syncthreads();
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

            const bool row_out_of_bounds_rowwise = (row_base_rowwise + it_offset_Y >= rows);
            const bool swizzled_col_out_of_bounds = (block_offset_X + swizzled_thread_idx >= cols);
            const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);

            // Load cached elements
            in_cached[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
            // Since TMA requirement for the data alignment is 16B (i.e. cols % 8 == 0, in case of BF16 elements)
            // only single check (w.r.t. column direction) is sufficient to be sure the entire wave is inside the boundaries
            if (!out_of_bounds) {
              if constexpr (std::is_same_v<IType, float>) {
#pragma unroll
                for (int e = 0; e < PACK_SIZE; ++e) {
                  block_amax = fmaxf(block_amax, fabsf(in_cached[w].data.elt[e]));
                }
              } else {
#pragma unroll
                for (int e = 0; e < PACK_SIZE; e += 2) {
                  const IType2 in_cached_2x = {in_cached[w].data.elt[e],
                                               in_cached[w].data.elt[e + 1]};
                  ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_cached_2x);
                }
              }
            }
          }
          if constexpr (!std::is_same_v<IType, float>) {
            block_amax =
                static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
          }
        } else {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

            Vec<IType, PACK_SIZE> in;
            Vec<IType, PACK_SIZE> act_in;

            in.load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE; ++e) {
              const size_t j = w * PACK_SIZE + e;
              // Compute element
              float elt = static_cast<float>(in.data.elt[e]);
              if constexpr (COMPUTE_ACTIVATIONS) {
                elt = OP(elt, {});
              }
              // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
              if constexpr (!std::is_same_v<IType, float>) {
                elt = static_cast<float>(static_cast<IType>(elt));
              }
              if constexpr (COMPUTE_ACTIVATIONS) {
                const bool row_out_of_bounds_rowwise = (row_base_rowwise + it_offset_Y >= rows);
                const bool swizzled_col_out_of_bounds =
                    (block_offset_X + swizzled_thread_idx >= cols);
                const bool out_of_bounds =
                    (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);
                if (!out_of_bounds) {
                  block_amax = fmaxf(block_amax, fabsf(elt));
                }
              } else {
                // If no activation, elt is 0 so we can safely do this
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
              in_compute_rowwise[j] = elt;
            }
          }
        }

        // 2. Compute E4M3 scaling factor
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

        // Check boundaries
        const size_t scales_offset_Y =
            scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
        const size_t scales_offset_X = scales_offset_X_rowwise;
        const size_t scale_idx_global = scales_offset_Y * scale_stride + scales_offset_X;

        // const bool rowwise_scale_is_within_bounds_Y = scales_offset_Y < rows;
        const bool rowwise_scale_is_within_bounds_Y =
            (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;
        if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
          scales_ptr[scale_idx_global] = S_dec_b_fp8;
        }

        // Compute "correct" per-block encoding scaling factor
        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);  // S_enc_b_fp8
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

// 3. Scale elements
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          Vec<fp4e2m1x4, PACK_SIZE / 4> out;
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 4; ++e) {
            const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
            IType2 in01;
            IType2 in23;
            if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
              const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_IType[w].data.elt[2 * e]);
              out.data.elt[e] = mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  elts, block_scale_inverse_2x, rbits);
            } else if constexpr (IS_CACHED_ACT_OP) {
              const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_cached[w].data.elt[4 * e]);
              out.data.elt[e] = mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  elts, block_scale_inverse_2x, rbits);
            } else {
              const int j = w * PACK_SIZE + 4 * e;
              const float2 in01 = make_float2(in_compute_rowwise[j], in_compute_rowwise[j + 1]);
              const float2 in23 = make_float2(in_compute_rowwise[j + 2], in_compute_rowwise[j + 3]);
              out.data.elt[e] = mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  in01, in23, block_scale_inverse_2x, rbits);
            }
          }
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_out + swizzled_idx / 2;
          out.store_to(&out_data_sh[shmem_offset_rowwise]);
        }
      }
    }

    __builtin_assume(thread_amax >= 0);
    thread_amax = fmaxf(thread_amax, block_amax);

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;

      const size_t global_offset_Y_t = block_offset_Y_t;
      const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), global_offset_X, global_offset_Y,
          reinterpret_cast<uint64_t *>(&out_data_sh[buff_offset_out]));

      if constexpr (RETURN_TRANSPOSE) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_t), global_offset_X_t,
            global_offset_Y_t, reinterpret_cast<uint64_t *>(&out_t_data_sh[buff_offset_out_t]));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
    }
  }  // end of stages

  // Vectorized store scaling factors through SHMEM
  if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
    using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
    const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;
    ScalesVec &scales_vec = *reinterpret_cast<ScalesVec *>(&out_colwise_scales_sh[scale_idx_sh]);
    const size_t scale_idx_global = scales_offset_Y_t * scale_stride_t + scales_offset_X_t;
    const size_t count =  // number of scales in Y dimension of this chunk
        (chunk_rows >= CHUNK_DIM_Y) ? SCALES_PER_CHUNK_Y : (chunk_rows / SCALE_DIM);
    nvfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];
    constexpr size_t vec_bytes = SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t);
    if (count == SCALES_PER_CHUNK_Y && (reinterpret_cast<uintptr_t>(dst) % vec_bytes == 0)) {
      // Fast path: vectorized store when destination is properly aligned
      scales_vec.store_to(dst);
    } else {
      // Safe path: element-wise store for tails or unaligned destinations
      scales_vec.store_to_elts(dst, 0, count);
    }
  }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, bool USE_STOCHASTIC_ROUNDING, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS_NUM)
    nvfp4_transpose_kernel_2D(const __grid_constant__ CUtensorMap tensor_map_input,
                              const __grid_constant__ CUtensorMap tensor_map_output,
                              const __grid_constant__ CUtensorMap tensor_map_output_t,
                              nvfp4_scale_t *const scales_ptr, nvfp4_scale_t *const scales_t_ptr,
                              const float *noop, const float *const amax_rowwise_ptr,
                              const float *const amax_colwise_ptr, const size_t rows,
                              const size_t cols, const size_t scale_stride,
                              const size_t scale_stride_t, const size_t rng_seed,
                              const size_t rng_sequence) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool NO_ACTIVATIONS_NOT_FP32_INPUT =
      (!COMPUTE_ACTIVATIONS) && (!std::is_same_v<IType, float>);

  using IType2 = typename ptx::FPx2<IType>;

  if constexpr (!COMPUTE_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }
  const size_t rng_offest = threadIdx.x;
  RNG rng(rng_seed, rng_sequence, rng_offest);  // seed, sequence, offset
  curanddx::uniform_bits dist;
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? dist.generate4(rng) : uint4{0, 0, 0, 0};
  int rnd_idx =
      0;  // Index of the random number. It increments each time when used and resets to 0 if reaches 4x

  // NEW: 2D Block-based scaling constants
  constexpr size_t BLOCK_DIM = 16;
  constexpr size_t BLOCKS_PER_TILE_Y = TILE_DIM_Y / BLOCK_DIM;  // 32/16 = 2
  constexpr size_t BLOCKS_PER_TILE_X = TILE_DIM_X / BLOCK_DIM;  // 128/16 = 8
  constexpr size_t ITERATIONS_BLOCK = 2;  // iterations to calculate 2d block amaxes of 1 tile
  constexpr size_t BLOCKS_PER_WARP = BLOCKS_PER_TILE_X / (THREADS_NUM / 32);  // 8 / (128/32) = 2

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS;

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;

  const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y;

  const size_t chunk_rows = rows - block_offset_Y;

  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * SCALES_PER_CHUNK_X;
  const size_t scales_block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t scales_block_offset_X_t = blockIdx.y * SCALES_PER_CHUNK_Y;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const size_t tid_X_colwise = threadIdx.x;
  const size_t tid_Y_t = tid_X_colwise;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_t = scales_block_offset_Y_t + tid_Y_t;
  const size_t scales_offset_X_t = scales_block_offset_X_t;

  const size_t SFs_per_row = cols / SCALE_DIM;

  const bool rowwise_scale_is_within_bounds_X = scales_offset_X_rowwise < SFs_per_row;
  const bool colwise_scale_is_within_bounds_Y = scales_offset_Y_t < cols;

  // Helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_mem_rowwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_colwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_rowwise_scales = 0;

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *out_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *out_t_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);

  nvfp4_scale_t *out_rowwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
  nvfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);
  IType *cached_act_sh = in_sh;  // in_sh is used as a cache buffer

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  // Compute a global encoding/decoding scaling factors for all S_dec_b
  const float S_enc_rowwise = (amax_rowwise_ptr == nullptr)
                                  ? 1.0f
                                  : compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);
  // NOTE: This is to match with how emulation code was written.
  const float S_dec_rowwise = 1.0 / S_enc_rowwise;

  const float S_enc_colwise = (amax_colwise_ptr == nullptr)
                                  ? S_enc_rowwise
                                  : compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);
  const float S_dec_colwise = 1.0 / S_enc_colwise;

  const size_t warp_id = threadIdx.x / 32;
  const size_t lane_id = threadIdx.x % 32;
  float thread_amax = 0.0f;
  const size_t block_in_warp = lane_id / BLOCKS_PER_WARP;

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  __shared__ __align__(16) float block_amax_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X + 1];

  // Helper function for warp reduction
  auto warp_reduce_amax = [](float thread_amax, int block_in_warp) -> float {
#pragma unroll
    for (int delta = 8; delta >= 1; delta /= 2) {
      float other_amax = __shfl_xor_sync(0xffffffff, thread_amax, delta);
      thread_amax = fmaxf(thread_amax, other_amax);
    }
    return thread_amax;
  };

  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, shmem_buff_size,
                    &mbar[0], is_master_thread);

#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    const size_t buff_offset_in = buff * BUFF_IN_SIZE;
    const size_t buff_offset_out = buff * BUFF_OUT_SIZE;
    const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

    if (next_stage < STAGES) {
      // Wait for TMA transfer to have finished reading shared memory.
      // I.e. the buffer is ready to be written to
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t next_buff_offset = next_buff * BUFF_IN_SIZE;

      copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                        global_offset_Y, shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;

#pragma unroll
    for (size_t block_iter = 0; block_iter < ITERATIONS_BLOCK; ++block_iter) {
      IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
      const size_t block_in_tile_y = block_iter;
      const size_t block_in_tile_x = threadIdx.x / BLOCK_DIM;

      if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
        for (int elem = 0; elem < BLOCK_DIM; elem += 2) {
          const size_t elem_0_row = block_iter * BLOCK_DIM + elem;
          const size_t elem_1_row = elem_0_row + 1;
          const size_t elem_0_col = warp_id * BLOCKS_PER_WARP * BLOCK_DIM + lane_id;
          const size_t elem_1_col = elem_0_col;

          const size_t shmem_offset_0 = buff_offset_in + elem_0_row * BUFF_IN_DIM_X + elem_0_col;
          const size_t shmem_offset_1 = buff_offset_in + elem_1_row * BUFF_IN_DIM_X + elem_1_col;

          IType2 val_2x;
          val_2x.x = in_sh[shmem_offset_0];
          val_2x.y = in_sh[shmem_offset_1];
          ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, val_2x);
        }

        thread_amax =
            static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
      } else {
        for (int elem = 0; elem < BLOCK_DIM; ++elem) {
          const size_t elem_row = block_iter * BLOCK_DIM + elem;
          const size_t elem_col = warp_id * BLOCKS_PER_WARP * BLOCK_DIM + lane_id;

          // Bounds checking
          const bool row_out_of_bounds = (block_offset_Y + stage_offset_Y + elem_row >= rows);
          const bool col_out_of_bounds = (block_offset_X + elem_col >= cols);
          if (!row_out_of_bounds && !col_out_of_bounds) {
            const size_t shmem_offset = buff_offset_in + elem_row * BUFF_IN_DIM_X + elem_col;
            float elt = static_cast<float>(in_sh[shmem_offset]);

            if constexpr (COMPUTE_ACTIVATIONS) {
              elt = OP(elt, {});
            }
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            // Cache computed activations
            if constexpr (IS_CACHED_ACT_OP) {
              cached_act_sh[shmem_offset] = static_cast<IType>(elt);
            }

            thread_amax = fmaxf(thread_amax, fabsf(elt));
          }
        }
      }
      // Warp reduction to get block amax
      block_amax = warp_reduce_amax(thread_amax, block_in_warp);

      if (lane_id == 0 || lane_id == 16) {
        block_amax_matrix[block_in_tile_y][block_in_tile_x] = block_amax;
      }
    }

    // sync thread to ensure block_amax_matrix is done storing
    __syncthreads();

    // COLWISE scaling
    if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
        const size_t block_in_tile_y = it;
        const size_t block_in_tile_x = threadIdx.x / BLOCK_DIM;

        const size_t in_thread_offset_Y = 0 + it * SCALE_DIM;
        const size_t in_thread_offset_X = thread_offset_X_colwise;

        const size_t out_t_thread_offset_Y = thread_offset_X_colwise;
        const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;

        const size_t shmem_offset_base_colwise_in =
            buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;
        const size_t shmem_offset_base_colwise_out_t =
            buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

        block_amax = block_amax_matrix[block_in_tile_y][block_in_tile_x];
        float in_compute_colwise[SCALE_DIM];
        IType in_colwise_IType[SCALE_DIM];
        // 3. Scale elements

        // Load data in
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            in_colwise_IType[i] = in_sh[shmem_offset_colwise];
          }
        } else {
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            float elt = static_cast<float>(in_sh[shmem_offset_colwise]);
            if constexpr (COMPUTE_ACTIVATIONS) {
              elt = OP(elt, {});
            }
            // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            // Cache computed activations to avoid computing them again in the 2nd pass along another dimension
            if constexpr (IS_CACHED_ACT_OP) {
              cached_act_sh[shmem_offset_colwise] = static_cast<IType>(elt);
            }

            in_compute_colwise[i] = elt;
          }
        }

        // 2. Compute E4M3 scaling factor
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_colwise);

        // // Store scaling factors through SHMEM
        const size_t scale_idx_sh =
            tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
        out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

        // Compute "correct" per-block encoding scaling factor
        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);  // S_enc_b_fp8
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

        fp4e2m1x4 regs[SCALE_DIM / 4];
#pragma unroll
        for (int e = 0; e < SCALE_DIM / 4; ++e) {
          const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
          if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
            const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_colwise_IType[4 * e]);
            regs[e] = mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(elts, block_scale_inverse_2x,
                                                                      rbits);
          } else {
            const float2 in01 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e]);
            const float2 in23 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e + 2]);
            regs[e] = mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                in01, in23, block_scale_inverse_2x, rbits);
          }
        }

        const int group = thread_lane / 16;
        uint32_t val[2];
        uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

        // Helps reducing bank conflicts
        switch (group) {
          case 0:
            val[0] = regs_4x[0];
            val[1] = regs_4x[1];
            break;
          case 1:
            val[0] = regs_4x[1];
            val[1] = regs_4x[0];
            break;
        }
        uint32_t *out_t_data_sh_as_uint32_t =
            reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);
        out_t_data_sh_as_uint32_t[group] = val[0];            // idx1 = (group + 0) % 2;
        out_t_data_sh_as_uint32_t[(group + 1) & 1] = val[1];  // idx2 = (group + 1) % 2;
      }
    }

    // ROWWISE scaling
    {
      const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
        const size_t block_in_tile_y = it;
        const size_t block_in_tile_x = tid_X_rowwise;
        const size_t it_thread_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const size_t shmem_offset_base_rowwise_in =
            buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
        const size_t shmem_offset_base_rowwise_out =
            buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

        block_amax = block_amax_matrix[block_in_tile_y][block_in_tile_x];
        float in_compute_rowwise[SCALE_DIM];
        Vec<IType, PACK_SIZE> in_cached[WAVES];

        // used as an IType container for BF16/FP16 --> NVFP4 CAST ONLY
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

        // 1. Read/Compute elements. Find NVFP4-block AMAX
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;
            // Load elements
            in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
          }
        } else if constexpr (IS_CACHED_ACT_OP) {
          // ensures that all writes to cache made in the section above are visible to all threads
          __syncthreads();
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

            // Load cached elements
            in_cached[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
          }
        } else {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

            Vec<IType, PACK_SIZE> in;
            Vec<IType, PACK_SIZE> act_in;

            in.load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE; ++e) {
              const size_t j = w * PACK_SIZE + e;
              // Compute element
              float elt = static_cast<float>(in.data.elt[e]);
              if constexpr (COMPUTE_ACTIVATIONS) {
                elt = OP(elt, {});
              }
              // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
              if constexpr (!std::is_same_v<IType, float>) {
                elt = static_cast<float>(static_cast<IType>(elt));
              }
              in_compute_rowwise[j] = elt;
            }
          }
        }

        // 2. Compute E4M3 scaling factor
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

        // Check boundaries
        const size_t scales_offset_Y =
            scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
        const size_t scales_offset_X = scales_offset_X_rowwise;
        const size_t scale_idx_global = scales_offset_Y * scale_stride + scales_offset_X;

        // const bool rowwise_scale_is_within_bounds_Y = scales_offset_Y < rows;
        const bool rowwise_scale_is_within_bounds_Y =
            (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;
        if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
          scales_ptr[scale_idx_global] = S_dec_b_fp8;
        }

        // Compute "correct" per-block encoding scaling factor
        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);  // S_enc_b_fp8
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

        // 3. Scale elements
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          Vec<fp4e2m1x4, PACK_SIZE / 4> out;
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 4; ++e) {
            const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
            IType2 in01;
            IType2 in23;
            if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
              const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_IType[w].data.elt[2 * e]);
              out.data.elt[e] = mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  elts, block_scale_inverse_2x, rbits);
            } else if constexpr (IS_CACHED_ACT_OP) {
              const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_cached[w].data.elt[4 * e]);
              out.data.elt[e] = mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  elts, block_scale_inverse_2x, rbits);
            } else {
              const int j = w * PACK_SIZE + 4 * e;
              const float2 in01 = make_float2(in_compute_rowwise[j], in_compute_rowwise[j + 1]);
              const float2 in23 = make_float2(in_compute_rowwise[j + 2], in_compute_rowwise[j + 3]);
              out.data.elt[e] = mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  in01, in23, block_scale_inverse_2x, rbits);
            }
          }

          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_out + swizzled_idx / 2;
          out.store_to(&out_data_sh[shmem_offset_rowwise]);
        }
      }
    }

    __builtin_assume(thread_amax >= 0);
    thread_amax = fmaxf(thread_amax, block_amax);

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;

      const size_t global_offset_Y_t = block_offset_Y_t;
      const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), global_offset_X, global_offset_Y,
          reinterpret_cast<uint64_t *>(&out_data_sh[buff_offset_out]));

      if constexpr (RETURN_TRANSPOSE) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_t), global_offset_X_t,
            global_offset_Y_t, reinterpret_cast<uint64_t *>(&out_t_data_sh[buff_offset_out_t]));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
    }
  }  // end of stages

  // Vectorized store scaling factors through SHMEM
  if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
    using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
    const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;
    ScalesVec &scales_vec = *reinterpret_cast<ScalesVec *>(&out_colwise_scales_sh[scale_idx_sh]);
    const size_t scale_idx_global = scales_offset_Y_t * scale_stride_t + scales_offset_X_t;
    const size_t count =  // number of scales in Y dimension of this chunk
        (chunk_rows >= CHUNK_DIM_Y) ? SCALES_PER_CHUNK_Y : (chunk_rows / SCALE_DIM);
    nvfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];
    constexpr size_t vec_bytes = SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t);
    if (count == SCALES_PER_CHUNK_Y && (reinterpret_cast<uintptr_t>(dst) % vec_bytes == 0)) {
      // Fast path: vectorized store when destination is properly aligned
      scales_vec.store_to(dst);
    } else {
      // Safe path: element-wise store for tails or unaligned destinations
      scales_vec.store_to_elts(dst, 0, count);
    }
  }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace nvfp4_transpose
#endif  // CUDA_VERSION > 12080

// Compile-time flag to choose kernel variant
#ifndef USE_2D_NVFP4_KERNEL
#define USE_2D_NVFP4_KERNEL 0
#endif

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          bool use_2d_quantization>
void nvfp4_quantize_transpose(const Tensor &input, const Tensor *noop, Tensor *output,
                              const QuantizationConfig *quant_config, cudaStream_t stream) {
#if CUDA_VERSION > 12080
  bool use_stochastic_rounding = quant_config ? quant_config->stochastic_rounding : false;

  // If transposed output is allocated, return the transposed data. Otherwise, it's not necesary to
  // return the transposed data.
  // TODO(Frank): Is there a better way to do this?
  bool return_transpose = output->has_columnwise_data();

  using namespace nvfp4_transpose;
  using namespace ptx;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->has_data(), "NVFP4 output tensor must be allocated.");
  NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
  if (return_transpose) {
    NVTE_CHECK(output->has_columnwise_data(), "NVFP4 transposed output tensor must be allocated.");
    NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
               "Transposed output must have FP4 type.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Transposed scaling tensor must be allocated");
  }

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  NVTE_CHECK(rows % 32 == 0,
             "Number of tensor rows must be a multiple of 32");  // 16B alignment for TMA
  NVTE_CHECK(cols % 32 == 0,
             "Number of tensor cols must be a multiple of 32");  // 16B alignment for TMA

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_NUM;

  const size_t scale_stride = output->scale_inv.shape[1];
  const size_t scale_stride_transpose = output->columnwise_scale_inv.shape[1];

  nvfp4_scale_t *const scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  nvfp4_scale_t *const scales_transpose_ptr =
      reinterpret_cast<nvfp4_scale_t *>(output->columnwise_scale_inv.dptr);

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const float *const amax_rowwise_ptr = reinterpret_cast<const float *>(output->amax.dptr);
  const float *const amax_colwise_ptr =
      reinterpret_cast<const float *>(output->columnwise_amax.dptr);

  const size_t rng_seed = (quant_config != nullptr) ? quant_config->rng_seed : 0;
  const size_t rng_sequence = (quant_config != nullptr) ? quant_config->rng_sequence : 0;

  using IType = bf16;

  alignas(64) CUtensorMap tensor_map_input{};
  alignas(64) CUtensorMap tensor_map_output{};
  alignas(64) CUtensorMap tensor_map_output_transpose{};

  create_2D_tensor_map(tensor_map_input, input.data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X, cols, 0,
                       sizeof(IType) * 8);

  create_2D_tensor_map(tensor_map_output, output->data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X, cols, 0,
                       4);
  if (return_transpose) {
    create_2D_tensor_map(tensor_map_output_transpose, output->columnwise_data, cols, rows,
                         BUFF_DIM_X, BUFF_DIM_Y, rows, 0, 4);
  }
  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_scales = (CHUNK_DIM_Y * CHUNK_DIM_X) / 16 * sizeof(nvfp4_scale_t);

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_data_mem = buff_size_aligned_out;
  constexpr size_t out_data_transpose_mem = buff_size_aligned_out;
  constexpr size_t out_scales_transpose_mem = buff_size_scales;

  constexpr size_t out_mem = out_data_mem + out_data_transpose_mem;

  constexpr size_t dshmem_size = in_mem + out_mem + out_scales_transpose_mem + TMA_SHMEM_ALIGNMENT;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_stochastic_rounding, USE_STOCHASTIC_ROUNDING,

      TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE, {
        auto kernel = nvfp4_transpose_kernel<COMPUTE_ACTIVATIONS, ParamOP, OP, IType,
                                             USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE>;

        if constexpr (use_2d_quantization) {
          kernel = nvfp4_transpose_kernel_2D<COMPUTE_ACTIVATIONS, ParamOP, OP, IType,
                                             USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE>;
        }

        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size);
        kernel<<<grid, block_size, dshmem_size, stream>>>(
            tensor_map_input, tensor_map_output, tensor_map_output_transpose, scales_ptr,
            scales_transpose_ptr, noop_ptr, amax_rowwise_ptr, amax_colwise_ptr, rows, cols,
            scale_stride, scale_stride_transpose, rng_seed, rng_sequence);
      }););
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // CUDA_VERSION > 12080
}
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_NVFP4_TRANSPOSE_CUH_
