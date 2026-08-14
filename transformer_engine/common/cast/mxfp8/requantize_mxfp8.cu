/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file requantize_mxfp8.cu
 *  \brief Fused grouped rowwise-MXFP8 to columnwise-MXFP8 conversion.
 */

#include <transformer_engine/cast.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/grouped_tma.cuh"
#include "swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace group_requantize_kernel {

using namespace dispatch::common;

struct DefaultRequantizeConfig {
  static constexpr size_t TILE_DIM_Y = 128;
  static constexpr size_t TILE_DIM_X = 128;
  static constexpr size_t CHUNK_DIM_Y = 128;
  static constexpr size_t CHUNK_DIM_X = 128;
  static constexpr size_t THREADS_PER_CHUNK = 128;
  static constexpr size_t PREFETCH_STAGES = 1;
  static constexpr size_t STATIC_PERSISTENT_BLOCKS_PER_SM = 24;
};

template <ShapeRepresentation SHAPE_REP>
struct RequantizeConfig;

template <>
struct RequantizeConfig<ShapeRepresentation::SAME_BOTH_DIMS>
    : DefaultRequantizeConfig {};

template <>
struct RequantizeConfig<ShapeRepresentation::VARYING_FIRST_DIM>
    : DefaultRequantizeConfig {};

template <>
struct RequantizeConfig<ShapeRepresentation::VARYING_LAST_DIM>
    : DefaultRequantizeConfig {};

template <>
struct RequantizeConfig<ShapeRepresentation::VARYING_BOTH_DIMS>
    : DefaultRequantizeConfig {
  static constexpr size_t CHUNK_DIM_X = 256;
};

template <ShapeRepresentation SHAPE_REP, typename Config>
struct RequantizeTraitsImpl {
  static constexpr ShapeRepresentation SHAPE_REPRESENTATION = SHAPE_REP;
  static constexpr size_t TILE_DIM_Y = Config::TILE_DIM_Y;
  static constexpr size_t TILE_DIM_X = Config::TILE_DIM_X;
  static constexpr size_t CHUNK_DIM_Y = Config::CHUNK_DIM_Y;
  static constexpr size_t CHUNK_DIM_X = Config::CHUNK_DIM_X;
  static constexpr size_t THREADS_PER_CHUNK = Config::THREADS_PER_CHUNK;
  static constexpr size_t PREFETCH_STAGES = Config::PREFETCH_STAGES;
  static constexpr size_t STATIC_PERSISTENT_BLOCKS_PER_SM =
      Config::STATIC_PERSISTENT_BLOCKS_PER_SM;

  static constexpr size_t BUFFS_NUM = PREFETCH_STAGES + 1;
  static constexpr size_t THREADS_X = TILE_DIM_X / MXFP8_SCALE_DIM;
  static constexpr size_t THREADS_Y = THREADS_PER_CHUNK / THREADS_X;
  static constexpr size_t BUFF_DIM_Y = THREADS_Y;
  static constexpr size_t BUFF_DIM_X = TILE_DIM_X;
  static constexpr size_t BUFF_DIM = BUFF_DIM_Y * BUFF_DIM_X;
  static constexpr size_t STAGES_Y = CHUNK_DIM_Y / BUFF_DIM_Y;
  static constexpr size_t STAGES_X = CHUNK_DIM_X / TILE_DIM_X;

  static_assert(TILE_DIM_Y == CHUNK_DIM_Y);
  static_assert(TILE_DIM_X == THREADS_PER_CHUNK);
  static_assert(BUFF_DIM_Y == MXFP8_SCALE_DIM);
  static_assert(PREFETCH_STAGES > 0);
  static_assert(CHUNK_DIM_Y % BUFF_DIM_Y == 0);
  static_assert(CHUNK_DIM_X % TILE_DIM_X == 0);
  static_assert(STATIC_PERSISTENT_BLOCKS_PER_SM > 0);
};

template <ShapeRepresentation SHAPE_REP>
struct RequantizeTraits
    : RequantizeTraitsImpl<SHAPE_REP, RequantizeConfig<SHAPE_REP>> {};

struct LaunchConfig {
  size_t work_blocks_x = 0;
  size_t same_both_rows = 0;
  dim3 grid;
};

template <typename Traits>
LaunchConfig get_launch_config(const size_t first_logical_dim,
                               const size_t last_logical_dim,
                               const size_t total_elements,
                               const size_t num_tensors) {
  constexpr ShapeRepresentation shape_rep = Traits::SHAPE_REPRESENTATION;
  constexpr size_t chunk_dim_y = Traits::CHUNK_DIM_Y;
  constexpr size_t chunk_dim_x = Traits::CHUNK_DIM_X;

  LaunchConfig config;
  if constexpr (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    NVTE_CHECK(first_logical_dim % num_tensors == 0,
               "SAME_BOTH_DIMS requires an integral row count per tensor.");
    config.same_both_rows = first_logical_dim / num_tensors;
    NVTE_CHECK(config.same_both_rows % chunk_dim_y == 0,
               "Each grouped MXFP8 tensor row count must be divisible by ",
               chunk_dim_y, ".");
    config.work_blocks_x = DIVUP(last_logical_dim, chunk_dim_x);
    const size_t work_blocks_y = DIVUP(config.same_both_rows, chunk_dim_y);
    NVTE_CHECK(config.work_blocks_x > 0 && work_blocks_y > 0,
               "SAME_BOTH_DIMS requires non-empty tensors.");
    NVTE_CHECK(work_blocks_y <= 65535 && num_tensors <= 65535,
               "SAME_BOTH_DIMS launch exceeds CUDA grid limits.");
    config.grid = dim3(config.work_blocks_x, work_blocks_y, num_tensors);
  } else if constexpr (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM) {
    NVTE_CHECK(first_logical_dim % chunk_dim_y == 0,
               "The grouped logical row capacity must be divisible by ", chunk_dim_y,
               ".");
    config.work_blocks_x = DIVUP(last_logical_dim, chunk_dim_x);
    const size_t work_blocks_y = DIVUP(first_logical_dim, chunk_dim_y);
    NVTE_CHECK(config.work_blocks_x > 0 && work_blocks_y > 0,
               "VARYING_FIRST_DIM requires a non-empty logical tensor.");
    NVTE_CHECK(work_blocks_y <= 65535,
               "VARYING_FIRST_DIM launch exceeds the CUDA grid Y limit.");
    config.grid = dim3(config.work_blocks_x, work_blocks_y);
  } else {
    NVTE_CHECK(num_tensors <= MAX_SUPPORTED_TENSOR_DESCRIPTORS,
               "Number of tensors exceeds the grouped TMA descriptor limit (",
               MAX_SUPPORTED_TENSOR_DESCRIPTORS, ").");
    const size_t estimated_work_blocks =
        DIVUP(total_elements, chunk_dim_y * chunk_dim_x);
    const size_t static_grid_size =
        static_cast<size_t>(transformer_engine::cuda::sm_count()) *
        Traits::STATIC_PERSISTENT_BLOCKS_PER_SM;
    NVTE_CHECK(static_grid_size > 0, "Persistent launch grid must be non-zero.");
    const size_t requested_workers = std::max<size_t>(1, static_grid_size / num_tensors);
    const size_t average_work =
        std::max<size_t>(1, DIVUP(estimated_work_blocks, num_tensors));
    config.work_blocks_x = std::min(requested_workers, average_work);
    config.grid = dim3(config.work_blocks_x, num_tensors);
  }
  return config;
}

__device__ __forceinline__ uint16_t e8m0_to_bf16_bits(const e8m0_t biased_exp) {
  if (biased_exp == 255) return 0x7fff;
  if (biased_exp == 0) return 0x0040;
  return static_cast<uint16_t>(biased_exp) << 7;
}

template <typename IType>
__device__ __forceinline__ ptx::bf16x2 dequantize_mxfp8_2x(
    const ptx::FPx2<IType> &values, const e8m0_t scale_code) {
  ptx::bf16x2 result;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#if (defined CUDA_VERSION) && (CUDA_VERSION >= 13020)
  constexpr bool is_blackwell_arch = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell_arch) {
    const uint16_t scale_x2 = static_cast<uint16_t>(scale_code) |
                              (static_cast<uint16_t>(scale_code) << 8);
    if constexpr (std::is_same_v<IType, fp8e4m3>) {
      asm volatile(
          "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, %1, %2;"
          : "=r"(reinterpret_cast<uint32_t &>(result))
          : "h"(reinterpret_cast<const uint16_t &>(values)), "h"(scale_x2));
    } else {
      static_assert(std::is_same_v<IType, fp8e5m2>);
      asm volatile(
          "cvt.rn.scaled::n2::ue8m0.bf16x2.e5m2x2 %0, %1, %2;"
          : "=r"(reinterpret_cast<uint32_t &>(result))
          : "h"(reinterpret_cast<const uint16_t &>(values)), "h"(scale_x2));
    }
    return result;
  }
#endif

  const uint16_t scale_bits = e8m0_to_bf16_bits(scale_code);
  const uint32_t scale_x2 = static_cast<uint32_t>(scale_bits) |
                            (static_cast<uint32_t>(scale_bits) << 16);
  if constexpr (std::is_same_v<IType, fp8e4m3>) {
    asm volatile(
        "{\n\t"
        ".reg.b32 values_f16x2, values_bf16x2; \n\t"
        ".reg.b16 value0_f16, value1_f16, value0_bf16, value1_bf16; \n\t"
        "cvt.rn.f16x2.e4m3x2 values_f16x2, %1; \n\t"
        "mov.b32 {value0_f16, value1_f16}, values_f16x2; \n\t"
        "cvt.rn.bf16.f16 value0_bf16, value0_f16; \n\t"
        "cvt.rn.bf16.f16 value1_bf16, value1_f16; \n\t"
        "mov.b32 values_bf16x2, {value0_bf16, value1_bf16}; \n\t"
        "mul.rn.bf16x2 %0, values_bf16x2, %2; \n"
        "}"
        : "=r"(reinterpret_cast<uint32_t &>(result))
        : "h"(reinterpret_cast<const uint16_t &>(values)), "r"(scale_x2));
  } else {
    static_assert(std::is_same_v<IType, fp8e5m2>);
    asm volatile(
        "{\n\t"
        ".reg.b32 values_f16x2, values_bf16x2; \n\t"
        ".reg.b16 value0_f16, value1_f16, value0_bf16, value1_bf16; \n\t"
        "cvt.rn.f16x2.e5m2x2 values_f16x2, %1; \n\t"
        "mov.b32 {value0_f16, value1_f16}, values_f16x2; \n\t"
        "cvt.rn.bf16.f16 value0_bf16, value0_f16; \n\t"
        "cvt.rn.bf16.f16 value1_bf16, value1_f16; \n\t"
        "mov.b32 values_bf16x2, {value0_bf16, value1_bf16}; \n\t"
        "mul.rn.bf16x2 %0, values_bf16x2, %2; \n"
        "}"
        : "=r"(reinterpret_cast<uint32_t &>(result))
        : "h"(reinterpret_cast<const uint16_t &>(values)), "r"(scale_x2));
  }
#else
  NVTE_DEVICE_ERROR("Packed MXFP8 dequantization requires Blackwell hardware.");
#endif
  return result;
}

template <typename OType>
__device__ __forceinline__ void store_colwise_4x_to_shared(
    OType *const output, const uint32_t stride_bytes, const uint32_t values) {
  static_assert(sizeof(OType) == 1);
  const uint32_t output_ptr = __cvta_generic_to_shared(output);
  asm volatile(
      "{\n\t"
      ".reg.u32 ptr1, ptr2, ptr3; \n\t"
      "mad.lo.u32 ptr1, 1, %1, %0; \n\t"
      "mad.lo.u32 ptr2, 2, %1, %0; \n\t"
      "mad.lo.u32 ptr3, 3, %1, %0; \n\t"
      ".reg.b8 value0, value1, value2, value3; \n\t"
      "mov.b32 {value0, value1, value2, value3}, %2; \n\t"
      "st.shared.b8 [%0], value0; \n\t"
      "st.shared.b8 [ptr1], value1; \n\t"
      "st.shared.b8 [ptr2], value2; \n\t"
      "st.shared.b8 [ptr3], value3; \n"
      "}"
      :
      : "r"(output_ptr), "r"(stride_bytes), "r"(values)
      : "memory");
}

template <typename OType>
__device__ __forceinline__ void store_colwise_2x_to_shared(
    OType *const output, const uint32_t stride_bytes, const uint32_t values) {
  static_assert(sizeof(OType) == 1);
  const uint32_t output_ptr = __cvta_generic_to_shared(output);
  asm volatile(
      "{\n\t"
      ".reg.u32 ptr1; \n\t"
      "mad.lo.u32 ptr1, 1, %1, %0; \n\t"
      ".reg.b8 value0, value1, unused0, unused1; \n\t"
      "mov.b32 {value0, value1, unused0, unused1}, %2; \n\t"
      "st.shared.b8 [%0], value0; \n\t"
      "st.shared.b8 [ptr1], value1; \n"
      "}"
      :
      : "r"(output_ptr), "r"(stride_bytes), "r"(values)
      : "memory");
}

template <typename Traits, typename IType, typename OType, bool USE_FAST_MATH,
          bool OUTPUT_SCALES_SWIZZLED>
__device__ __forceinline__ void process_chunk(
    const CUtensorMap &tensor_map_input, const CUtensorMap &tensor_map_output,
    const e8m0_t *const input_scales, e8m0_t *const output_scales,
    const size_t input_scale_base, const size_t output_scale_base,
    const size_t rows, const size_t cols, const size_t block_offset_y,
    const size_t block_offset_x, const size_t tma_offset_y,
    IType *const input_shared, void *const dequantized_shared,
    OType *const output_shared, uint64_t *const input_barriers,
    int *const input_barrier_parity, const bool leading_thread) {
  using TCompute = std::conditional_t<USE_FAST_MATH, bf16, float>;

  constexpr size_t tile_dim_x = Traits::TILE_DIM_X;
  constexpr size_t chunk_dim_y = Traits::CHUNK_DIM_Y;
  constexpr size_t chunk_dim_x = Traits::CHUNK_DIM_X;
  constexpr size_t buffs_num = Traits::BUFFS_NUM;
  constexpr size_t buff_dim_y = Traits::BUFF_DIM_Y;
  constexpr size_t buff_dim_x = Traits::BUFF_DIM_X;
  constexpr size_t buff_dim = Traits::BUFF_DIM;
  constexpr size_t prefetch_stages = Traits::PREFETCH_STAGES;
  constexpr size_t elements_per_load = 16;
  constexpr size_t loads_per_row = tile_dim_x / elements_per_load;
  constexpr size_t rows_per_dequantize_iteration =
      Traits::THREADS_PER_CHUNK / loads_per_row;
  constexpr size_t dequantize_iterations =
      buff_dim_y / rows_per_dequantize_iteration;
  constexpr size_t padding_per_vector = sizeof(float) / sizeof(TCompute);
  constexpr size_t dequantized_stride =
      buff_dim_x + loads_per_row * padding_per_vector;
  constexpr size_t input_buffer_bytes = buff_dim * sizeof(IType);
  constexpr uint32_t output_stride_bytes = buff_dim_x * sizeof(OType);

  static_assert(rows_per_dequantize_iteration == 16);
  static_assert(dequantize_iterations == 2);
  static_assert(dequantized_stride == (USE_FAST_MATH ? 144 : 136));

  using InputShared = IType[buffs_num][buff_dim_y][buff_dim_x];
  using ComputeShared = TCompute[buff_dim_y][dequantized_stride];
  using OutputShared = OType[buffs_num][buff_dim_y][buff_dim_x];
  const auto &s_input = *reinterpret_cast<const InputShared *>(input_shared);
  auto &s_dequantized = *reinterpret_cast<ComputeShared *>(dequantized_shared);
  auto &s_output = *reinterpret_cast<OutputShared *>(output_shared);

  const size_t input_scale_stride = DIVUP_TO_MULTIPLE(
      DIVUP(cols, MXFP8_SCALE_DIM),
      static_cast<size_t>(scale_tensor_alignment_X_rowwise));
  const size_t output_scale_stride = DIVUP_TO_MULTIPLE(
      cols, static_cast<size_t>(scale_tensor_alignment_X_colwise));
  const size_t output_scale_tiles_x =
      DIVUP(rows, static_cast<size_t>(scale_tensor_alignment_Y_rowwise));

  const size_t chunk_rows = min(chunk_dim_y, rows - block_offset_y);
  const size_t chunk_cols = min(chunk_dim_x, cols - block_offset_x);
  const int stages_y = static_cast<int>(DIVUP(chunk_rows, buff_dim_y));
  const int stages_x = static_cast<int>(DIVUP(chunk_cols, tile_dim_x));
  const int stages = stages_y * stages_x;

#pragma unroll
  for (int stage = 0; stage < static_cast<int>(prefetch_stages); ++stage) {
    const size_t stage_y = stage % stages_y;
    const size_t stage_x = stage / stages_y;
    const size_t global_y = tma_offset_y + stage_y * buff_dim_y;
    const size_t global_x = block_offset_x + stage_x * tile_dim_x;
    const size_t buffer_offset = stage * buff_dim;
    prefetch_input_stage<IType, false>(
        input_shared, nullptr, tensor_map_input, tensor_map_input, global_x, global_y,
        buffer_offset, input_buffer_bytes, &input_barriers[stage], leading_thread);
  }

  int input_buffer = 0;
#pragma unroll
  for (int stage = 0; stage < stages; ++stage) {
    const size_t stage_y = stage % stages_y;
    const size_t stage_x = stage / stages_y;
    const size_t stage_offset_y = stage_y * buff_dim_y;
    const size_t stage_offset_x = stage_x * tile_dim_x;

    if (stage < stages - static_cast<int>(prefetch_stages)) {
      const int next_stage = stage + prefetch_stages;
      const int next_buffer = (input_buffer + prefetch_stages) % buffs_num;
      const size_t next_stage_y = next_stage % stages_y;
      const size_t next_stage_x = next_stage / stages_y;
      const size_t global_y = tma_offset_y + next_stage_y * buff_dim_y;
      const size_t global_x = block_offset_x + next_stage_x * tile_dim_x;
      const size_t buffer_offset = next_buffer * buff_dim;
      prefetch_input_stage<IType, false>(
          input_shared, nullptr, tensor_map_input, tensor_map_input, global_x, global_y,
          buffer_offset, input_buffer_bytes, &input_barriers[next_buffer], leading_thread);
    }

    ptx::mbarrier_wait_parity_acquire_cta_shared_cta(
        &input_barriers[input_buffer], input_barrier_parity[input_buffer]);
    input_barrier_parity[input_buffer] ^= 1;

    // Do not overwrite an output buffer that is still consumed by TMA.
    ptx::cp_async_bulk_wait_group_read<prefetch_stages>();

    const int lane = threadIdx.x % THREADS_PER_WARP;
#pragma unroll
    for (int iteration = 0; iteration < static_cast<int>(dequantize_iterations);
         ++iteration) {
      const int local_chunk = threadIdx.x % loads_per_row;
      const int local_row = threadIdx.x / loads_per_row +
                            iteration * rows_per_dequantize_iteration;
      const int local_col = local_chunk * elements_per_load;
      const size_t tensor_row = block_offset_y + stage_offset_y + local_row;
      const size_t tensor_col = block_offset_x + stage_offset_x + local_col;
      const bool data_is_in_bounds = tensor_row < rows && tensor_col < cols;

      int scale_code = 0;
      if (data_is_in_bounds && (local_chunk % 2) == 0) {
        const size_t scale_col = tensor_col / MXFP8_SCALE_DIM;
        const size_t scale_idx = input_scale_base + tensor_row * input_scale_stride +
                                 scale_col;
        scale_code = static_cast<int>(input_scales[scale_idx]);
      }
      scale_code = __shfl_sync(0xffffffff, scale_code, lane & ~1);

      const int shared_col =
          local_col + (local_col / elements_per_load) * padding_per_vector;
      if (data_is_in_bounds) {
        Vec<IType, elements_per_load> values;
        values.load_from(&s_input[input_buffer][local_row][local_col]);
        if constexpr (USE_FAST_MATH) {
#pragma unroll
          for (int element = 0; element < elements_per_load; element += 2) {
            const ptx::FPx2<IType> pair = {values.data.elt[element],
                                           values.data.elt[element + 1]};
            const ptx::bf16x2 result =
                dequantize_mxfp8_2x(pair, static_cast<e8m0_t>(scale_code));
            *reinterpret_cast<ptx::bf16x2 *>(
                &s_dequantized[local_row][shared_col + element]) = result;
          }
        } else {
          const float scale = ptx::exp2f(static_cast<e8m0_t>(scale_code));
#pragma unroll
          for (int element = 0; element < elements_per_load; ++element) {
            s_dequantized[local_row][shared_col + element] =
                scale * static_cast<float>(values.data.elt[element]);
          }
        }
      } else {
#pragma unroll
        for (int element = 0; element < elements_per_load; ++element) {
          s_dequantized[local_row][shared_col + element] =
              static_cast<TCompute>(0.0f);
        }
      }
    }

    __syncthreads();

    const int dequantized_col =
        threadIdx.x + (threadIdx.x / elements_per_load) * padding_per_vector;
    float thread_amax = 0.0f;
    ptx::bf16x4 bf16_values[MXFP8_SCALE_DIM / 4];
    if constexpr (USE_FAST_MATH) {
      ptx::bf16x2 thread_amax_x2 = {static_cast<bf16>(0.0f),
                                     static_cast<bf16>(0.0f)};
#pragma unroll
      for (int row = 0; row < static_cast<int>(MXFP8_SCALE_DIM); row += 4) {
        const ptx::bf16x4 values = {
            s_dequantized[row][dequantized_col],
            s_dequantized[row + 1][dequantized_col],
            s_dequantized[row + 2][dequantized_col],
            s_dequantized[row + 3][dequantized_col]};
        bf16_values[row / 4] = values;
        const ptx::bf16x2 values01 = {values.x1, values.x2};
        const ptx::bf16x2 values23 = {values.x3, values.x4};
        ptx::abs_max_2x(thread_amax_x2, thread_amax_x2, values01);
        ptx::abs_max_2x(thread_amax_x2, thread_amax_x2, values23);
      }
      thread_amax = static_cast<float>(
          ptx::get_amax(thread_amax_x2.x, thread_amax_x2.y));
    } else {
#pragma unroll
      for (int row = 0; row < static_cast<int>(MXFP8_SCALE_DIM); ++row) {
        thread_amax = fmaxf(
            thread_amax,
            fabsf(static_cast<float>(s_dequantized[row][dequantized_col])));
      }
    }

    const e8m0_t output_scale =
        ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);
    const size_t output_col = block_offset_x + stage_offset_x + threadIdx.x;
    const size_t output_scale_row =
        (block_offset_y + stage_offset_y) / MXFP8_SCALE_DIM;
    size_t output_scale_idx;
    if constexpr (OUTPUT_SCALES_SWIZZLED) {
      output_scale_idx = output_scale_base +
                         swizzle::gemm_swizzled_scale_idx(
                             output_col, output_scale_row, output_scale_tiles_x);
    } else {
      output_scale_idx = output_scale_base +
                         output_scale_row * output_scale_stride + output_col;
    }
    output_scales[output_scale_idx] =
        output_col < cols ? output_scale : static_cast<e8m0_t>(0);

    if constexpr (USE_FAST_MATH) {
      const bf16 multiplier = ptx::exp2f_rcp<bf16>(output_scale);
      const ptx::bf16x2 multiplier_x2 = {multiplier, multiplier};
#pragma unroll
      for (int row = 0; row < static_cast<int>(MXFP8_SCALE_DIM); row += 4) {
        uint32_t result_data = 0;
        auto &result = *reinterpret_cast<ptx::FPx4<OType> *>(&result_data);
        ptx::mul_cvt_4x(result, bf16_values[row / 4], multiplier_x2);
        store_colwise_4x_to_shared(&s_output[input_buffer][row][threadIdx.x],
                                   output_stride_bytes, result_data);
      }
    } else {
      const float multiplier = ptx::exp2f_rcp<float>(output_scale);
      const ptx::floatx2 multiplier_x2 = {multiplier, multiplier};
#pragma unroll
      for (int row = 0; row < static_cast<int>(MXFP8_SCALE_DIM); row += 2) {
        const ptx::floatx2 values = {
            s_dequantized[row][dequantized_col],
            s_dequantized[row + 1][dequantized_col]};
        uint32_t result_data = 0;
        auto &result = *reinterpret_cast<ptx::FPx2<OType> *>(&result_data);
        ptx::mul_cvt_2x(result, values, multiplier_x2);
        store_colwise_2x_to_shared(&s_output[input_buffer][row][threadIdx.x],
                                   output_stride_bytes, result_data);
      }
    }

    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    const size_t global_y = tma_offset_y + stage_offset_y;
    const size_t global_x = block_offset_x + stage_offset_x;
    const size_t buffer_offset = input_buffer * buff_dim;
    store_output_stage<OType, false, true>(
        nullptr, output_shared, tensor_map_output, tensor_map_output, global_x,
        global_y, buffer_offset, leading_thread);

    input_buffer = (input_buffer + 1) % buffs_num;
  }
}

template <typename Traits, typename IType, typename OType, bool USE_FAST_MATH,
          bool OUTPUT_SCALES_SWIZZLED>
__global__ void __launch_bounds__(Traits::THREADS_PER_CHUNK)
    group_requantize_mxfp8_kernel(
        const __grid_constant__ CUtensorMap tensor_map_input_static,
        const __grid_constant__ CUtensorMap tensor_map_output_static,
        const size_t num_tensors, const size_t first_logical_dim,
        const size_t last_logical_dim, const size_t same_both_rows,
        const int64_t *const __restrict__ offsets_ptr,
        const int64_t *const __restrict__ first_dims_ptr,
        const int64_t *const __restrict__ last_dims_ptr,
        const e8m0_t *const __restrict__ input_scales,
        e8m0_t *const __restrict__ output_scales) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr ShapeRepresentation shape_rep = Traits::SHAPE_REPRESENTATION;
  constexpr bool direct_same = shape_rep == ShapeRepresentation::SAME_BOTH_DIMS;
  constexpr bool direct_varying_first =
      shape_rep == ShapeRepresentation::VARYING_FIRST_DIM;
  constexpr bool direct_mapper = direct_same || direct_varying_first;
  constexpr bool single_tma_tensor = direct_mapper;
  constexpr size_t buffs_num = Traits::BUFFS_NUM;
  constexpr size_t buff_dim = Traits::BUFF_DIM;
  constexpr size_t input_bytes =
      DIVUP_TO_MULTIPLE(buffs_num * buff_dim * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  using TCompute = std::conditional_t<USE_FAST_MATH, bf16, float>;
  constexpr size_t padding_per_vector = sizeof(float) / sizeof(TCompute);
  constexpr size_t dequantized_stride =
      Traits::BUFF_DIM_X + (Traits::BUFF_DIM_X / 16) * padding_per_vector;
  constexpr size_t dequantized_bytes = DIVUP_TO_MULTIPLE(
      Traits::BUFF_DIM_Y * dequantized_stride * sizeof(TCompute),
      TMA_SHMEM_ALIGNMENT);
  constexpr size_t output_bytes =
      DIVUP_TO_MULTIPLE(buffs_num * buff_dim * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  extern __shared__ unsigned char dynamic_shared[];
  unsigned char *const shared_base =
      align_smem_ptr_per_TMA_requirements(dynamic_shared);
  IType *const input_shared = reinterpret_cast<IType *>(shared_base);
  void *const dequantized_shared = shared_base + input_bytes;
  OType *const output_shared =
      reinterpret_cast<OType *>(shared_base + input_bytes + dequantized_bytes);

  const bool leading_thread = threadIdx.x == 0;

  size_t tensor_id = 0;
  size_t rows = 0;
  size_t cols = 0;
  size_t tensor_base = 0;
  size_t block_offset_y = 0;
  size_t block_offset_x = 0;
  size_t tma_offset_y = 0;

  __shared__ size_t varying_first_metadata[4];
  if constexpr (direct_same) {
    tensor_id = blockIdx.z;
    rows = same_both_rows;
    cols = last_logical_dim;
    tensor_base = tensor_id * rows * cols;
    block_offset_y = blockIdx.y * Traits::CHUNK_DIM_Y;
    block_offset_x = blockIdx.x * Traits::CHUNK_DIM_X;
    tma_offset_y = tensor_id * rows + block_offset_y;
  } else if constexpr (direct_varying_first) {
    const size_t global_block_y = blockIdx.y * Traits::CHUNK_DIM_Y;
    const size_t global_element_offset = global_block_y * last_logical_dim;
    if (leading_thread) {
      const size_t active_elements = static_cast<size_t>(offsets_ptr[num_tensors]);
      varying_first_metadata[0] = active_elements;
      if (global_element_offset < active_elements) {
        const size_t mapped_tensor =
            find_tensor_from_offsets(offsets_ptr, num_tensors, global_element_offset);
        varying_first_metadata[1] = mapped_tensor;
        varying_first_metadata[2] = get_tensor_rows_num<shape_rep>(
            mapped_tensor, first_logical_dim, first_dims_ptr, num_tensors);
        varying_first_metadata[3] =
            static_cast<size_t>(offsets_ptr[mapped_tensor]);
      }
    }
    __syncthreads();
    if (global_element_offset >= varying_first_metadata[0]) return;

    tensor_id = varying_first_metadata[1];
    rows = varying_first_metadata[2];
    cols = last_logical_dim;
    tensor_base = varying_first_metadata[3];
    const size_t tensor_start_y = tensor_base / cols;
    block_offset_y = global_block_y - tensor_start_y;
    block_offset_x = blockIdx.x * Traits::CHUNK_DIM_X;
    tma_offset_y = global_block_y;
  } else {
    tensor_id = blockIdx.y;
    if (tensor_id >= num_tensors) return;
    rows = get_tensor_rows_num<shape_rep>(
        tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
    cols = get_tensor_cols_num<shape_rep>(
        tensor_id, last_logical_dim, last_dims_ptr);
    tensor_base = static_cast<size_t>(offsets_ptr[tensor_id]);
    if (rows == 0 || cols == 0) return;
  }

  const CUtensorMap &tensor_map_input =
      single_tma_tensor ? tensor_map_input_static : g_tensor_maps.input[tensor_id];
  const CUtensorMap &tensor_map_output =
      single_tma_tensor ? tensor_map_output_static
                        : g_tensor_maps.output_colwise[tensor_id];
  if constexpr (!single_tma_tensor) {
    if (leading_thread) {
      fence_acquire_tensormap(&tensor_map_input);
      fence_acquire_tensormap(&tensor_map_output);
    }
    __syncthreads();
  }

  const size_t input_scale_stride = DIVUP_TO_MULTIPLE(
      DIVUP(cols, MXFP8_SCALE_DIM),
      static_cast<size_t>(scale_tensor_alignment_X_rowwise));
  const size_t output_scale_stride = DIVUP_TO_MULTIPLE(
      cols, static_cast<size_t>(scale_tensor_alignment_X_colwise));
  size_t input_scale_base;
  size_t output_scale_base;
  if constexpr (single_tma_tensor) {
    const size_t tensor_start_row = tensor_base / cols;
    input_scale_base = tensor_start_row * input_scale_stride;
    output_scale_base =
        (tensor_start_row / MXFP8_SCALE_DIM) * output_scale_stride;
  } else {
    input_scale_base = tensor_base / MXFP8_SCALE_DIM;
    output_scale_base = tensor_base / MXFP8_SCALE_DIM;
  }

  __shared__ uint64_t input_barriers[buffs_num];
  initialize_barriers<buffs_num, 1>(input_barriers, leading_thread);
  int input_barrier_parity[buffs_num] = {0};

  if constexpr (direct_mapper) {
    process_chunk<Traits, IType, OType, USE_FAST_MATH,
                  OUTPUT_SCALES_SWIZZLED>(
        tensor_map_input, tensor_map_output, input_scales, output_scales,
        input_scale_base, output_scale_base, rows, cols, block_offset_y,
        block_offset_x, tma_offset_y, input_shared, dequantized_shared,
        output_shared, input_barriers, input_barrier_parity, leading_thread);
  } else {
    const size_t blocks_x = DIVUP(cols, Traits::CHUNK_DIM_X);
    const size_t blocks_y = DIVUP(rows, Traits::CHUNK_DIM_Y);
    const size_t total_blocks = blocks_x * blocks_y;
    for (size_t block_id = blockIdx.x; block_id < total_blocks;
         block_id += gridDim.x) {
      const size_t block_y = block_id / blocks_x;
      const size_t block_x = block_id - block_y * blocks_x;
      process_chunk<Traits, IType, OType, USE_FAST_MATH,
                    OUTPUT_SCALES_SWIZZLED>(
          tensor_map_input, tensor_map_output, input_scales, output_scales,
          input_scale_base, output_scale_base, rows, cols,
          block_y * Traits::CHUNK_DIM_Y, block_x * Traits::CHUNK_DIM_X,
          block_y * Traits::CHUNK_DIM_Y, input_shared, dequantized_shared,
          output_shared, input_barriers, input_barrier_parity, leading_thread);
    }
  }

  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();
  destroy_barriers<buffs_num>(input_barriers, leading_thread);
#else
  NVTE_DEVICE_THREAD0_ERROR(
      "Grouped MXFP8 requantization requires Blackwell (SM100+) hardware.");
#endif
}

template <typename Traits, typename IType, bool USE_FAST_MATH,
          bool OUTPUT_SCALES_SWIZZLED>
void launch_group_requantize(
    const GroupedTensor &input, GroupedTensor *output,
    const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const size_t total_elements,
    const int64_t *const offsets_ptr, const int64_t *const first_dims_ptr,
    const int64_t *const last_dims_ptr, const ShapeRepresentation shape_rep,
    cudaStream_t stream) {
  using OType = fp8e4m3;
  using TCompute = std::conditional_t<USE_FAST_MATH, bf16, float>;
  constexpr size_t buffs_num = Traits::BUFFS_NUM;
  constexpr size_t buff_dim = Traits::BUFF_DIM;
  constexpr size_t padding_per_vector = sizeof(float) / sizeof(TCompute);
  constexpr size_t dequantized_stride =
      Traits::BUFF_DIM_X + (Traits::BUFF_DIM_X / 16) * padding_per_vector;
  constexpr size_t input_bytes = DIVUP_TO_MULTIPLE(
      buffs_num * buff_dim * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t dequantized_bytes = DIVUP_TO_MULTIPLE(
      Traits::BUFF_DIM_Y * dequantized_stride * sizeof(TCompute),
      TMA_SHMEM_ALIGNMENT);
  constexpr size_t output_bytes = DIVUP_TO_MULTIPLE(
      buffs_num * buff_dim * sizeof(OType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t dynamic_shared_bytes =
      input_bytes + dequantized_bytes + output_bytes + TMA_SHMEM_ALIGNMENT;

  const LaunchConfig launch_config = get_launch_config<Traits>(
      first_logical_dim, last_logical_dim, total_elements, num_tensors);

  alignas(64) CUtensorMap tensor_map_input{};
  alignas(64) CUtensorMap tensor_map_output{};
  create_2D_tensor_map(tensor_map_input, input.data, first_logical_dim,
                       last_logical_dim, Traits::BUFF_DIM_Y,
                       Traits::BUFF_DIM_X, last_logical_dim, 0,
                       TypeInfo<IType>::size);
  create_2D_tensor_map(tensor_map_output, output->columnwise_data,
                       first_logical_dim, last_logical_dim,
                       Traits::BUFF_DIM_Y, Traits::BUFF_DIM_X,
                       last_logical_dim, 0, TypeInfo<OType>::size);

  constexpr bool single_tma_tensor =
      Traits::SHAPE_REPRESENTATION == ShapeRepresentation::SAME_BOTH_DIMS ||
      Traits::SHAPE_REPRESENTATION == ShapeRepresentation::VARYING_FIRST_DIM;
  if constexpr (!single_tma_tensor) {
    alignas(64) CUtensorMap empty_tensor_map{};
    update_tma_descriptors<IType, OType><<<num_tensors, 1, 0, stream>>>(
        tensor_map_input, empty_tensor_map, empty_tensor_map, tensor_map_output,
        reinterpret_cast<const IType *>(input.data.dptr), nullptr, nullptr,
        reinterpret_cast<OType *>(output->columnwise_data.dptr), shape_rep,
        num_tensors, first_logical_dim, last_logical_dim, offsets_ptr,
        first_dims_ptr, last_dims_ptr, false, true, false);
  }

  auto kernel = group_requantize_mxfp8_kernel<
      Traits, IType, OType, USE_FAST_MATH, OUTPUT_SCALES_SWIZZLED>;
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_bytes));
  kernel<<<launch_config.grid, Traits::THREADS_PER_CHUNK,
           dynamic_shared_bytes, stream>>>(
      tensor_map_input, tensor_map_output, num_tensors, first_logical_dim,
      last_logical_dim, launch_config.same_both_rows, offsets_ptr,
      first_dims_ptr, last_dims_ptr,
      reinterpret_cast<const e8m0_t *>(input.scale_inv.dptr),
      reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr));
}

}  // namespace group_requantize_kernel

void group_requantize(const GroupedTensor &input, GroupedTensor *output,
                      const QuantizationConfig *quant_config,
                      cudaStream_t stream) {
  using namespace group_requantize_kernel;

  checkCuDriverContext(stream);
  NVTE_CHECK(is_supported_by_CC_100(),
             "Grouped MXFP8 requantization requires Blackwell (SM100+) hardware.");
  CheckInputGroupedTensor(input, "group_requantize_input");
  CheckOutputGroupedTensor(*output, "group_requantize_output");

  NVTE_CHECK(input.scaling_mode == NVTE_MXFP8_1D_SCALING,
             "Input must use MXFP8 1D scaling.");
  NVTE_CHECK(output->scaling_mode == NVTE_MXFP8_1D_SCALING,
             "Output must use MXFP8 1D scaling.");
  NVTE_CHECK(input.has_data() && !input.has_columnwise_data(),
             "Input must contain rowwise MXFP8 data only.");
  NVTE_CHECK(!input.with_gemm_swizzled_scales,
             "Input rowwise MXFP8 scales must be in compact format.");
  NVTE_CHECK(!output->has_data() && output->has_columnwise_data(),
             "Output must contain columnwise MXFP8 data only.");
  NVTE_CHECK(is_fp8_dtype(input.data.dtype),
             "Input rowwise data must have an FP8 type.");
  NVTE_CHECK(output->columnwise_data.dtype == DType::kFloat8E4M3,
             "Output columnwise data must have E4M3 type.");
  NVTE_CHECK(input.scale_inv.dtype == DType::kFloat8E8M0 &&
                 output->columnwise_scale_inv.dtype == DType::kFloat8E8M0,
             "MXFP8 scaling tensors must have E8M0 type.");
  NVTE_CHECK(input.num_tensors == output->num_tensors,
             "Input and output must contain the same number of tensors.");
  NVTE_CHECK(input.num_tensors > 0,
             "Grouped tensor must contain at least one tensor.");
  NVTE_CHECK(input.logical_shape.ndim == 2 && output->logical_shape.ndim == 2 &&
                 input.logical_shape.data[0] == output->logical_shape.data[0] &&
                 input.logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output logical shapes must match.");
  NVTE_CHECK(input.all_same_first_dim() == output->all_same_first_dim() &&
                 input.all_same_last_dim() == output->all_same_last_dim(),
             "Input and output grouped shape representations must match.");
  NVTE_CHECK(input.data.dptr != output->columnwise_data.dptr,
             "In-place MXFP8 data requantization is not supported.");
  NVTE_CHECK(is_aligned_ptr(input.data.dptr, TMA_GMEM_ALIGNMENT) &&
                 is_aligned_ptr(output->columnwise_data.dptr, TMA_GMEM_ALIGNMENT),
             "Input and output data pointers must be 16-byte aligned.");

  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (input.all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (input.all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else if (input.all_same_first_dim()) {
    shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
  } else {
    shape_rep = ShapeRepresentation::VARYING_BOTH_DIMS;
  }

  const size_t first_logical_dim = input.logical_shape.data[0];
  const size_t last_logical_dim = input.logical_shape.data[1];
  const size_t total_elements = first_logical_dim * last_logical_dim;
  NVTE_CHECK(last_logical_dim % MXFP8_SCALE_DIM == 0 ||
                 shape_rep == ShapeRepresentation::VARYING_LAST_DIM ||
                 shape_rep == ShapeRepresentation::VARYING_BOTH_DIMS,
             "Every MXFP8 hidden dimension must be divisible by ",
             MXFP8_SCALE_DIM, ".");

  const int64_t *const offsets_ptr =
      reinterpret_cast<const int64_t *>(input.tensor_offsets.dptr);
  const int64_t *const first_dims_ptr =
      reinterpret_cast<const int64_t *>(input.first_dims.dptr);
  const int64_t *const last_dims_ptr =
      reinterpret_cast<const int64_t *>(input.last_dims.dptr);
  const bool use_fast_math =
      quant_config != nullptr && quant_config->use_fast_math;

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          use_fast_math, USE_FAST_MATH,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              output->with_gemm_swizzled_scales, OUTPUT_SCALES_SWIZZLED,
              TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
                  shape_rep, SHAPE_REP,
                  {
                    using ActiveTraits = RequantizeTraits<SHAPE_REP>;
                    launch_group_requantize<ActiveTraits, IType, USE_FAST_MATH,
                                            OUTPUT_SCALES_SWIZZLED>(
                        input, output, input.num_tensors, first_logical_dim,
                        last_logical_dim, total_elements, offsets_ptr,
                        first_dims_ptr, last_dims_ptr, shape_rep, stream);
                  }););););  // NOLINT(*), readability/fn_size
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

void nvte_group_requantize_mxfp8(
    const NVTEGroupedTensor input, NVTEGroupedTensor output,
    const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_requantize_mxfp8);
  using namespace transformer_engine;
  const GroupedTensor *const input_cu = convertNVTEGroupedTensorCheck(input);
  GroupedTensor *const output_cu = convertNVTEGroupedTensorCheck(output);
  const auto *const quant_config_cu =
      reinterpret_cast<const QuantizationConfig *>(quant_config);
  dispatch::mxfp8::group_requantize(*input_cu, output_cu,
                                    quant_config_cu, stream);
}
