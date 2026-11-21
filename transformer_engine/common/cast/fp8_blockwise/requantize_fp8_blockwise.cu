/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file requantize_fp8_blockwise.cu
 *  \brief Grouped block128 FP8 requantization: rowwise wire tensor -> GEMM-ready.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/logging.h"
#include "../../util/ptx.cuh"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"
#include "../core/grouped_layout.cuh"

namespace transformer_engine {
namespace requantize {
namespace {
// Block scaling FP8 grouped requantization shares the common grouped requant entry point.
namespace ptx = transformer_engine::ptx;

constexpr int kTileDim = 128; // Fixed to 128 beacause we are using 1x128 quantization
constexpr int kThreadsPerBlock = 256; // Thread block size, 8 warps in total
constexpr int kThreadsPerWarp = 32;

struct RequantizeDebugTrace {
  uint64_t tma_issue;
  uint64_t tma_ready;
  uint64_t compute_done;
  uint64_t drain_done;
};

// Decode four adjacent FP8 values with two packed conversion instructions.
// Rowwise scales are different for the four rows, so the final multiply stays
// in FP32; packing still halves the FP8-to-half conversion count compared with
// four scalar casts.
template <typename IType>
__device__ __forceinline__ ptx::bf16x4 dequantize_fp8x4(const IType x0, const IType x1,
                                                        const IType x2, const IType x3,
                                                        const float s0, const float s1,
                                                        const float s2, const float s3) {
  static_assert(sizeof(IType) == 1, "FP8 input is one byte.");
  const uint16_t pair01 = static_cast<uint16_t>(reinterpret_cast<const uint8_t&>(x0)) |
                          (static_cast<uint16_t>(reinterpret_cast<const uint8_t&>(x1)) << 8);
  const uint16_t pair23 = static_cast<uint16_t>(reinterpret_cast<const uint8_t&>(x2)) |
                          (static_cast<uint16_t>(reinterpret_cast<const uint8_t&>(x3)) << 8);
  uint32_t half01;
  uint32_t half23;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  if constexpr (std::is_same_v<IType, fp8e4m3>) {
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half01) : "h"(pair01));
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half23) : "h"(pair23));
  } else {
    static_assert(std::is_same_v<IType, fp8e5m2>);
    asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(half01) : "h"(pair01));
    asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(half23) : "h"(pair23));
  }
#else
  half01 = 0;
  half23 = 0;
#endif
  const auto half_to_float = [](const uint32_t bits) {
    const uint16_t raw = static_cast<uint16_t>(bits);
    float value;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(raw));
    return value;
  };
  ptx::bf16x4 result;
  result.x1 = static_cast<bf16>(half_to_float(half01) * s0);
  result.x2 = static_cast<bf16>(half_to_float(half01 >> 16) * s1);
  result.x3 = static_cast<bf16>(half_to_float(half23) * s2);
  result.x4 = static_cast<bf16>(half_to_float(half23 >> 16) * s3);
  return result;
}

// The column path already owns four reconstructed BF16 values in registers.
// Convert and store them directly; staging a transposed tile in shared memory
// would only add a write/read round trip before the same strided global stores.
template <typename OType, int kRows>
__device__ __forceinline__ void store_columnwise_bf16(
    OType* __restrict__ output, const ptx::bf16x4* __restrict__ values, const float scale,
    const size_t tensor_base, const size_t column, const size_t group_rows,
    const size_t row) {
  static_assert(kRows % 4 == 0);
  Vec<OType, kRows> packed;
#pragma unroll
  for (int i = 0; i < kRows; ++i) {
    const ptx::bf16x4& value = values[i / 4];
    const bf16 element = (i % 4 == 0) ? value.x1
                         : (i % 4 == 1) ? value.x2
                         : (i % 4 == 2) ? value.x3
                                        : value.x4;
    packed.data.elt[i] = static_cast<OType>(static_cast<float>(element) * scale);
  }
  packed.store_to(output + tensor_base + column * group_rows + row);
}

template <typename OType>
__device__ __forceinline__ void drain_columnwise_output(
    OType (&smem_t)[kTileDim][kTileDim + 4], OType* __restrict__ output,
    const size_t global_col_base, const size_t local_row_base, const size_t tensor_base,
    const size_t group_rows, const size_t hidden, const int tid) {
  constexpr int kStoreRows = 16;
  constexpr int kChunksPerColumn = kTileDim / kStoreRows;
  constexpr int kColumnsPerIteration = kThreadsPerBlock / kChunksPerColumn;
  constexpr int kIterations = kTileDim / kColumnsPerIteration;
  const int warp_id = tid / kThreadsPerWarp;
  const int lane = tid % kThreadsPerWarp;
  const int column_in_warp = lane / kChunksPerColumn;
  const int row_chunk = lane % kChunksPerColumn;
  const int row_offset = row_chunk * kStoreRows;

#pragma unroll
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const int column_local = iteration * kColumnsPerIteration + warp_id * 4 + column_in_warp;
    const size_t column = global_col_base + column_local;
    if (column < hidden) {
      Vec<OType, kStoreRows> values;
#pragma unroll
      for (int e = 0; e < kStoreRows; ++e) {
        values.data.elt[e] = smem_t[column_local][row_offset + e];
      }
      values.store_to(output + tensor_base + column * group_rows + local_row_base + row_offset);
    }
  }
}

template <ShapeRepresentation ShapeRep>
struct RequantizeTraits {
  static_assert(ShapeRep == ShapeRepresentation::SAME_BOTH_DIMS ||
                ShapeRep == ShapeRepresentation::VARYING_FIRST_DIM);
  static constexpr ShapeRepresentation SHAPE_REPRESENTATION = ShapeRep;
};

/*! \brief Validate per-tile group offsets once per kernel launch.
 *
 *  Runs in a single CTA (blockIdx == 0,0 / threadIdx == 0); other CTAs do a
 *  cheaper per-tile bounds check below. Traps on violation so release builds
 *  surface the failure rather than silently dropping tiles.
 */
__device__ void validate_group_offsets(const int64_t* tensor_offsets, size_t num_groups,
                                       size_t last_logical_dim, size_t first_logical_dim) {
  const int64_t last = tensor_offsets[num_groups];
  const bool power_of_two_last_dim = (last_logical_dim & (last_logical_dim - 1)) == 0;
  const size_t tile_elements = last_logical_dim * kTileDim;
  bool valid = tensor_offsets[0] == 0 && last >= 0 &&
               static_cast<uint64_t>(last) <= first_logical_dim * last_logical_dim;
  int64_t begin = 0;
  for (size_t group = 0; group < num_groups && valid; ++group) {
    const int64_t end = tensor_offsets[group + 1];
    valid = end >= begin && end <= last &&
            (power_of_two_last_dim ? (static_cast<uint64_t>(end) & (last_logical_dim - 1)) == 0
                                 : static_cast<uint64_t>(end) % last_logical_dim == 0) &&
            (begin == last ||
             (power_of_two_last_dim ? (static_cast<uint64_t>(begin) & (tile_elements - 1)) == 0
                                  : static_cast<uint64_t>(begin) % tile_elements == 0));
    begin = end;
  }
  if (!valid) {
    NVTE_DEVICE_ERROR("Block scaling grouped requant invalid offsets: require zero origin, "
                      "monotonic row boundaries within capacity, and 128-aligned group starts.");
    __trap();
  }
}

/*! \brief Requant one 128x128 tile from rowwise to columnwise block128 FP8.
 *
 *  Grid: (ceil(last_logical_dim/128), ceil(first_logical_dim/128)). Block: 256 threads.
 *  4 threads per column x 32 rows each, two column passes over the tile.
 *
 *  Input supports E4M3/E5M2; columnwise output uses E4M3.
 */
template <typename Traits, typename IType, typename OType, bool kReturnDequantized,
          bool kReturnColumnwise, bool kReturnRowwiseScale, bool kDebugTrace>
__global__ void __launch_bounds__(kThreadsPerBlock) block128_group_requantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const IType* __restrict__ rowwise_data, const float* __restrict__ rowwise_scale,
    float* __restrict__ gemm_rowwise_scale,
    OType* __restrict__ colwise_data, float* __restrict__ colwise_scale,
    bf16* __restrict__ dequantized, const size_t first_logical_dim, const size_t last_logical_dim,
    const size_t row_scale_cols, const bool compact_scale_layout,
    const size_t col_scale_rows,
    const int64_t* tensor_offsets, const int64_t* first_dims_ptr, const size_t num_groups,
    const float epsilon, const bool power_of_two, RequantizeDebugTrace* debug_trace) {
  const bool leading_thread = threadIdx.x == 0;
  constexpr ShapeRepresentation shape_rep = Traits::SHAPE_REPRESENTATION;
  const size_t tile_x = blockIdx.x;  // hidden block index
  const size_t tile_y = blockIdx.y;  // row block index
  const size_t global_col_base = tile_x * kTileDim;
  const size_t trace_index = (static_cast<size_t>(blockIdx.z) * gridDim.y + blockIdx.y) *
                             gridDim.x + blockIdx.x;
  RequantizeDebugTrace* trace = nullptr;
  if constexpr (kDebugTrace) {
    trace = debug_trace + trace_index;
    if (threadIdx.x == 0) trace->tma_issue = trace->tma_ready = trace->compute_done = 0;
  }
  size_t tensor_base = 0;
  size_t group_rows = first_logical_dim;
  size_t local_row_base = 0;
  if constexpr (Traits::SHAPE_REPRESENTATION == ShapeRepresentation::SAME_BOTH_DIMS) {
    group_rows = first_logical_dim / num_groups;
    tensor_base = blockIdx.z * group_rows * last_logical_dim;
    local_row_base = tile_y * kTileDim;
  } else if constexpr (Traits::SHAPE_REPRESENTATION == ShapeRepresentation::VARYING_FIRST_DIM) {
    const size_t global_element_offset = tile_y * kTileDim * last_logical_dim;
    // [active elements, expert id, expert rows, expert element offset].
    __shared__ size_t varying_first_metadata[4];
    if (leading_thread && blockIdx.x == 0 && blockIdx.y == 0) {
      validate_group_offsets(tensor_offsets, num_groups, last_logical_dim, first_logical_dim);
    }
    if (leading_thread) {
      const size_t active_elements = static_cast<size_t>(tensor_offsets[num_groups]);
      varying_first_metadata[0] = active_elements;
      if (global_element_offset < active_elements) {
        const size_t mapped_tensor = dispatch::common::find_tensor_from_offsets(
            tensor_offsets, num_groups, global_element_offset);
        varying_first_metadata[1] = mapped_tensor;
        varying_first_metadata[2] = dispatch::common::get_tensor_rows_num<shape_rep>(
            mapped_tensor, first_logical_dim, first_dims_ptr, num_groups);
        // The helper's assert is disabled in release builds.
        if (varying_first_metadata[2] % kTileDim != 0) __trap();
        varying_first_metadata[3] = static_cast<size_t>(tensor_offsets[mapped_tensor]);
      }
    }
    __syncthreads();
    if (global_element_offset >= varying_first_metadata[0]) return;
    group_rows = varying_first_metadata[2];
    tensor_base = varying_first_metadata[3];
    local_row_base = tile_y * kTileDim - tensor_base / last_logical_dim;
  }
  const size_t group_start = tensor_base / last_logical_dim;
  const size_t local_tile_y = local_row_base / kTileDim;
  const size_t global_row_base = group_start + local_row_base;
  const size_t group_end = group_start + group_rows;
  const size_t live_end = first_logical_dim < group_end ? first_logical_dim : group_end;

  // Load each row's scale once, shared by all 128 columns and the layout conversion.
  __shared__ float row_scales[kTileDim];
  if (threadIdx.x < kTileDim) {
    const size_t row = global_row_base + threadIdx.x;
    const size_t index = compact_scale_layout
                             ? row * row_scale_cols + tile_x
                             : group_start * (last_logical_dim / kTileDim) +
                                   tile_x * group_rows + (row - group_start);
    const float value = rowwise_scale[index];
    if constexpr (kReturnRowwiseScale) {
      const size_t destination = group_start * (last_logical_dim / kTileDim) +
                                 tile_x * group_rows + (row - group_start);
      gemm_rowwise_scale[destination] = value;
    }
    row_scales[threadIdx.x] = value;
  }

  // Scale-only specialization does not load or reconstruct activation data.
  if constexpr (!kReturnColumnwise && !kReturnDequantized) return;

  // The rowwise input is loaded into a linear 128x128 shared tile. The
  // columnwise path does not use the swizzled row/column access pattern for
  // which TMA swizzling is useful; the four threads for one column already
  // access fixed-stride rows. Keep the input path identical to the direct
  // grouped quantizer and reserve swizzled shared memory only for output
  // layouts that actually need it.
  extern __shared__ char smem_raw[];
  char* smem_base = align_up(smem_raw, TMA_SHMEM_ALIGNMENT);
  IType* input_smem = reinterpret_cast<IType*>(smem_base);
  __shared__ uint64_t tma_mbar;
  const int tid = threadIdx.x;

  if (leading_thread) {
    ptx::mbarrier_init(&tma_mbar, 1);
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if (leading_thread) {
    constexpr uint32_t tx_bytes = kTileDim * kTileDim * sizeof(IType);
    if constexpr (kDebugTrace) trace->tma_issue = clock64();
    ptx::mbarrier_arrive_expect_tx(&tma_mbar, tx_bytes);
    ptx::cp_async_bulk_tensor_2d_global_to_shared_cta(
        reinterpret_cast<uint64_t*>(smem_base),
        reinterpret_cast<const uint64_t*>(&tensor_map_input),
        static_cast<uint32_t>(global_col_base), static_cast<uint32_t>(global_row_base), &tma_mbar);
  }
  ptx::mbarrier_wait_parity(&tma_mbar, 0);
  __syncthreads();
  if constexpr (kDebugTrace) {
    if (leading_thread) trace->tma_ready = clock64();
  }
  if (leading_thread) ptx::mbarrier_invalid(&tma_mbar);

  if constexpr (kReturnColumnwise) {
    // Keep the reconstructed values in registers across the amax reduction.
    // This avoids rereading and re-decoding the TMA tile for the
    // output pass; the four-thread mapping bounds the register footprint to
    // one 32-row column slice per thread.
    // Requantization already receives the rowwise FP8 tile, so it does not
    // need the separate rowwise quantization pass used by the direct BF16
    // quantizer.  Two producers per column cover the whole 128-row tile in
    // one pass and avoid a second subwarp reduction/scale computation.
    constexpr int kThreadsPerCol = 2;
    constexpr int kRowsPerThread = kTileDim / kThreadsPerCol;
    constexpr int kColsPerPass = kThreadsPerBlock / kThreadsPerCol;
    constexpr int kColumnPasses = kTileDim / kColsPerPass;
    const int col_in_pass = tid / kThreadsPerCol;
    const int sub = tid % kThreadsPerCol;
    const int row_start = sub * kRowsPerThread;
    const auto* input_tile = reinterpret_cast<const IType*>(input_smem);
    if constexpr (Traits::SHAPE_REPRESENTATION == ShapeRepresentation::VARYING_FIRST_DIM) {
      // Varying expert row counts make the column-major destination stride
      // jagged. Stage this path so the final drain remains coalesced; SAME
      // shapes use the lower-overhead direct register stores below.
      constexpr int kSMemTRowStride = kTileDim + 4;
      __shared__ OType output_smem[kTileDim][kSMemTRowStride];
#pragma unroll
      for (int pass = 0; pass < kColumnPasses; ++pass) {
        const int col_local = pass * kColsPerPass + col_in_pass;
        const size_t col = global_col_base + col_local;
        float amax = 0.f;
        ptx::bf16x4 reg_data[kRowsPerThread / 4];
#pragma unroll
        for (int r = 0; r < kRowsPerThread; r += 4) {
          reg_data[r / 4] = dequantize_fp8x4(
              input_tile[(row_start + r + 0) * kTileDim + col_local],
              input_tile[(row_start + r + 1) * kTileDim + col_local],
              input_tile[(row_start + r + 2) * kTileDim + col_local],
              input_tile[(row_start + r + 3) * kTileDim + col_local], row_scales[row_start + r + 0],
              row_scales[row_start + r + 1], row_scales[row_start + r + 2],
              row_scales[row_start + r + 3]);
          const ptx::bf16x4& values = reg_data[r / 4];
          amax = fmaxf(amax, fabsf(static_cast<float>(values.x1)));
          amax = fmaxf(amax, fabsf(static_cast<float>(values.x2)));
          amax = fmaxf(amax, fabsf(static_cast<float>(values.x3)));
          amax = fmaxf(amax, fabsf(static_cast<float>(values.x4)));
        }
        amax = subwarp_reduce_max_broadcast<kThreadsPerCol>(amax);
        const float scale = compute_scale_from_types<float, OType>(amax, epsilon, power_of_two);
        if (sub == 0) {
          const size_t scale_stride = DIVUP_TO_MULTIPLE(last_logical_dim, 4);
          const size_t group_scale_base = (group_start / kTileDim) * scale_stride;
          colwise_scale[group_scale_base + local_tile_y * scale_stride + col] = __frcp_rn(scale);
        }
#pragma unroll
        for (int r = 0; r < kRowsPerThread; r += 4) {
          const ptx::bf16x4& values = reg_data[r / 4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const bf16 value = e == 0 ? values.x1 : e == 1 ? values.x2 : e == 2 ? values.x3 : values.x4;
            const int row = row_start + r + e;
            if constexpr (kReturnDequantized) {
              dequantized[tensor_base + (local_row_base + row) * last_logical_dim + col] = value;
            }
            output_smem[col_local][row] = static_cast<OType>(static_cast<float>(value) * scale);
          }
        }
      }
      __syncthreads();
      drain_columnwise_output(output_smem, colwise_data, global_col_base, local_row_base,
                              tensor_base, group_rows, last_logical_dim, tid);
      if constexpr (kDebugTrace) {
        if (leading_thread) trace->compute_done = clock64();
      }
    } else {
#pragma unroll
      for (int pass = 0; pass < kColumnPasses; ++pass) {
      const int col_local = pass * kColsPerPass + col_in_pass;
      const size_t col = global_col_base + col_local;
      float amax = 0.f;
      // Keep the reconstructed values packed as BF16 pairs.  The previous
      // float[64] staging consumed one 32-bit register per value even though
      // the dequantized result is rounded to BF16 before both output paths.
      // Packing four values cuts the staging footprint in half and leaves
      // more room for resident CTAs on Hopper.
      ptx::bf16x4 reg_data[kRowsPerThread / 4];
#pragma unroll
      for (int r = 0; r < kRowsPerThread; r += 4) {
        reg_data[r / 4] = dequantize_fp8x4(
            input_tile[(row_start + r + 0) * kTileDim + col_local],
            input_tile[(row_start + r + 1) * kTileDim + col_local],
            input_tile[(row_start + r + 2) * kTileDim + col_local],
            input_tile[(row_start + r + 3) * kTileDim + col_local], row_scales[row_start + r + 0],
            row_scales[row_start + r + 1], row_scales[row_start + r + 2],
            row_scales[row_start + r + 3]);
        const ptx::bf16x4 &values = reg_data[r / 4];
        amax = fmaxf(amax, fabsf(static_cast<float>(values.x1)));
        amax = fmaxf(amax, fabsf(static_cast<float>(values.x2)));
        amax = fmaxf(amax, fabsf(static_cast<float>(values.x3)));
        amax = fmaxf(amax, fabsf(static_cast<float>(values.x4)));
      }
      amax = subwarp_reduce_max_broadcast<kThreadsPerCol>(amax);
      const float scale = compute_scale_from_types<float, OType>(amax, epsilon, power_of_two);
      if (sub == 0) {
        const size_t scale_stride = DIVUP_TO_MULTIPLE(last_logical_dim, 4);
        const size_t group_scale_base = (group_start / kTileDim) * scale_stride;
        colwise_scale[group_scale_base + local_tile_y * scale_stride + col] = __frcp_rn(scale);
      }
#pragma unroll
      for (int r = 0; r < kRowsPerThread; r += 16) {
        if constexpr (kReturnDequantized) {
#pragma unroll
          for (int e = 0; e < 16; ++e) {
            const ptx::bf16x4& value = reg_data[(r + e) / 4];
            dequantized[tensor_base + (local_row_base + row_start + r + e) * last_logical_dim + col] =
                (e % 4 == 0) ? value.x1
                : (e % 4 == 1) ? value.x2
                : (e % 4 == 2) ? value.x3
                               : value.x4;
          }
        }
        store_columnwise_bf16<OType, 16>(
            colwise_data, &reg_data[r / 4], scale, tensor_base, col, group_rows,
            local_row_base + row_start + r);
      }
      }
      if constexpr (kDebugTrace) {
        __syncthreads();
        if (leading_thread) trace->compute_done = clock64();
      }
    }
  } else if constexpr (kReturnDequantized) {
    // Dequantized-only output does not need a column transpose. Keep the
    // compact four-thread column mapping to avoid changing its established
    // CUDA-graph and numerical behavior.
    constexpr int kThreadsPerCol = 4;
    constexpr int kColsPerPass = kThreadsPerBlock / kThreadsPerCol;
    constexpr int kColumnPasses = kTileDim / kColsPerPass;
    constexpr int kRowsPerThread = kTileDim / kThreadsPerCol;
    const int col_in_pass = tid / kThreadsPerCol;
    const int thr_row = tid % kThreadsPerCol;
    const int row_start = thr_row;
    const auto* input_tile = reinterpret_cast<const IType*>(input_smem);
#pragma unroll
    for (int pass = 0; pass < kColumnPasses; ++pass) {
      const int thr_col = pass * kColsPerPass + col_in_pass;
      const size_t col = global_col_base + thr_col;
      const bool col_valid = col < last_logical_dim;
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
        const int row_local = row_start + r * kThreadsPerCol;
        const size_t global_row = global_row_base + row_local;
        if (global_row < live_end && col_valid) {
          const float value = static_cast<float>(static_cast<bf16>(
              static_cast<float>(input_tile[row_local * kTileDim + thr_col]) *
              row_scales[row_local]));
          dequantized[tensor_base + (local_row_base + row_local) * last_logical_dim + col] =
              static_cast<bf16>(value);
        }
      }
    }
    if constexpr (kDebugTrace) {
      __syncthreads();
      if (leading_thread) trace->compute_done = clock64();
    }
  }
  if constexpr (kDebugTrace) {
    __syncthreads();
    if (leading_thread) trace->drain_done = clock64();
  }
}

template <typename Traits, typename IType, bool kReturnDequantized, bool kReturnColumnwise,
          bool kReturnRowwiseScale, bool kDebugTrace>
void launch_block128_group_requantize(const GroupedTensor* input, GroupedTensor* output,
                                     Tensor* dequantized, size_t first_logical_dim,
                                     size_t last_logical_dim, size_t row_scale_cols,
                                     bool compact_scale_layout, size_t col_scale_rows,
                                     float epsilon, bool power_of_two, cudaStream_t stream) {
  using OType = fp8e4m3;
  if (first_logical_dim == 0 || last_logical_dim == 0) return;

  NVTE_CHECK(last_logical_dim % kTileDim == 0, "EP block128 requant requires hidden divisible by 128");
  if constexpr (kReturnColumnwise) {
    NVTE_CHECK(output->columnwise_data.dptr != nullptr && output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise requantization requires both data and scale output storage.");
  }
  if constexpr (kReturnRowwiseScale) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr,
               "Rowwise scale conversion requires output storage.");
  }
  const size_t smem_bytes = kReturnColumnwise || kReturnDequantized
                               ? kTileDim * kTileDim * sizeof(IType) + TMA_SHMEM_ALIGNMENT - 1
                               : 0;

  CUtensorMap tensor_map_input{};
  create_2D_tensor_map(tensor_map_input, input->data, first_logical_dim, last_logical_dim, kTileDim, kTileDim,
                       last_logical_dim, 0, TypeInfo<IType>::size);


  const unsigned int grid_x = static_cast<unsigned int>(last_logical_dim / kTileDim);
  dim3 grid(grid_x, (first_logical_dim + kTileDim - 1) / kTileDim);
  if constexpr (Traits::SHAPE_REPRESENTATION == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t group_rows = first_logical_dim / input->num_tensors;
    grid.y = (group_rows + kTileDim - 1) / kTileDim;
    grid.z = input->num_tensors;
  }

  RequantizeDebugTrace* debug_trace = nullptr;
  const size_t trace_count = static_cast<size_t>(grid.x) * grid.y * grid.z;
  if constexpr (kDebugTrace) {
    NVTE_CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&debug_trace),
                                    trace_count * sizeof(RequantizeDebugTrace), stream));
    NVTE_CHECK_CUDA(cudaMemsetAsync(debug_trace, 0,
                                    trace_count * sizeof(RequantizeDebugTrace), stream));
  }

  auto* kernel = block128_group_requantize_kernel<Traits, IType, OType, kReturnDequantized,
                                                 kReturnColumnwise, kReturnRowwiseScale,
                                                 kDebugTrace>;
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(smem_bytes)));
  kernel<<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
      tensor_map_input, reinterpret_cast<const IType*>(input->data.dptr),
      reinterpret_cast<const float*>(input->scale_inv.dptr),
      kReturnRowwiseScale ? reinterpret_cast<float*>(output->scale_inv.dptr) : nullptr,
      kReturnColumnwise ? reinterpret_cast<OType*>(output->columnwise_data.dptr) : nullptr,
      kReturnColumnwise ? reinterpret_cast<float*>(output->columnwise_scale_inv.dptr) : nullptr,
      kReturnDequantized ? reinterpret_cast<bf16*>(dequantized->data.dptr) : nullptr,
      first_logical_dim, last_logical_dim, row_scale_cols, compact_scale_layout, col_scale_rows,
      reinterpret_cast<const int64_t*>(input->tensor_offsets.dptr),
      reinterpret_cast<const int64_t*>(input->first_dims.dptr),
      input->num_tensors, epsilon, power_of_two, debug_trace);
  NVTE_CHECK_CUDA(cudaGetLastError());

  if constexpr (kDebugTrace) {
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<RequantizeDebugTrace> traces(trace_count);
    NVTE_CHECK_CUDA(cudaMemcpy(traces.data(), debug_trace,
                               trace_count * sizeof(RequantizeDebugTrace),
                               cudaMemcpyDeviceToHost));
    int device = 0;
    int clock_khz = 0;
    NVTE_CHECK_CUDA(cudaGetDevice(&device));
    NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
    uint64_t tma_wait_min = UINT64_MAX, compute_min = UINT64_MAX, drain_min = UINT64_MAX;
    uint64_t tma_wait_max = 0, compute_max = 0, drain_max = 0;
    uint64_t tma_wait_sum = 0, compute_sum = 0, drain_sum = 0;
    size_t valid = 0;
    for (const auto& item : traces) {
      if (item.tma_issue == 0 || item.tma_ready < item.tma_issue ||
          item.compute_done < item.tma_ready || item.drain_done < item.compute_done) {
        continue;
      }
      const uint64_t tma_wait = item.tma_ready - item.tma_issue;
      const uint64_t compute = item.compute_done - item.tma_ready;
      const uint64_t drain = item.drain_done - item.compute_done;
      tma_wait_min = std::min(tma_wait_min, tma_wait);
      compute_min = std::min(compute_min, compute);
      drain_min = std::min(drain_min, drain);
      tma_wait_max = std::max(tma_wait_max, tma_wait);
      compute_max = std::max(compute_max, compute);
      drain_max = std::max(drain_max, drain);
      tma_wait_sum += tma_wait;
      compute_sum += compute;
      drain_sum += drain;
      ++valid;
    }
    if (valid != 0) {
      const auto us = [clock_khz](uint64_t cycles) {
        return static_cast<double>(cycles) * 1000.0 / static_cast<double>(clock_khz);
      };
      std::fprintf(stderr,
                   "[block128_requant_trace] tiles=%zu clock_khz=%d "
                   "tma_wait_us(avg/min/max)=%.3f/%.3f/%.3f "
                   "compute_us(avg/min/max)=%.3f/%.3f/%.3f "
                   "drain_us(avg/min/max)=%.3f/%.3f/%.3f\n",
                   valid, clock_khz, us(tma_wait_sum / valid), us(tma_wait_min), us(tma_wait_max),
                   us(compute_sum / valid), us(compute_min), us(compute_max),
                   us(drain_sum / valid), us(drain_min), us(drain_max));
    }
    NVTE_CHECK_CUDA(cudaFreeAsync(debug_trace, stream));
  }
}

}  // namespace

void group_requantize_block_scaling(const GroupedTensor& input, GroupedTensor* output,
                                   Tensor* dequantized, const QuantizationConfig* quant_config,
                                   cudaStream_t stream) {
  NVTE_CHECK(input.logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Block scaling requant requires 2D grouped shapes.");
  const size_t first_logical_dim = input.logical_shape.data[0];
  const size_t last_logical_dim = input.logical_shape.data[1];
  NVTE_CHECK(output->logical_shape.data[0] == first_logical_dim &&
                 output->logical_shape.data[1] == last_logical_dim &&
                 input.num_tensors == output->num_tensors,
             "Block scaling requant input and output grouped shapes must match.");
  const ShapeRepresentation shape_rep = input.all_same_shape()
      ? ShapeRepresentation::SAME_BOTH_DIMS
      : input.all_same_last_dim() ? ShapeRepresentation::VARYING_FIRST_DIM
      : input.all_same_first_dim() ? ShapeRepresentation::VARYING_LAST_DIM
                                  : ShapeRepresentation::VARYING_BOTH_DIMS;
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM,
             "Block scaling requant supports only SAME_BOTH_DIMS and VARYING_FIRST_DIM.");
  NVTE_CHECK(input.all_same_first_dim() == output->all_same_first_dim() &&
                 input.all_same_last_dim() == output->all_same_last_dim(),
             "Input and output grouped shape representations must match.");
  if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM ||
      input.tensor_offsets.dptr != nullptr || input.tensor_offsets.numel() != 0) {
    NVTE_CHECK(input.tensor_offsets.shape == std::vector<size_t>{input.num_tensors + 1},
               "Block scaling requant requires num_tensors + 1 offsets.");
    NVTE_CHECK(input.tensor_offsets.dtype == DType::kInt64,
               "Block scaling grouped requant requires int64 tensor offsets.");
  }
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS && input.num_tensors != 0) {
    NVTE_CHECK(first_logical_dim % input.num_tensors == 0,
               "SAME_BOTH_DIMS requires an integral row count per tensor.");
    NVTE_CHECK((first_logical_dim / input.num_tensors) % kTileDim == 0,
               "SAME_BOTH_DIMS requires each tensor's row count divisible by 128.");
  }
  if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM) {
    NVTE_CHECK(input.first_dims.dtype == DType::kInt64 &&
                   input.first_dims.shape == std::vector<size_t>{input.num_tensors},
               "VARYING_FIRST_DIM requires num_tensors int64 first dimensions.");
  }
  NVTE_CHECK(input.data.numel() == first_logical_dim * last_logical_dim &&
                 output->data.dptr == input.data.dptr,
             "Block scaling requant output rowwise data must alias the input.");
  NVTE_CHECK(input.scaling_mode == NVTE_BLOCK_SCALING_1D,
             "Block scaling grouped requant requires 1D block scaling input.");
  NVTE_CHECK((input.data.dtype == DType::kFloat8E4M3 || input.data.dtype == DType::kFloat8E5M2) &&
                 input.scale_inv.dtype == DType::kFloat32,
             "Block scaling grouped requant requires 2D FP8 input and FP32 scales.");
  NVTE_CHECK(output->scaling_mode == NVTE_BLOCK_SCALING_1D,
             "Block scaling grouped requant requires 1D block scaling output.");
  NVTE_CHECK((output->columnwise_data.dptr == nullptr) ==
                 (output->columnwise_scale_inv.dptr == nullptr),
             "Block scaling columnwise data and scales must be supplied together.");
  if (dequantized != nullptr) {
    NVTE_CHECK(dequantized->data.dtype == DType::kBFloat16 &&
                   dequantized->data.shape ==
                       std::vector<size_t>({first_logical_dim, last_logical_dim}),
               "Block scaling dequantized output must be BF16 with the input shape.");
  }

  NVTE_CHECK(last_logical_dim > 0 && last_logical_dim % kTileDim == 0,
             "Block scaling grouped requant requires hidden divisible by 128.");
  const size_t blocks = last_logical_dim / kTileDim;
  if (input.with_gemm_swizzled_scales) {
    NVTE_CHECK(input.scale_inv.numel() == first_logical_dim * blocks,
               "Block scaling GEMM rowwise scale capacity must match logical shape.");
  } else {
    NVTE_CHECK(first_logical_dim == 0 || input.scale_inv.numel() % first_logical_dim == 0,
               "Block scaling compact scales must contain whole rows.");
  }
  NVTE_CHECK(output->scale_inv.numel() == first_logical_dim * blocks,
             "Block scaling output rowwise scale capacity must match logical shape.");
  NVTE_CHECK(input.with_gemm_swizzled_scales ==
                 (input.scale_inv.dptr == output->scale_inv.dptr) || first_logical_dim == 0,
             "Block scaling compact-to-GEMM scale conversion requires separate storage.");
  const bool compact_scale_layout = !input.with_gemm_swizzled_scales;
  const size_t row_scale_cols = compact_scale_layout && first_logical_dim != 0
                                   ? input.scale_inv.numel() / first_logical_dim : blocks;
  NVTE_CHECK(row_scale_cols >= blocks,
             "Block scaling compact scales must have at least one scale per column block.");
  NVTE_CHECK(output->scale_inv.dtype == DType::kFloat32,
             "Block scaling output rowwise scales must be FP32.");
  if (output->columnwise_data.dptr != nullptr) {
    const size_t scale_rows = (first_logical_dim + kTileDim - 1) / kTileDim;
    NVTE_CHECK(output->columnwise_data.dtype == DType::kFloat8E4M3 &&
                   output->columnwise_data.numel() == first_logical_dim * last_logical_dim &&
                   output->columnwise_scale_inv.dtype == DType::kFloat32 &&
                   output->columnwise_scale_inv.numel() >=
                       scale_rows * DIVUP_TO_MULTIPLE(last_logical_dim, 4),
               "Block scaling columnwise output must be E4M3 [hidden, rows] with sufficient "
               "FP32 scale storage for [ceil(rows/128), roundup(hidden, 4)].");
  }

  // Empty work needs valid metadata, but does not dereference backing storage.
  if (first_logical_dim == 0 || input.num_tensors == 0) {
    output->with_gemm_swizzled_scales = true;
    return;
  }

  NVTE_CHECK(input.data.dptr != nullptr && input.scale_inv.dptr != nullptr &&
                 output->scale_inv.dptr != nullptr,
             "Block scaling grouped requant requires rowwise data and scales.");
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 input.tensor_offsets.dptr != nullptr,
             "Block scaling grouped requant requires tensor offsets storage.");
  NVTE_CHECK(dequantized == nullptr || dequantized->data.dptr != nullptr,
             "Block scaling dequantized output requires storage.");

  output->with_gemm_swizzled_scales = true;

  const size_t col_scale_rows = (first_logical_dim + kTileDim - 1) / kTileDim;
  const float epsilon = quant_config == nullptr ? 0.0f : quant_config->amax_epsilon;
  const bool power_of_two = quant_config == nullptr ? true : quant_config->force_pow_2_scales;
  const bool debug_trace = std::getenv("NVTE_REQUANT_DEBUG_TRACE") != nullptr;

  TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(shape_rep, SHAPE_REP, {
    if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                  SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
      using ActiveTraits = RequantizeTraits<SHAPE_REP>;
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          input.data.dtype, IType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              compact_scale_layout, RETURN_ROWWISE_SCALE,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  dequantized != nullptr, RETURN_DEQUANTIZED,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      output->columnwise_data.dptr != nullptr, RETURN_COLUMNWISE,
                      TRANSFORMER_ENGINE_SWITCH_CONDITION(
                          debug_trace, DEBUG_TRACE,
                          launch_block128_group_requantize<
                              ActiveTraits, IType, RETURN_DEQUANTIZED, RETURN_COLUMNWISE,
                              RETURN_ROWWISE_SCALE, DEBUG_TRACE>(
                              &input, output, dequantized, first_logical_dim, last_logical_dim,
                              row_scale_cols, compact_scale_layout, col_scale_rows, epsilon,
                              power_of_two, stream););););););  // NOLINT(*)
    } else {
      NVTE_ERROR("Block scaling requant supports only SAME_BOTH_DIMS and VARYING_FIRST_DIM.");
    }
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace requantize
}  // namespace transformer_engine
