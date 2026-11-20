/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cuda/barrier>
#include <utility>

#include "common/common.h"
#include "common/recipe/recipe_common.cuh"
#include "common/transpose/cast_transpose.h"
#include "common/util/cuda_runtime.h"
#include "common/utils.cuh"

namespace transformer_engine {
namespace {

using transformer_engine::detail::FP8BlockwiseColumnwiseOption;
using transformer_engine::detail::FP8BlockwiseRowwiseOption;

// clang-format off
/*

Step 1: Load input to shared memory
* shard memory: 128x128 elements with type=InputType (below graph doesn't consider padding)
* 8 warps
* Loop 8 times
* What each thread does in each loop:
    * 8 elements are read from the input at a time
    * 2 elements are written to the shared memory at a time, for a total of 4 times
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|  T0   |  T1   |  T2   |  T3   |  T4   |  T5   |  T6   |  T7   |  T8   |  T9   |  T10  |  T11  |  T12  |  T13  |  T14  |  T15  |
|  T16  |  T17  |  T18  |  T19  |  T20  |  T21  |  T22  |  T23  |  T24  |  T25  |  T26  |  T27  |  T28  |  T29  |  T30  |  T31  |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|                                                             Warp 1                                                            |
|                                                                                                                               |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|                                                             Warp 7                                                            |
|                                                                                                                               |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                          Loop 8 times                                                         |
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+

Step 2: Cast and store to output_c
* shard memory: 128x128 elements with type=InputType (below graph doesn't consider padding)
* 8 warps
* Loop 4 times
* What each thread does in each loop:
    * 2 elements are read from the shared memory at a time, for a total of 8 times
    * Every 8 consecutive threads do reduction and calculate the amax of each row
    * 16 elements are quantized and write to output_c at a time
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|      T0       |      T1       |      T2       |      T3       |      T4       |      T5       |      T6       |      T7       |
|      T8       |      T9       |      T10      |      T11      |      T12      |      T13      |      T14      |      T15      |
|      T16      |      T17      |      T18      |      T19      |      T20      |      T21      |      T22      |      T23      |
|      T24      |      T25      |      T26      |      T27      |      T28      |      T29      |      T30      |      T31      |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|                                                                                                                               |
|                                                             Warp 1                                                            |
|                                                                                                                               |
|                                                                                                                               |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|                                                                                                                               |
|                                                             Warp 7                                                            |
|                                                                                                                               |
|                                                                                                                               |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                          Loop 4 times                                                         |
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
|                                                              ...                                                              |
+-------------------------------+-------------------------------+-------------------------------+-------------------------------+

Step 3 (if columnwise transpose is True, GEMM_READY): Transpose, cast and store to output_t
* shard memory: 128x128 elements with type=InputType (below graph doesn't consider padding)
* 8 warps
* Loop 2 times
* What each thread does in each loop:
    * 2 elements (in a row) are read from the shared memory at a time, for a total of 16 times
    * Every 8 consecutive threads do reduction and calculate the amax of each column
    * 16 elements are quantized and write to output_t at a time, for a total of 2 times
+------8 elements-------+------8 elements-------+-----40 elements-------+------8 elements-------+------8 elements-------+------8 elements-------+-----40 elements-------+------8 elements-------+
| T0  | T8  | T16 | T24 |                       |                       |                       | T0  | T8  | T16 | T24 |                       |                       |                       |
| T1  | T9  | T17 | T25 |                       |                       |                       | T1  | T9  | T17 | T25 |                       |                       |                       |
| T2  | T10 | T18 | T26 |                       |                       |                       | T2  | T10 | T18 | T26 |                       |                       |                       |
| T3  | T11 | T19 | T27 |        Warp 1         |         ...           |        Warp 7         | T3  | T11 | T19 | T27 |        Warp 1         |         ...           |        Warp 7         |
| T4  | T12 | T20 | T28 |                       |                       |                       | T4  | T12 | T20 | T28 |                       |                       |                       |
| T5  | T13 | T21 | T29 |                       |                       |                       | T5  | T13 | T21 | T29 |                       |                       |                       |
| T6  | T14 | T22 | T30 |                       |                       |                       | T6  | T14 | T22 | T30 |                       |                       |                       |
| T7  | T15 | T23 | T31 |                       |                       |                       | T7  | T15 | T23 | T31 |                       |                       |                       |
+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+-----------------------+

Step 4 (kColumnwiseNotTransposed): Cast and store to output_t with the input layout
* shared memory: 128x128 elements with type=InputType (below graph doesn't consider padding), then
  8x128 per-warp column amaxes and 128 column scales with type=float
* 8 warps, each one handles 16 rows
* What each thread does:
    * Reads 2 elements from the shared memory at a time, 2 times per row, for 16 rows, and writes the
      amax of each of its 4 columns over these rows to the shared memory
    * Thread i < 128 reduces the 8 warp amaxes of column i, writes scale_inv_t of column i and writes
      the scale of column i to the shared memory
    * Reads 2 elements from the shared memory at a time, quantizes them and writes them to output_t,
      2 times per row, for 16 rows
+---------------64 elements---------------+---------------64 elements---------------+
| T0 | T1 | T2 |          ...        | T31 | T0 | T1 | T2 |          ...        | T31 |
|                                  Warp 0 (16 rows)                                 |
+-----------------------------------------+-----------------------------------------+
|                                  Warp 1 (16 rows)                                 |
+-----------------------------------------+-----------------------------------------+
|                                        ...                                        |
+-----------------------------------------+-----------------------------------------+
|                                  Warp 7 (16 rows)                                 |
+-----------------------------------------+-----------------------------------------+

*/
// clang-format on

constexpr size_t kThreadsPerWarp = 32;

// Hyperparameters for performance tuning
constexpr int kTileDim = 128;  // Fixed to 128 beacause we are using 1x128 and 128x1 quantization
constexpr int kNVecIn = 8;     // The number of elements each LDG touches
constexpr int kNVecOut = 16;   // The number of elements each STG touches in Steps 2-3
constexpr int kNVecSMem = 2;   // The number of elements each LDS/STS touches
constexpr int kThreadsPerBlock = 256;  // Thread block size, 8 warps in total

// Auto-calculated constants, do not modify directly)
static_assert(kNVecIn % kNVecSMem == 0, "kNVecIn must be divisible by kNVecSMem");
static_assert(kNVecOut % kNVecSMem == 0, "kNVecOut must be divisible by kNVecSMem");
constexpr int kSMemRow = kTileDim;
constexpr int kSMemCol = (kTileDim / kNVecSMem) + 1;
constexpr int kSMemSize = kSMemRow * kSMemCol * kNVecSMem;
constexpr int kNumThreadsLoad = kTileDim / kNVecIn;
constexpr int kNumThreadsStore = kTileDim / kNVecOut;
static_assert(kNumThreadsLoad <= kThreadsPerWarp, "kNumThreadsLoad must be <= kThreadsPerWarp");
static_assert(kNumThreadsStore <= kThreadsPerWarp, "kNumThreadsStore must be <= kThreadsPerWarp");
constexpr int kNumWarps = kThreadsPerBlock / kThreadsPerWarp;
// Number of compute-type elements Step 4 keeps in shared memory after the input tile: per-warp
// column amaxes and column scales
constexpr int kStep4SMemSize = (kNumWarps + 1) * kTileDim;

// kColumnwiseNotTransposed instantiates Step 4 for COLUMNWISE_COMPACT. Otherwise, the kernel
// runs Step 3 for COLUMNWISE_GEMM_READY.
template <bool kAligned, bool kColumnwiseNotTransposed, typename CType, typename IType,
          typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock) block_scaled_1d_cast_transpose_kernel(
    const IType* const input, OType* const output_c, OType* const output_t,
    CType* const tile_scales_inv_c, CType* const tile_scales_inv_t, const size_t row_length,
    const size_t num_rows, const size_t scale_stride_x, const size_t scale_stride_y,
    const size_t scale_t_stride_x, const size_t scale_t_stride_y, const float epsilon,
    FP8BlockwiseRowwiseOption rowwise_option, FP8BlockwiseColumnwiseOption columnwise_option,
    const bool pow_2_scaling, const float* noop_ptr) {
  // skip execution if noop
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  bool return_rowwise = rowwise_option != FP8BlockwiseRowwiseOption::NONE;
  bool return_columnwise_gemm_ready =
      !kColumnwiseNotTransposed &&
      columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY;

  using SMemVec = Vec<IType, kNVecSMem>;
  using OVec = Vec<OType, kNVecOut>;
  union IVec {
    Vec<IType, kNVecIn> input_type;
    Vec<SMemVec, kNVecIn / kNVecSMem> smem_type;
  };

  extern __shared__ char smem_base[];
  SMemVec* smem = reinterpret_cast<SMemVec*>(&smem_base[0]);

  // Step 1: Load input to shared memory
  {
    constexpr int r_stride = kThreadsPerBlock / kNumThreadsLoad;  // stride in rows of shared memory
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s =
        (threadIdx.x % kNumThreadsLoad) * (kNVecIn / kNVecSMem);  // Column in shared memory
    int r_s = threadIdx.x / kNumThreadsLoad;                      // Row in shared memory
    const size_t c_g =
        static_cast<size_t>(blockIdx.x) * kTileDim + c_s * kNVecSMem;    // Column in global memory
    size_t r_g = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;       // Row in global memory
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;  // Stride in global memory
    const size_t num_ele = c_g < row_length ? min(static_cast<size_t>(kNVecIn), row_length - c_g)
                                            : 0;            // For not aligned case
    const IType* input_g = &input[r_g * row_length + c_g];  // Input address in global memory
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      IVec input_vec;
      // Step 1.1: Load from global memory (input) to registers
      if constexpr (kAligned) {
        input_vec.input_type.load_from(input_g);
      } else {
        if (r_g < num_rows) {
          input_vec.input_type.load_from_elts(input_g, 0, num_ele);
        } else {
          input_vec.input_type.clear();
        }
      }
      // Step 1.2: Write to shared memory
#pragma unroll
      for (int i = 0; i < kNVecIn / kNVecSMem; ++i) {
        int c = c_s + i;
        int r = r_s;
        smem[r * kSMemCol + c] = input_vec.smem_type.data.elt[i];
      }
      // Step 1.3: Update input address, row index of shared memory, (and row index of global memory for not aligned case)
      input_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }

  __syncthreads();

  // Step 2: Cast and store to output_c
  if (return_rowwise) {
    constexpr int r_stride =
        kThreadsPerBlock / kNumThreadsStore;  // stride in rows of shared memory
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s =
        (threadIdx.x % kNumThreadsStore) * (kNVecOut / kNVecSMem);  // Column in shared memory
    int r_s = threadIdx.x / kNumThreadsStore;                       // Row in shared memory
    const size_t c_g =
        static_cast<size_t>(blockIdx.x) * kTileDim + c_s * kNVecSMem;    // Column in global memory
    size_t r_g = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;       // Row in global memory
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;  // Stride in global memory
    const size_t num_ele = c_g < row_length ? min(static_cast<size_t>(kNVecOut), row_length - c_g)
                                            : 0;          // For not aligned case
    OType* output_g = &output_c[r_g * row_length + c_g];  // Output address in global memory
    // Each kNumThreadsStore threads form a warp process one row, we need to find the lane id of
    // the first thread to do the reduction.
    const unsigned src_lane = (threadIdx.x % kThreadsPerWarp) / kNumThreadsStore * kNumThreadsStore;
    // This mask represents which threads should do the reduction together.
    const unsigned mask = ((1 << kNumThreadsStore) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsStore) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut / kNVecSMem];
      // Step 2.1: Load from shared memory to registers
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
        int c = c_s + i;
        int r = r_s;
        smem_vec[i] = smem[r * kSMemCol + c];
      }
      // Step 2.2: Compute local amax
      CType amax = 0;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, fabsf(smem_vec[i].data.elt[j]));
        }
      }
      // Step 2.3: Reduce amax
#pragma unroll
      for (int delta = kNumThreadsStore / 2; delta > 0; delta /= 2) {
        const float other_amax = __shfl_down_sync(mask, amax, delta);
        __builtin_assume(amax >= 0);
        __builtin_assume(other_amax >= 0);
        amax = fmaxf(amax, other_amax);
      }
      amax = __shfl_sync(mask, amax, src_lane);
      CType scale;
      // Step 2.4: Compute scale
      scale = compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scaling);
      // Step 2.5: Write scale_inv
      bool write_scale_inv = is_src_lane;
      if constexpr (!kAligned) {
        write_scale_inv &= (r_g < num_rows);
      }
      if (write_scale_inv) {
        CType scale_inv = 1.0 / scale;
        size_t row_idx = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;
        size_t col_idx = static_cast<size_t>(blockIdx.x);
        tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
      }
      // Step 2.6: Quantize
      OVec output_vec;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          output_vec.data.elt[i * kNVecSMem + j] =
              static_cast<OType>(static_cast<CType>(smem_vec[i].data.elt[j]) * scale);
        }
      }
      // Step 2.7: Store output_c
      if constexpr (kAligned) {
        output_vec.store_to(output_g);
      } else {
        if (r_g < num_rows) {
          output_vec.store_to_elts(output_g, 0, num_ele);
        }
      }
      // Step 2.8: Update output address, row index of shared memory (and row index of global memory for not aligned case)
      output_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }

  // Step 3 (return_columnwise_gemm_ready): Transpose, cast and store to output_t
  if (return_columnwise_gemm_ready) {
    constexpr int c_stride =
        kThreadsPerBlock / kNumThreadsStore;  // Stride in columns of shared memory
    constexpr int num_iterations = kTileDim / (c_stride * kNVecSMem);
    const int r_s = (threadIdx.x % kNumThreadsStore) * kNVecOut;  // Row in shared memory
    int c_s = threadIdx.x / kNumThreadsStore;                     // Column in shared memory
    size_t r_g =
        static_cast<size_t>(blockIdx.x) * kTileDim + c_s * kNVecSMem;     // Row in global memory
    const size_t c_g = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;  // Column in global memory
    const size_t stride_g =
        static_cast<size_t>(c_stride) * kNVecSMem * num_rows;  // Stride in global memory
    const size_t num_ele = c_g < num_rows ? min(static_cast<size_t>(kNVecOut), num_rows - c_g)
                                          : 0;          // For not aligned case
    OType* output_g = &output_t[r_g * num_rows + c_g];  // Output address in global memory
    // Each kNumThreadsStore threads form a warp process one row, we need to find the lane id of
    // the first thread to do the reduction.
    const unsigned src_lane = (threadIdx.x % kThreadsPerWarp) / kNumThreadsStore * kNumThreadsStore;
    // This mask represents which threads should do the reduction together.
    const unsigned mask = ((1 << kNumThreadsStore) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsStore) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut];
      // Step 3.1: Load from shared memory to registers
#pragma unroll
      for (int i = 0; i < kNVecOut; ++i) {
        int r = r_s + i;
        int c = c_s;
        smem_vec[i] = smem[r * kSMemCol + c];
      }
#pragma unroll
      for (int smem_idx = 0; smem_idx < kNVecSMem; ++smem_idx) {
        // Step 3.2: Compute local amax
        CType amax = 0;
#pragma unroll
        for (int i = 0; i < kNVecOut; ++i) {
          amax = fmaxf(amax, fabsf(smem_vec[i].data.elt[smem_idx]));
        }
        // Step 3.3: Reduce amax
#pragma unroll
        for (int delta = kNumThreadsStore / 2; delta > 0; delta /= 2) {
          const float other_amax = __shfl_down_sync(mask, amax, delta);
          __builtin_assume(amax >= 0);
          __builtin_assume(other_amax >= 0);
          amax = fmaxf(amax, other_amax);
        }
        amax = __shfl_sync(mask, amax, src_lane);
        // Step 3.4: Compute scale
        CType scale;
        scale = compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scaling);
        // Step 3.5: Write scale_inv_t
        bool write_scale_inv = is_src_lane;
        if constexpr (!kAligned) {
          write_scale_inv &= (r_g + smem_idx < row_length);
        }
        if (write_scale_inv) {
          CType scale_inv = 1.0 / scale;
          size_t row_idx = static_cast<size_t>(blockIdx.x) * kTileDim + c_s * kNVecSMem + smem_idx;
          size_t col_idx = static_cast<size_t>(blockIdx.y);
          tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
        }
        // Step 3.6: Quantize
        OVec output_vec;
#pragma unroll
        for (int i = 0; i < kNVecOut; ++i) {
          output_vec.data.elt[i] =
              static_cast<OType>(static_cast<CType>(smem_vec[i].data.elt[smem_idx]) * scale);
        }
        // Step 3.7: Store output_t
        if constexpr (kAligned) {
          output_vec.store_to(output_g + smem_idx * num_rows);
        } else {
          if (r_g + smem_idx < row_length) {
            output_vec.store_to_elts(output_g + smem_idx * num_rows, 0, num_ele);
          }
        }
      }
      // Step 3.8: Update output address, column index of shared memory (and row index of global memory for not aligned case)
      output_g += stride_g;
      c_s += c_stride;
      if constexpr (!kAligned) {
        r_g += c_stride * kNVecSMem;
      }
    }
  }

  // Step 4 (kColumnwiseNotTransposed): Cast in 128x1 blocks and store to output_t with the
  // input layout
  if constexpr (kColumnwiseNotTransposed) {
    // Warp w handles rows [w * kRowsPerWarp, (w + 1) * kRowsPerWarp) of the tile. In each row,
    // lane l handles shared memory vectors l and l + kThreadsPerWarp, so the warp accesses
    // contiguous shared and global memory.
    constexpr int kRowsPerWarp = kTileDim / kNumWarps;
    // Shared memory vectors per thread in each row
    constexpr int kNumVecs = kTileDim / (kThreadsPerWarp * kNVecSMem);
    using CVec = Vec<CType, kNVecSMem>;
    using OSMemVec = Vec<OType, kNVecSMem>;
    CType* const smem_amax = reinterpret_cast<CType*>(&smem_base[kSMemSize * sizeof(IType)]);
    CType* const smem_scale = smem_amax + kNumWarps * kTileDim;
    const int lane = threadIdx.x % kThreadsPerWarp;
    const int warp = threadIdx.x / kThreadsPerWarp;
    const int r_s = warp * kRowsPerWarp;  // First row in shared memory
    int c_s[kNumVecs];                    // Columns in shared memory
#pragma unroll
    for (int v = 0; v < kNumVecs; ++v) {
      c_s[v] = lane + v * kThreadsPerWarp;
    }

    // Step 4.1: Compute the amax of each column over the warp's rows
    CVec amax[kNumVecs];
#pragma unroll
    for (int v = 0; v < kNumVecs; ++v) {
      amax[v].clear();
    }
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
#pragma unroll
      for (int v = 0; v < kNumVecs; ++v) {
        const SMemVec smem_vec = smem[(r_s + i) * kSMemCol + c_s[v]];
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          amax[v].data.elt[j] = fmaxf(amax[v].data.elt[j], fabsf(smem_vec.data.elt[j]));
        }
      }
    }
#pragma unroll
    for (int v = 0; v < kNumVecs; ++v) {
      amax[v].store_to(&smem_amax[warp * kTileDim + c_s[v] * kNVecSMem]);
    }
    __syncthreads();

    // Step 4.2: Reduce the column amaxes across warps, compute scales and write scale_inv_t
    if (threadIdx.x < kTileDim) {
      const int c = threadIdx.x;  // Column in the tile
      CType col_amax = 0;
#pragma unroll
      for (int w = 0; w < kNumWarps; ++w) {
        col_amax = fmaxf(col_amax, smem_amax[w * kTileDim + c]);
      }
      const CType col_scale =
          compute_scale_from_types<IType, OType>(col_amax, epsilon, pow_2_scaling);
      smem_scale[c] = col_scale;
      // Column in global memory
      const size_t col_g = static_cast<size_t>(blockIdx.x) * kTileDim + c;
      bool write_scale_inv = true;
      if constexpr (!kAligned) {
        write_scale_inv = col_g < row_length;
      }
      if (write_scale_inv) {
        CType scale_inv = 1.0 / col_scale;
        size_t row_idx = static_cast<size_t>(blockIdx.y);
        tile_scales_inv_t[row_idx * scale_t_stride_y + col_g * scale_t_stride_x] = scale_inv;
      }
    }
    __syncthreads();

    // Step 4.3: Quantize and store output_t
    CVec scale[kNumVecs];
    size_t c_g[kNumVecs];      // Columns in global memory
    size_t num_ele[kNumVecs];  // For not aligned case
#pragma unroll
    for (int v = 0; v < kNumVecs; ++v) {
      scale[v].load_from(&smem_scale[c_s[v] * kNVecSMem]);
      c_g[v] = static_cast<size_t>(blockIdx.x) * kTileDim + c_s[v] * kNVecSMem;
      num_ele[v] =
          c_g[v] < row_length ? min(static_cast<size_t>(kNVecSMem), row_length - c_g[v]) : 0;
    }
    size_t r_g = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;  // Row in global memory
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
#pragma unroll
      for (int v = 0; v < kNumVecs; ++v) {
        const SMemVec smem_vec = smem[(r_s + i) * kSMemCol + c_s[v]];
        OSMemVec output_vec;
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          output_vec.data.elt[j] =
              static_cast<OType>(static_cast<CType>(smem_vec.data.elt[j]) * scale[v].data.elt[j]);
        }
        OType* output_g = &output_t[r_g * row_length + c_g[v]];  // Output address in global memory
        if constexpr (kAligned) {
          output_vec.store_to(output_g);
        } else {
          if (r_g < num_rows) {
            output_vec.store_to_elts(output_g, 0, num_ele[v]);
          }
        }
      }
      ++r_g;
    }
  }
}

}  // namespace
}  // namespace transformer_engine

namespace transformer_engine::detail {

void quantize_transpose_vector_blockwise(const SimpleTensor& input, SimpleTensor& scale_inv,
                                         SimpleTensor& scale_inv_t, SimpleTensor& output,
                                         SimpleTensor& output_t, const float epsilon,
                                         FP8BlockwiseRowwiseOption rowwise_option,
                                         FP8BlockwiseColumnwiseOption columnwise_option,
                                         const bool pow2_scale, const SimpleTensor& noop_tensor,
                                         cudaStream_t stream) {
  NVTE_API_CALL(quantize_transpose_vector_blockwise);

  if (transformer_engine::cuda::sm_arch() >= 100) {
    NVTE_CHECK(pow2_scale, "On Blackwell and newer, the FP8 block scaling recipe is emulated ",
               "with MXFP8, which requires using power of two scaling factors.");
  }

  const size_t row_length = input.shape.size() > 0 ? input.shape.at(input.shape.size() - 1) : 1u;
  size_t num_elements = row_length;
  size_t num_rows = 1;
  for (size_t i = 0; (i < input.shape.size() - 1) && (input.shape.size() > 0); ++i) {
    num_rows *= input.shape.at(i);
    num_elements *= input.shape.at(i);
  }

  // Early return if the input tensor is empty
  if (num_elements == 0) {
    return;
  }

  // Options for scale layout of cuBLAS GEMM kernel.
  size_t scale_stride_x = 0;
  size_t scale_stride_y = 0;
  size_t scale_t_stride_x = 0;
  size_t scale_t_stride_y = 0;

  if (rowwise_option != FP8BlockwiseRowwiseOption::NONE) {
    NVTE_CHECK(rowwise_option == FP8BlockwiseRowwiseOption::ROWWISE_GEMM_READY ||
                   rowwise_option == FP8BlockwiseRowwiseOption::ROWWISE_COMPACT,
               "Unexpected rowwise enum value");
    NVTE_CHECK(input.shape == output.shape, "Input and output must have the same shape.");
    NVTE_CHECK(scale_inv.shape.size() == 2, "Scale dimension must be 2.");
    size_t scale_k = scale_inv.shape[1];
    bool rowwise_compact = rowwise_option == FP8BlockwiseRowwiseOption::ROWWISE_COMPACT;
    scale_stride_x = rowwise_compact ? 1 : scale_k;
    scale_stride_y = rowwise_compact ? scale_k : 1;
  }

  if (columnwise_option != FP8BlockwiseColumnwiseOption::NONE) {
    NVTE_CHECK(output_t.shape.size() == input.shape.size(),
               "output_t must have same number of dimensions as input.");

    if (output_t.shape.size() > 0) {
      if (columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY) {
        NVTE_CHECK(output_t.shape[0] == row_length, "Wrong dimension 0 of output_t.");
        for (size_t i = 1; i < output_t.shape.size(); ++i) {
          NVTE_CHECK(output_t.shape.at(i) == input.shape.at(i - 1), "Wrong dimension in output_t");
        }
      } else {
        NVTE_CHECK(columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT,
                   "Unexpected columnwise option enum value");
        NVTE_CHECK(output_t.shape[0] == input.shape[0], "Wrong dimension 0 of output_t.");
        NVTE_CHECK(
            input.shape == output_t.shape,
            "Input and output_t must have the same shape for columnwise non-transpose case.");
      }
    }
    if (rowwise_option != FP8BlockwiseRowwiseOption::NONE) {
      // output may not be defined if rowwise quantization is not needed.
      NVTE_CHECK(output.dtype == output_t.dtype,
                 "output and output_t need to have the same dtype.");
    }
    NVTE_CHECK(scale_inv_t.shape.size() == 2, "Scale_t dimension must be 2.");
    bool columnwise_compact = columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT;
    size_t scale_t_k = scale_inv_t.shape[1];
    scale_t_stride_x = columnwise_compact ? 1 : scale_t_k;
    scale_t_stride_y = columnwise_compact ? scale_t_k : 1;
  }
  auto output_dtype =
      rowwise_option != FP8BlockwiseRowwiseOption::NONE ? output.dtype : output_t.dtype;

  const size_t num_blocks_x = DIVUP(row_length, (size_t)kTileDim);
  const size_t num_blocks_y = DIVUP(num_rows, (size_t)kTileDim);

  const float* noop_ptr = reinterpret_cast<const float*>(noop_tensor.dptr);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype, InputType,

      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output_dtype, OutputType,

          dim3 grid(num_blocks_x, num_blocks_y, 1);

          const bool full_tile = row_length % kTileDim == 0 && num_rows % kTileDim == 0;
          const bool columnwise_not_transposed =
              columnwise_option == FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT;

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              full_tile, kAligned,

              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  columnwise_not_transposed, kColumnwiseNotTransposed,

                  using ComputeType = float;
                  auto kernel =
                      &block_scaled_1d_cast_transpose_kernel<kAligned, kColumnwiseNotTransposed,
                                                             ComputeType, InputType, OutputType>;
                  const size_t smem_bytes =
                      kSMemSize * sizeof(InputType) +
                      (kColumnwiseNotTransposed ? kStep4SMemSize * sizeof(ComputeType) : 0);
                  // shared memory must be requested up
                  if (smem_bytes >= 48 * 1024) {
                    cudaError_t err = cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                    NVTE_CHECK(err == cudaSuccess, "Failed to set dynamic shared memory size.");
                  }

                  kernel<<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
                      reinterpret_cast<const InputType*>(input.dptr),
                      reinterpret_cast<OutputType*>(output.dptr),
                      reinterpret_cast<OutputType*>(output_t.dptr),
                      reinterpret_cast<float*>(scale_inv.dptr),
                      reinterpret_cast<float*>(scale_inv_t.dptr), row_length, num_rows,
                      scale_stride_x, scale_stride_y, scale_t_stride_x, scale_t_stride_y, epsilon,
                      rowwise_option, columnwise_option, pow2_scale,
                      noop_ptr);)  // kColumnwiseNotTransposed
              )                    // kAligned
          )                        // OutputType
      )                            // InputType
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine::detail
