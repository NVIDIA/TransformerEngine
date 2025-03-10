/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "common/utils.cuh"
#include "compute_scale.cuh"

namespace transformer_engine {
namespace {

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

Step 3: Transpose, cast and store to output_t
* shard memory: 128x128 elements with type=InputType (below graph doesn't consider padding)
* 8 warps
* Loop 2 times
* What each thread does in each loop:
    * 2 elements (in a row) are read from the shared memory at a time, for a total of 16 times
    * Every 8 consecutive threads do reduction and calculate the amax of each column
    * 16 elements are quantized and write to output_c at a time, for a total of 2 times
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

*/
// clang-format on

constexpr size_t kThreadsPerWarp = 32;

// Hyperparameters for performance tuning
constexpr int kTileDim = 128;  // Fixed to 128 beacause we are using 1x128 and 128x1 quantization
constexpr int kNVecIn = 8;     // The number of elements each LDG touches
constexpr int kNVecOut = 16;   // The number of elements each STG touches
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

template <bool kReturnTranspose, bool kIsE8Scaling, bool kPermuteScale, bool kAligned,
          typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    block_scaled_1d_cast_transpose_kernel(const IType* const input, OType* const output_c,
                                          OType* const output_t, CType* const tile_scales_inv_c,
                                          CType* const tile_scales_inv_t, const size_t row_length,
                                          const size_t num_rows, const size_t scale_stride_x,
                                          const size_t scale_stride_y,
                                          const size_t scale_t_stride_x,
                                          const size_t scale_t_stride_y, const float epsilon) {
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
  {
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
      // Step 2.4: Compute scale
      CType scale = ComputeScale<IType, OType, kIsE8Scaling>(amax, epsilon);
      // Step 2.5: Write scale_inv
      bool write_scale_inv = is_src_lane;
      if constexpr (!kAligned) {
        write_scale_inv &= (r_g < num_rows);
      }
      if (write_scale_inv) {
        CType scale_inv = 1.0 / scale;
        size_t row_idx = static_cast<size_t>(blockIdx.y) * kTileDim + r_s;
        size_t col_idx = static_cast<size_t>(blockIdx.x);
        if constexpr (kPermuteScale) {
          size_t p_row = row_idx / kTileDim;
          size_t p_col = col_idx;
          size_t p_dep = row_idx % kTileDim;
          size_t p_2d_stride = kTileDim * scale_stride_y;
          tile_scales_inv_c[p_row * p_2d_stride + p_col * kTileDim + p_dep] = scale_inv;
        } else {
          tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
        }
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

  // Step 3: Transpose, cast and store to output_t
  if constexpr (kReturnTranspose) {
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
        CType scale = ComputeScale<IType, OType, kIsE8Scaling>(amax, epsilon);
        // Step 3.5: Write scale_inv_t
        bool write_scale_inv = is_src_lane;
        if constexpr (!kAligned) {
          write_scale_inv &= (r_g + smem_idx < row_length);
        }
        if (write_scale_inv) {
          CType scale_inv = 1.0 / scale;
          size_t row_idx = static_cast<size_t>(blockIdx.x) * kTileDim + c_s * kNVecSMem + smem_idx;
          size_t col_idx = static_cast<size_t>(blockIdx.y);
          if constexpr (kPermuteScale) {
            size_t p_row = row_idx / kTileDim;
            size_t p_col = col_idx;
            size_t p_dep = row_idx % kTileDim;
            size_t p_2d_stride = kTileDim * scale_t_stride_y;
            tile_scales_inv_t[p_row * p_2d_stride + p_col * kTileDim + p_dep] = scale_inv;
          } else {
            tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
          }
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
}

}  // namespace
}  // namespace transformer_engine

namespace transformer_engine::detail {

void quantize_transpose_vector_blockwise(const SimpleTensor& input, SimpleTensor& scale_inv,
                                         SimpleTensor& scale_inv_t, SimpleTensor& output,
                                         SimpleTensor& output_t, const float epsilon,
                                         const bool return_transpose, const bool pow2_scale,
                                         cudaStream_t stream) {
  NVTE_API_CALL(quantize_transpose_vector_blockwise);
  NVTE_CHECK(input.shape == output.shape, "Input and output must have the same shape.");

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
  constexpr bool kPermuteScale = false;
  bool permute_scale = false;
  bool transpose_scales = true;

  NVTE_CHECK(input.shape.size() == output.shape.size(),
             "Input and output must have the same shape.");
  NVTE_CHECK((!transpose_scales || !permute_scale),
             "Permute scale and transpose scales are mutually exclusive flags.");

  size_t scale_stride_x = 0;
  size_t scale_stride_y = 0;
  if (permute_scale) {
    NVTE_CHECK(scale_inv.shape.size() == 3, "scale_inv must have 3 dimensions.");
    size_t scale_k = scale_inv.shape[1];
    NVTE_CHECK(scale_inv.shape[2] == kTileDim, "Scale inner dimension must be kTileDim.");
    scale_stride_x = 1;
    scale_stride_y = scale_k;
  } else {
    NVTE_CHECK(scale_inv.shape.size() == 2, "Scale dimension must be 2 when not permuting scale.");
    size_t scale_k = scale_inv.shape[1];
    scale_stride_x = 1;
    scale_stride_y = scale_k;
    if (transpose_scales) {
      std::swap(scale_stride_x, scale_stride_y);
    }
  }

  size_t scale_t_stride_x = 0;
  size_t scale_t_stride_y = 0;

  if (return_transpose) {
    NVTE_CHECK(output_t.shape.size() == input.shape.size(),
               "output_t must have same number of dimensions as input.");
    if (output_t.shape.size() > 0) {
      NVTE_CHECK(output_t.shape[0] == row_length, "Wrong dimension 0 of output_t.");
      for (size_t i = 1; i < output_t.shape.size(); ++i) {
        NVTE_CHECK(output_t.shape.at(i) == input.shape.at(i - 1), "Wrong dimension in output_t");
      }
    }

    NVTE_CHECK(output.dtype == output_t.dtype, "output and output_t need to have the same dtype.");

    if (permute_scale) {
      NVTE_CHECK(scale_inv_t.shape.size() == 3, "Scale_t dimension must be 3.");
      scale_t_stride_x = 1;
      scale_t_stride_y = scale_inv_t.shape[1];
      NVTE_CHECK(scale_inv_t.shape[2] == kTileDim, "Scale_t inner dimension must be kTileDim.");
    } else {
      NVTE_CHECK(scale_inv_t.shape.size() == 2,
                 "Scale_t dimension must be 2 when not permuting scale.");
      scale_t_stride_x = 1;
      scale_t_stride_y = scale_inv_t.shape[1];
      if (transpose_scales) {
        std::swap(scale_t_stride_x, scale_t_stride_y);
      }
    }
  }

  const size_t num_blocks_x = DIVUP(row_length, (size_t)kTileDim);
  const size_t num_blocks_y = DIVUP(num_rows, (size_t)kTileDim);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype, InputType,

      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output.dtype, OutputType,

          dim3 grid(num_blocks_x, num_blocks_y, 1);

          const bool full_tile = row_length % kTileDim == 0 && num_rows % kTileDim == 0;

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_transpose, kReturnTranspose,

              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  pow2_scale, kPow2Scale,

                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      full_tile, kAligned,

                      size_t smem_bytes = kSMemSize * sizeof(InputType);
                      // shared memory must be requested up
                      if (smem_bytes >= 48 * 1024) {
                        cudaError_t err = cudaFuncSetAttribute(
                            &block_scaled_1d_cast_transpose_kernel<kReturnTranspose, kPow2Scale,
                                                                   kPermuteScale, kAligned, float,
                                                                   InputType, OutputType>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                        NVTE_CHECK(err == cudaSuccess, "Failed to set dynamic shared memory size.");
                      } block_scaled_1d_cast_transpose_kernel<kReturnTranspose, kPow2Scale,
                                                              kPermuteScale, kAligned, float,
                                                              InputType, OutputType>
                      <<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
                          reinterpret_cast<const InputType*>(input.dptr),
                          reinterpret_cast<OutputType*>(output.dptr),
                          reinterpret_cast<OutputType*>(output_t.dptr),
                          reinterpret_cast<float*>(scale_inv.dptr),
                          reinterpret_cast<float*>(scale_inv_t.dptr), row_length, num_rows,
                          scale_stride_x, scale_stride_y, scale_t_stride_x, scale_t_stride_y,
                          epsilon);)  // kAligned
                  )                   // kPow2Scale
              )                       // kReturnTranspose
          )                           // OutputType
      )                               // InputType
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine::detail
