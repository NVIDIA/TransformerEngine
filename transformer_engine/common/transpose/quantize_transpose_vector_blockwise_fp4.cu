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
#include "common/recipe/recipe_common.cuh"
#include "common/transpose/cast_transpose.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"
#include "curanddx.hpp"

namespace transformer_engine {

#if CUDA_VERSION >= 12080
namespace quantize_transpose_nvfp4 {
namespace {

using std::int32_t;
using std::uint32_t;
using std::uint8_t;

using transformer_engine::detail::TypeExtrema;

// Define a cuRANDDx descriptor
// Note curanddx::PhiloxRounds<4> means 4 rounds of philox4_32. If the operator is not specified, it will be default to 10.
// curanddx::SM<800>() does NOT mean the code can only run on SM 800. The operator is used for do some internal checks, e.g.,
// if shared memory, if needed, is enough for the described problem, usually not applicable.
// curanddx doc: https://docs.nvidia.com/cuda/curanddx/index.html
using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() + curanddx::PhiloxRounds<10>() +
                     curanddx::SM<800>() + curanddx::Thread());

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

constexpr int kThreadsPerWarp = 32;

// for fp4, we use uint8_t to store 2 fp4 numbers
constexpr int kNFP4PerContainer = 2;

// Hyperparameters for performance tuning
constexpr int kTileDim = 128;
// constexpr int kScaleDim = 32;
constexpr int kNVecIn = 8;             // The number of elements each LDG touches
constexpr int kNVecOut = 16;           // The number of elements each STG touches
constexpr int kNVecSMem = 2;           // The number of elements each LDS/STS touches
constexpr int kThreadsPerBlock = 256;  // Thread block size, 8 warps in total

// Auto-calculated constants, do not modify directly)
static_assert(kNVecIn % kNVecSMem == 0, "kNVecIn must be divisible by kNVecSMem");
static_assert(kNVecOut % kNVecSMem == 0, "kNVecOut must be divisible by kNVecSMem");
constexpr int kSMemRow = kTileDim;
constexpr int kSMemCol = (kTileDim / kNVecSMem) + 1;
constexpr int kSMemSize = kSMemRow * kSMemCol * kNVecSMem;
constexpr int kNumThreadsLoad = kTileDim / kNVecIn;    // 16
constexpr int kNumThreadsStore = kTileDim / kNVecOut;  // 8
// constexpr int kNumThreadsReduce = kScaleDim / kNVecOut;
static_assert(kNumThreadsLoad <= kThreadsPerWarp, "kNumThreadsLoad must be <= kThreadsPerWarp");
static_assert(kNumThreadsStore <= kThreadsPerWarp, "kNumThreadsStore must be <= kThreadsPerWarp");

// for 2D block scaling, we need to reduce amax in warp
static __device__ constexpr unsigned int WARP_REDUCE_AMAX_GROUP_MASKS[8] = {
    0x01010101, 0x02020202, 0x04040404, 0x08080808, 0x10101010, 0x20202020, 0x40404040, 0x80808080};

// max for every group_size elements in warp
template <int group_size, int shfl_down_stride>
__device__ __forceinline__ float groupMax(float val, unsigned int groupMask) {
  for (int offset = group_size / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(groupMask, val, offset * shfl_down_stride));
  }
  return val;
}

template <typename ScaleType>
__device__ __forceinline__ ScaleType ComputeDecodeScaleFP4(const float amax,
                                                           const float global_encode_scale) {
  float decode_scale = amax / TypeExtrema<fp4e2m1>::max;
  decode_scale = decode_scale * global_encode_scale;
  decode_scale = fminf(decode_scale, TypeExtrema<float>::max);
  return static_cast<ScaleType>(decode_scale);
}

template <typename ScaleType>
__device__ __forceinline__ float ComputeEncodeScaleFP4(ScaleType decode_scale,
                                                       const float global_decode_scale) {
  return fminf(1.0f / (static_cast<float>(decode_scale) * global_decode_scale),
               TypeExtrema<float>::max);
}

template <typename IType, typename ScaleType>
__device__ __forceinline__ float ComputeOutputFP4(IType input, float encode_scale) {
  return static_cast<float>(input) * encode_scale;
}

__device__ __forceinline__ float ComputeGlobalEncodeScaleFP4(const float global_amax) {
  constexpr float fp8_max = TypeExtrema<fp8e4m3>::max;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale = fminf(global_encode_scale, TypeExtrema<float>::max);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.f || global_encode_scale == 0.f) {
    return 1.f;
  }
  return global_encode_scale;
}

__device__ __forceinline__ uint32_t get_rbits(RNG& rng, uint4& random_uint4, int& rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    curanddx::uniform_bits dist;
    random_uint4 = dist.generate4(rng);
  }
  // Treat uint4 as an array of 4x uint32_t elements for indexing
  const uint32_t* const rbits_arr = reinterpret_cast<uint32_t*>(&random_uint4);
  const uint32_t rbits = rbits_arr[rnd_idx++];
  return rbits;
}

template <class ScaleType>
__device__ __forceinline__ size_t scale_factor_swizzled_offset(size_t row_idx, size_t col_idx,
                                                               uint32_t col_length) {
  // This function takes in indices from the scale factor matrix and returns an offset in the
  // swizzled format. row_idx, col_idx are original indices from the scale factor matrix (unswizzled
  // index). col_length is the column length of the scale factor matrix. tile_scales_inv is the
  // pointer to the scale factor matrix.

  // https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
  // For any scale factor matrix, it's 512B base block. Each base block consists of 128 rows and 4
  // columns. Base block is divided into 4 column blocks, each column block has 32 rows and 4
  // columns.

  // NOTE: There are not a lot of good illustrations about the swizzled scale factor matrix.
  // To think in high level, the swizzled scale factor matrix could be composed as:
  // unswizzled_scale_factor_matrix = torch.empty((M, N // 16), dtype=torch.uint8)
  // cbg_cnt = N // 16 // 4  # Assuming N is divisible by 64
  // rb_cnt = M // 128  # Assuming M is divisible by 128
  // tmp = unswizzled_scale_factor_matrix.reshape(rb_cnt, 4, 32, cbg_cnt, 4)
  // tmp = torch.permute(tmp, (0, 3, 2, 1, 4))
  // swizzled_scale_factor_matrix = tmp.reshape((-1, 128, 4))

  constexpr uint32_t kTotalRowsPerBaseBlock = 128;
  constexpr uint32_t kRowsPerBaseBlockCol = 32;
  constexpr uint32_t kColsPerBaseBlockCol = 4;

  const size_t rb = row_idx / kTotalRowsPerBaseBlock;
  const size_t rem = row_idx % kTotalRowsPerBaseBlock;
  const size_t d4 = rem / kRowsPerBaseBlockCol;
  const size_t d3 = rem % kRowsPerBaseBlockCol;
  const size_t cbg = col_idx / kColsPerBaseBlockCol;
  const size_t d5 = col_idx % kColsPerBaseBlockCol;

  const size_t cbg_cnt = DIVUP(col_length, kColsPerBaseBlockCol);
  // row-major offset in the logical shape
  // (rb_cnt , cbg_cnt , 32 , 4 , 4)
  // Magic number 16 below comes from the fact we have kColsPerBaseBlockCol = 4, and d4 ([0-128] /
  // 32 = [0-4])
  return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16 + d4 * kColsPerBaseBlockCol + d5;
}

__device__ __forceinline__ __nv_fp4x4_e2m1 cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const uint32_t rbits) {
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  uint16_t out_4x;
  asm volatile(
      "{\n"
      "cvt.rs.satfinite.e2m1x4.f32 %0, {%3, %4, %1, %2}, %5; \n\t"
      "}"
      : "=h"(out_4x)
      : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x), "r"(rbits));
  return *reinterpret_cast<__nv_fp4x4_e2m1*>(&out_4x);
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
  uint16_t dummy = 0;
  return *reinterpret_cast<__nv_fp4x4_e2m1*>(&dummy);
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
}

__device__ __forceinline__ __nv_fp4x4_e2m1 cvt_fp32_to_fp4_4x_with_rn(const float2 in01,
                                                                      const float2 in23,
                                                                      const uint32_t rbits) {
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  // NOTE: rbits unused for rn.
  uint32_t out_4x;  // Only need 16 bit. Using 32 bit container for packing.
  asm volatile(
      "{\n"
      ".reg.b8 f0; \n\t"
      ".reg.b8 f1; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, %1, %2;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, %3, %4;\n\t"
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x));
  return reinterpret_cast<__nv_fp4x4_e2m1*>(&out_4x)[0];
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
  uint16_t dummy = 0;
  return *reinterpret_cast<__nv_fp4x4_e2m1*>(&dummy);
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
}

template <bool kApplyStochasticRounding>
__device__ __forceinline__ __nv_fp4x4_e2m1 cvt_fp32_to_fp4_4x(const float2 in01, const float2 in23,
                                                              const uint32_t rbits) {
  if constexpr (kApplyStochasticRounding) {
    return cvt_fp32_to_fp4_4x_with_stochastic_rounding(in01, in23, rbits);
  } else {
    return cvt_fp32_to_fp4_4x_with_rn(in01, in23, rbits);
  }
}

template <bool kReturnIdentity, bool kReturnTranspose, bool kIsE8Scaling, bool kAligned,
          typename CType, typename IType, typename OType, typename ScaleType, bool kSwizzledScale,
          bool kApplyStochasticRounding, bool kIs2DBlockScaling>
__global__ void __launch_bounds__(kThreadsPerBlock) block_scaled_1d_cast_transpose_kernel(
    const IType* const input, const float* global_amax, OType* const output_c,
    OType* const output_t, ScaleType* const tile_scales_inv_c, ScaleType* const tile_scales_inv_t,
    const size_t row_length, const size_t num_rows, const size_t scale_stride_x,
    const size_t scale_stride_y, const size_t scale_t_stride_x, const size_t scale_t_stride_y,
    const size_t kScaleBlockDim, const float epsilon, const size_t* rng_state,
    const float* noop_ptr) {
  constexpr int kNVecContainer = kNVecOut / kNFP4PerContainer;
  using SMemVec = Vec<IType, kNVecSMem>;
  using OVec = Vec<OType, kNVecContainer>;
  union IVec {
    Vec<IType, kNVecIn> input_type;
    Vec<SMemVec, kNVecIn / kNVecSMem> smem_type;
  };

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t block_idx_x = blockIdx.x;
  const size_t block_idx_y = blockIdx.y;
  const size_t rng_sequence =
      threadIdx.x + block_idx_x * kThreadsPerBlock + block_idx_y * gridDim.x * kThreadsPerBlock;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  RNG rng(rng_seed, rng_sequence, rng_offset);
  curanddx::uniform_bits dist;
  uint4 random_uint4 = kApplyStochasticRounding ? dist.generate4(rng) : uint4{0, 0, 0, 0};
  int rnd_idx =
      0;  // Index of the random number. It increments each time when used and resets to 0 if reaches 4x

  extern __shared__ char smem_base[];
  SMemVec* smem = reinterpret_cast<SMemVec*>(&smem_base[0]);

  // 2D block scaling is not supported for E8 scaling MXFP4 or for colwise only mode.
  // Instead of static_assert, return early if these invalid modes are detected.
  if constexpr (kIs2DBlockScaling && kIsE8Scaling) {
    return;
  }
  if constexpr (kIs2DBlockScaling && !kReturnIdentity) {
    return;
  }
  // for 128x128 block, 2D block scaling means there will be 8x8 amax values for nvfp4, 4x4 for 2D mxfp4
  // use constexpr to define the size, when not using 2D, use minimal size 1x1
  constexpr int kFP4BlockScalingSize = 16;
  constexpr int k2DBlockAmaxDim = kIs2DBlockScaling ? (kTileDim / kFP4BlockScalingSize) : 1;
  constexpr int kNumRowsPerWarp = kThreadsPerWarp / kNumThreadsStore;  // 4
  constexpr int k2DBlockAmaxReduceDim =
      kIs2DBlockScaling ? (kFP4BlockScalingSize / kNumRowsPerWarp) : 1;
  __shared__ CType amax_smem_red[k2DBlockAmaxDim][k2DBlockAmaxDim][k2DBlockAmaxReduceDim];
  __shared__ CType amax_smem[k2DBlockAmaxDim][k2DBlockAmaxDim];

  // Step 1: Load input to shared memory
  {
    constexpr int r_stride = kThreadsPerBlock / kNumThreadsLoad;  // stride in rows of shared memory
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s =
        (threadIdx.x % kNumThreadsLoad) * (kNVecIn / kNVecSMem);         // Column in shared memory
    int r_s = threadIdx.x / kNumThreadsLoad;                             // Row in shared memory
    const size_t c_g = block_idx_x * kTileDim + c_s * kNVecSMem;         // Column in global memory
    size_t r_g = block_idx_y * kTileDim + r_s;                           // Row in global memory
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;  // Stride in global memory
    const size_t num_ele = (c_g < row_length ? min(static_cast<size_t>(kNVecIn), row_length - c_g)
                                             : 0);          // For not aligned case
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
      // Step 1.3: Update input address, row index of shared memory, (and row index of global memory
      // for not aligned case)
      input_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }

  __syncthreads();

  const int kNumThreadsReduce = kScaleBlockDim / kNVecOut;
  const float global_encode_scale =
      kIsE8Scaling ? 1.0f : ComputeGlobalEncodeScaleFP4(global_amax[0]);
  const float global_decode_scale = 1.0 / global_encode_scale;

  // Step 2: Cast and store to output_c
  if constexpr (kReturnIdentity) {
    constexpr int r_stride =
        kThreadsPerBlock / kNumThreadsStore;  // stride in rows of shared memory
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s =
        (threadIdx.x % kNumThreadsStore) * (kNVecOut / kNVecSMem);       // Column in shared memory
    int r_s = threadIdx.x / kNumThreadsStore;                            // Row in shared memory
    const size_t c_g = block_idx_x * kTileDim + c_s * kNVecSMem;         // Column in global memory
    size_t r_g = block_idx_y * kTileDim + r_s;                           // Row in global memory
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;  // Stride in global memory
    const size_t num_ele =
        (c_g < row_length ? min(static_cast<size_t>(kNVecOut / kNFP4PerContainer),
                                (row_length - c_g) / kNFP4PerContainer)
                          : 0);  // For not aligned case
    OType* output_g =
        &output_c[(r_g * row_length + c_g) / kNFP4PerContainer];  // Output address in global memory
    // Each kNumThreadsStore threads form a warp process one row, we need to find the lane id of
    // the first thread to do the reduction.
    const unsigned src_lane =
        (threadIdx.x % kThreadsPerWarp) / kNumThreadsReduce * kNumThreadsReduce;
    // This mask represents which threads should do the reduction together.
    const unsigned mask = ((1 << kNumThreadsReduce) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsReduce) == 0;
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
      if constexpr (kIsE8Scaling) {
#pragma unroll
        for (int delta = kNumThreadsReduce / 2; delta > 0; delta /= 2) {
          const float other_amax = __shfl_down_sync(mask, amax, delta);
          __builtin_assume(amax >= 0);
          __builtin_assume(other_amax >= 0);
          amax = fmaxf(amax, other_amax);
        }
        amax = __shfl_sync(mask, amax, src_lane);
      }
      // doing shuffle sync for 2D block scaling (not applicable for E8 scaling)
      if constexpr (kIs2DBlockScaling) {
        // first amax shuffle sync in warp, then reduce in smem
        // T0 T8 T16 T24 should do amax reduction together
        constexpr int kNumRowsPerIter = kThreadsPerBlock / kNumThreadsStore;  // 32
        int warp_idx = threadIdx.x / kThreadsPerWarp;                         // 0 ~ 7
        int tid_in_warp_x = threadIdx.x % kNumThreadsStore;
        int tid_in_warp_y = (threadIdx.x / kNumThreadsStore) % kNumRowsPerWarp;
        CType amax_warp_reduced = groupMax<kNumRowsPerWarp, kNumThreadsStore>(
            amax, WARP_REDUCE_AMAX_GROUP_MASKS[tid_in_warp_x]);
        // now T0 ~ T8 in each warp has the reduced amax values
        int data_row_idx = iter * kNumRowsPerIter + warp_idx * kNumRowsPerWarp + tid_in_warp_y;
        if (tid_in_warp_y == 0) {
          amax_smem_red[data_row_idx / kFP4BlockScalingSize][tid_in_warp_x]
                       [warp_idx % k2DBlockAmaxReduceDim] = amax_warp_reduced;
        }
        __syncthreads();

        if (data_row_idx % kFP4BlockScalingSize == 0) {
          CType amax_2d = 0.0;
          for (int i = 0; i < k2DBlockAmaxReduceDim; i++) {
            amax_2d = fmaxf(amax_2d,
                            amax_smem_red[data_row_idx / kFP4BlockScalingSize][tid_in_warp_x][i]);
          }
          amax_smem[data_row_idx / kFP4BlockScalingSize][tid_in_warp_x] = amax_2d;
        }
        __syncthreads();
        // every thread now knows 2D amax
        amax = amax_smem[data_row_idx / kFP4BlockScalingSize][tid_in_warp_x];
      }
      // Step 2.4: Compute scale
      ScaleType scale_inv = ComputeDecodeScaleFP4<ScaleType>(amax, global_encode_scale);
      float encode_scale = ComputeEncodeScaleFP4<ScaleType>(scale_inv, global_decode_scale);
      // Step 2.5: Write scale_inv
      bool write_scale_inv = is_src_lane;
      if constexpr (!kAligned) {
        write_scale_inv &= (r_g < num_rows);
        write_scale_inv &= (c_g < row_length);
      }
      if (write_scale_inv) {
        size_t row_idx = block_idx_y * kTileDim + r_s;
        size_t col_idx = block_idx_x * (kNumThreadsStore / kNumThreadsReduce) +
                         (threadIdx.x % kNumThreadsStore) / kNumThreadsReduce;
        if constexpr (kSwizzledScale) {
          size_t offset = scale_factor_swizzled_offset<ScaleType>(
              row_idx, col_idx, DIVUP(row_length, kScaleBlockDim));
          tile_scales_inv_c[offset] = scale_inv;
        } else {
          tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
        }
      }
      // Step 2.6: Quantize
      OVec output_vec;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; i += 2) {
        // Pack two elements into __nv_bfloat162
        float2 f2_a;
        float2 f2_b;
        f2_a.x = ComputeOutputFP4<IType, ScaleType>(smem_vec[i].data.elt[0], encode_scale);
        f2_a.y = ComputeOutputFP4<IType, ScaleType>(smem_vec[i].data.elt[1], encode_scale);
        f2_b.x = ComputeOutputFP4<IType, ScaleType>(smem_vec[i + 1].data.elt[0], encode_scale);
        f2_b.y = ComputeOutputFP4<IType, ScaleType>(smem_vec[i + 1].data.elt[1], encode_scale);
        const uint32_t rbits = kApplyStochasticRounding ? get_rbits(rng, random_uint4, rnd_idx) : 0;
        // Convert to __nv_fp4x4_e2m1
        __nv_fp4x4_e2m1 out_4x = cvt_fp32_to_fp4_4x<kApplyStochasticRounding>(f2_a, f2_b, rbits);

        output_vec.data.elt[i] = reinterpret_cast<__nv_fp4x2_storage_t*>(&out_4x)[0];
        output_vec.data.elt[i + 1] = reinterpret_cast<__nv_fp4x2_storage_t*>(&out_4x)[1];
      }
      // Step 2.7: Store output_c
      if constexpr (kAligned) {
        output_vec.store_to(output_g);
      } else {
        if (r_g < num_rows) {
          output_vec.store_to_elts(output_g, 0, num_ele);
        }
      }
      // Step 2.8: Update output address, row index of shared memory (and row index of global memory
      // for not aligned case)
      output_g += stride_g / kNFP4PerContainer;
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
    size_t r_g = block_idx_x * kTileDim + c_s * kNVecSMem;        // Row in global memory
    const size_t c_g = block_idx_y * kTileDim + r_s;              // Column in global memory
    const size_t stride_g =
        static_cast<size_t>(c_stride) * kNVecSMem * num_rows;  // Stride in global memory
    const size_t num_ele = (c_g < num_rows ? min(static_cast<size_t>(kNVecOut / kNFP4PerContainer),
                                                 (num_rows - c_g) / kNFP4PerContainer)
                                           : 0);  // For not aligned case
    OType* output_g =
        &output_t[(r_g * num_rows + c_g) / kNFP4PerContainer];  // Output address in global memory
    // Each kNumThreadsStore threads form a warp process one row, we need to find the lane id of
    // the first thread to do the reduction.
    const unsigned src_lane =
        (threadIdx.x % kThreadsPerWarp) / kNumThreadsReduce * kNumThreadsReduce;
    // This mask represents which threads should do the reduction together.
    const unsigned mask = ((1 << kNumThreadsReduce) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsReduce) == 0;
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
        if constexpr (kIs2DBlockScaling) {
          // TODO(zhongbo): 2D block scaling, directly read from amax_smem
          int warp_idx = threadIdx.x / kThreadsPerWarp;  // 0 ~ 7
          constexpr int kNumColsPerWarp =
              kThreadsPerWarp / kNumThreadsStore * kNVecSMem;  // 8 elements
          constexpr int kNumWarpsPerBlock =
              kThreadsPerBlock / kThreadsPerWarp;  // 8 warps per block
          constexpr int kNumColsPerIter = kNumColsPerWarp * kNumWarpsPerBlock;
          int tid_in_warp_x = (threadIdx.x / kNumThreadsStore) % kNumColsPerWarp;
          int tid_in_warp_y = (threadIdx.x % kThreadsPerWarp) % kNumThreadsStore;
          int data_col_idx = iter * kNumColsPerIter + warp_idx * kNumColsPerWarp + tid_in_warp_x;
          amax = amax_smem[tid_in_warp_y][data_col_idx / kFP4BlockScalingSize];
        } else {
#pragma unroll
          for (int i = 0; i < kNVecOut; ++i) {
            amax = fmaxf(amax, fabsf(smem_vec[i].data.elt[smem_idx]));
          }
        }
        // Step 3.3: Reduce amax
        if constexpr (kIsE8Scaling) {
#pragma unroll
          for (int delta = kNumThreadsReduce / 2; delta > 0; delta /= 2) {
            const float other_amax = __shfl_down_sync(mask, amax, delta);
            __builtin_assume(amax >= 0);
            __builtin_assume(other_amax >= 0);
            amax = fmaxf(amax, other_amax);
          }
          amax = __shfl_sync(mask, amax, src_lane);
        }
        // Step 3.4: Compute scale
        ScaleType scale_inv = ComputeDecodeScaleFP4<ScaleType>(amax, global_encode_scale);
        float encode_scale = ComputeEncodeScaleFP4<ScaleType>(scale_inv, global_decode_scale);
        // Step 3.5: Write scale_inv_t
        bool write_scale_inv = is_src_lane;
        if constexpr (!kAligned) {
          write_scale_inv &= (r_g + smem_idx < row_length);
          write_scale_inv &= (c_g < num_rows);
        }
        if (write_scale_inv) {
          size_t row_idx = block_idx_x * kTileDim + c_s * kNVecSMem + smem_idx;
          size_t col_idx = (block_idx_y * (kNumThreadsStore / kNumThreadsReduce) +
                            (threadIdx.x % kNumThreadsStore) / kNumThreadsReduce);
          if constexpr (kSwizzledScale) {
            size_t offset = scale_factor_swizzled_offset<ScaleType>(
                row_idx, col_idx, DIVUP(num_rows, kScaleBlockDim));
            tile_scales_inv_t[offset] = scale_inv;
          } else {
            tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
          }
        }
        // Step 3.6: Quantize
        OVec output_vec;
#pragma unroll
        for (int i = 0; i < kNVecOut / kNFP4PerContainer; i += 2) {
          // Pack two elements into __nv_bfloat162
          float2 f2_a;
          float2 f2_b;
          f2_a.x =
              ComputeOutputFP4<IType, ScaleType>(smem_vec[2 * i].data.elt[smem_idx], encode_scale);
          f2_a.y = ComputeOutputFP4<IType, ScaleType>(smem_vec[2 * i + 1].data.elt[smem_idx],
                                                      encode_scale);
          f2_b.x = ComputeOutputFP4<IType, ScaleType>(smem_vec[2 * (i + 1)].data.elt[smem_idx],
                                                      encode_scale);
          f2_b.y = ComputeOutputFP4<IType, ScaleType>(smem_vec[2 * (i + 1) + 1].data.elt[smem_idx],
                                                      encode_scale);
          const uint32_t rbits =
              kApplyStochasticRounding ? get_rbits(rng, random_uint4, rnd_idx) : 0;
          // Convert to __nv_fp4x4_e2m1
          __nv_fp4x4_e2m1 out_4x = cvt_fp32_to_fp4_4x<kApplyStochasticRounding>(f2_a, f2_b, rbits);

          output_vec.data.elt[i] = reinterpret_cast<__nv_fp4x2_storage_t*>(&out_4x)[0];
          output_vec.data.elt[i + 1] = reinterpret_cast<__nv_fp4x2_storage_t*>(&out_4x)[1];
        }
        // Step 3.7: Store output_t
        if constexpr (kAligned) {
          output_vec.store_to(output_g + smem_idx * num_rows / kNFP4PerContainer);
        } else {
          if (r_g + smem_idx < row_length) {
            output_vec.store_to_elts(output_g + smem_idx * num_rows / kNFP4PerContainer, 0,
                                     num_ele);
          }
        }
      }
      // Step 3.8: Update output address, column index of shared memory (and row index of global
      // memory for not aligned case)
      output_g += stride_g / kNFP4PerContainer;
      c_s += c_stride;
      if constexpr (!kAligned) {
        r_g += c_stride * kNVecSMem;
      }
    }
  }
}

}  // namespace
}  // namespace quantize_transpose_nvfp4
#endif  // CUDA_VERSION >= 12080

namespace detail {

void quantize_transpose_vector_blockwise_fp4(
    const SimpleTensor& input, const SimpleTensor& global_amax, SimpleTensor& scale_inv,
    SimpleTensor& scale_inv_t, SimpleTensor& output, SimpleTensor& output_t, const float epsilon,
    const bool return_identity, const bool return_transpose, const bool pow2_scale,
    const bool swizzled_scale, const bool use_stochastic_rounding,
    const NVTETensor rng_state_tensor, const bool use_2d_quantization,
    const SimpleTensor& noop_tensor, cudaStream_t stream) {
  NVTE_API_CALL(quantize_transpose_vector_blockwise_fp4);
#if CUDA_VERSION >= 12080

  // pow 2 scale is for MXFP4 since it's using E8M0 scaling
  // raise error if pow2_scale is true
  NVTE_CHECK(!pow2_scale, "No support for pow2_scale for MXFP4 for now");

  if (!return_identity && !return_transpose) {
    return;
  }

  if (use_2d_quantization && !return_identity) {
    return;
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

  size_t scale_stride_x = 0;
  size_t scale_stride_y = 0;

  if (return_identity) {
    scale_stride_x = 1;
    scale_stride_y = scale_inv.shape[1];
  }

  size_t scale_t_stride_x = 0;
  size_t scale_t_stride_y = 0;

  if (return_transpose) {
    scale_t_stride_x = 1;
    scale_t_stride_y = scale_inv_t.shape[1];
  }

  using namespace transformer_engine::quantize_transpose_nvfp4;

  const size_t num_blocks_x = DIVUP(row_length, static_cast<size_t>(kTileDim));
  const size_t num_blocks_y = DIVUP(num_rows, static_cast<size_t>(kTileDim));

  // noop tensor for cuda graph
  const float* noop_ptr = reinterpret_cast<const float*>(noop_tensor.dptr);

  const size_t* rng_state = nullptr;
  if (rng_state_tensor != nullptr) {
    Tensor& rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
    NVTE_CHECK(rng_state_te_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_te_tensor.data.shape == std::vector<size_t>{2},
               "Shape of the RNG state should be [2], but got ", rng_state_te_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t*>(rng_state_te_tensor.data.dptr);
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype, InputType,

      TRANSFORMER_ENGINE_TYPE_SWITCH_FP4x2_ONLY(
          output.dtype, 2, OutputType,

          dim3 grid(num_blocks_x, num_blocks_y, 1);

          using ScaleType = fp8e4m3; constexpr int kScaleBlockDim = 16;
          constexpr bool kPow2Scale = false;

          const bool full_tile = row_length % kTileDim == 0 && num_rows % kTileDim == 0;

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_identity, kReturnIdentity,

              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  return_transpose, kReturnTranspose,

                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      full_tile, kAligned,

                      TRANSFORMER_ENGINE_SWITCH_CONDITION(
                          swizzled_scale, kSwizzledScale,

                          TRANSFORMER_ENGINE_SWITCH_CONDITION(
                              use_stochastic_rounding, kApplyStochasticRounding,

                              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                                  use_2d_quantization, kIs2DBlockScaling,

                                  size_t smem_bytes = kSMemSize * sizeof(InputType);
                                  auto kernel = block_scaled_1d_cast_transpose_kernel<
                                      kReturnIdentity, kReturnTranspose, kPow2Scale, kAligned,
                                      float, InputType, OutputType, ScaleType, kSwizzledScale,
                                      kApplyStochasticRounding, kIs2DBlockScaling>;
                                  if (smem_bytes >= 48 * 1024) {
                                    cudaError_t err = cudaFuncSetAttribute(
                                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_bytes);
                                    NVTE_CHECK(err == cudaSuccess,
                                               "Failed to set dynamic shared memory size.");
                                  } kernel<<<grid, kThreadsPerBlock, smem_bytes,
                                             stream>>>(
                                      reinterpret_cast<const InputType*>(input.dptr),
                                      reinterpret_cast<const float*>(global_amax.dptr),
                                      reinterpret_cast<OutputType*>(output.dptr),
                                      reinterpret_cast<OutputType*>(output_t.dptr),
                                      reinterpret_cast<ScaleType*>(scale_inv.dptr),
                                      reinterpret_cast<ScaleType*>(scale_inv_t.dptr), row_length,
                                      num_rows, scale_stride_x, scale_stride_y, scale_t_stride_x,
                                      scale_t_stride_y, kScaleBlockDim, epsilon, rng_state,
                                      noop_ptr);)  // kIs2DBlockScaling
                              )                    // kApplyStochasticRounding
                          )                        // kSwizzledScale
                      )                            // kAligned
                  )                                // kReturnTranspose
              )                                    // kReturnIdentity
          )                                        // OutputType
      )                                            // InputType

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // CUDA_VERSION >= 12080
}

}  // namespace detail
}  // namespace transformer_engine
