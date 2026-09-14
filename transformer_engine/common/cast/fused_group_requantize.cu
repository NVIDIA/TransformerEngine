/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_group_requantize.cu
 *  \brief Fused grouped MXFP8 requantization: rowwise wire tensor -> GEMM-ready.
 */

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <limits>
#include <type_traits>

#include "../common.h"
#include "../util/ptx_arch_spec.cuh"
#include "../utils.cuh"
#include "mxfp8/swizzle.cuh"

namespace transformer_engine {
namespace requantize {
namespace {

constexpr int MXFP8_SCALE_DIM = 32;
constexpr int kTileRows = MXFP8_SCALE_DIM;
constexpr int kTileCols = 128;
constexpr int kElementsPerLoad = 16;
constexpr int kLoadsPerRow = kTileCols / kElementsPerLoad;
constexpr int kThreads = 128;
constexpr int kRowsPerGatherIteration = kThreads / kLoadsPerRow;
constexpr int kGatherIterations = kTileRows / kRowsPerGatherIteration;

static_assert(kRowsPerGatherIteration == 16);
static_assert(kGatherIterations == 2);
static_assert(kThreads == kTileCols);

__device__ __forceinline__ uint16_t e8m0_to_bf16_bits(const e8m0_t biased_exp) {
  // E8M0 encodes the exponent bits directly. Codes 0 and 255 need explicit handling because
  // 2^-127 is BF16-subnormal and 255 represents NaN.
  if (biased_exp == 255) return 0x7fff;
  if (biased_exp == 0) return 0x0040;
  return static_cast<uint16_t>(biased_exp) << 7;
}

template <typename IType>
__device__ __forceinline__ ptx::bf16x2 dequantize_mxfp8_2x(const ptx::FPx2<IType> &values,
                                                           const e8m0_t scale_code) {
  ptx::bf16x2 result;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#if (defined CUDA_VERSION) && (CUDA_VERSION >= 13020)
  // PTX ISA 9.2 can apply two packed E8M0 scaling factors while converting FP8x2 directly
  // to BF16x2. Arch-specific Blackwell targets (sm_100a) include these family features.
  constexpr bool kHasScaledFP8ToBF16 = ARCH_BLACKWELL_FAMILY;
  if constexpr (kHasScaledFP8ToBF16) {
    const uint16_t scale_x2 =
        static_cast<uint16_t>(scale_code) | (static_cast<uint16_t>(scale_code) << 8);
    if constexpr (std::is_same_v<IType, fp8e4m3>) {
      asm volatile("cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, %1, %2;"
                   : "=r"(reinterpret_cast<uint32_t &>(result))
                   : "h"(reinterpret_cast<const uint16_t &>(values)), "h"(scale_x2));
    } else {
      static_assert(std::is_same_v<IType, fp8e5m2>);
      asm volatile("cvt.rn.scaled::n2::ue8m0.bf16x2.e5m2x2 %0, %1, %2;"
                   : "=r"(reinterpret_cast<uint32_t &>(result))
                   : "h"(reinterpret_cast<const uint16_t &>(values)), "h"(scale_x2));
    }
    return result;
  }
#endif

  // CUDA 12.8-compatible fallback. Every E4M3/E5M2 value is exactly representable in
  // FP16, so the FP16 bridge does not lose information before rounding to BF16.
  const uint16_t scale_bits = e8m0_to_bf16_bits(scale_code);
  const uint32_t scale_x2 =
      static_cast<uint32_t>(scale_bits) | (static_cast<uint32_t>(scale_bits) << 16);
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
__device__ __forceinline__ void store_colwise_4x_to_shared(OType *const output,
                                                           const int stride_elements,
                                                           const uint32_t values) {
  static_assert(sizeof(OType) == 1);
  const uint32_t output_ptr = __cvta_generic_to_shared(output);
  const uint32_t stride_bytes = stride_elements * sizeof(OType);
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
__device__ __forceinline__ void store_colwise_2x_to_shared(OType *const output,
                                                           const int stride_elements,
                                                           const uint32_t values) {
  static_assert(sizeof(OType) == 1);
  const uint32_t output_ptr = __cvta_generic_to_shared(output);
  const uint32_t stride_bytes = stride_elements * sizeof(OType);
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

// ---------------------------------------------------------------------------
// Fused grouped requantization.
//
// Rowwise MXFP8 grouped input -> columnwise MXFP8 output with GEMM-swizzled
// scales for BOTH directions, plus an optional BF16 dequantized copy. Replaces
// the group_dequantize -> group_quantize(columnwise) -> grouped_swizzle(rowwise
// scales) chain with one kernel; the dequantized values only exist in shared
// memory unless dequantized_out is requested.
//
// Contract (asserted host-side where possible): the hidden dim and every
// group's row count are multiples of 128, so data tiles and swizzle tiles
// never straddle a group boundary and the scale layouts carry no padding.
// Group boundaries arrive as the grouped tensor's cached element-based
// tensor_offsets (offsets[g] = row offset x cols); they live on the device
// (host reads would break CUDA-graph capture), so per-group divisibility is
// the caller's contract. Rows at or past offsets[num_groups] (capacity-mode /
// paged-stash tail) are left untouched, matching the unfused chain.

template <typename IType, typename OType, bool kUseFastMath, bool kReturnDequantized>
__global__ void __launch_bounds__(kThreads)
    fused_group_requantize_kernel(const __grid_constant__ CUtensorMap output_tensor_map,
                                  const IType *const input, const e8m0_t *const input_scale_inv,
                                  e8m0_t *const rowwise_scale_inv_swizzled,
                                  e8m0_t *const colwise_scale_inv, bf16 *const dequantized_out,
                                  const int64_t *const tensor_offsets, const int num_groups,
                                  const int num_cols, const int input_scale_stride) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;
  using TCompute = std::conditional_t<kUseFastMath, bf16, float>;

  constexpr int kPaddingPerVector = sizeof(float) / sizeof(TCompute);
  constexpr int kDequantizedStride = kTileCols + kLoadsPerRow * kPaddingPerVector;

  const int tid = threadIdx.x;
  const int row_base = blockIdx.y * kTileRows;
  const int col_base = blockIdx.x * kTileCols;

  // Tail guard: with capacity-mode / paged-stash tensors the allocated rows
  // exceed the live rows (tensor_offsets[num_groups] elements). The unfused
  // chain never touches those rows; skip them entirely, or their out-of-range
  // tile indices would alias INTO the last group's columnwise scales. The
  // branch is CTA-uniform: live rows are 128-aligned like every group, so the
  // boundary cannot split a 32-row tile.
  const int64_t row_element_base = static_cast<int64_t>(row_base) * num_cols;
  if (row_element_base >= tensor_offsets[num_groups]) {
    return;
  }

  __shared__ alignas(16) TCompute dequantized[kTileRows][kDequantizedStride];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) OType quantized[kTileRows][kTileCols];
  __shared__ int group_info[2];  // {group_start_row, group_num_rows}

  // The columnwise scale layout is per-group; find the group owning this tile.
  // 128-aligned group sizes mean a 32-row tile never straddles a boundary.
  // upper_bound minus one lands on the non-empty owner even when zero-sized
  // groups share an offset. Offsets are in elements (row offset x num_cols),
  // exactly the grouped tensor's cached tensor_offsets.
  if (tid == 0) {
    int lo = 0;
    int hi = num_groups - 1;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (row_element_base < tensor_offsets[mid + 1]) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    group_info[0] = static_cast<int>(tensor_offsets[lo] / num_cols);
    group_info[1] = static_cast<int>((tensor_offsets[lo + 1] - tensor_offsets[lo]) / num_cols);
  }

  const int rowwise_scale_tiles_x = num_cols / kTileCols;

  // Phase 1: each of the 128 threads loads one contiguous 16-byte vector in
  // each of two iterations; together the CTA dequantizes all 32x128 values
  // into shared memory.
  const int lane = tid % THREADS_PER_WARP;
#pragma unroll
  for (int gather_iteration = 0; gather_iteration < kGatherIterations; ++gather_iteration) {
    const int local_chunk = tid % kLoadsPerRow;
    const int local_row = tid / kLoadsPerRow + gather_iteration * kRowsPerGatherIteration;
    const int local_col = local_chunk * kElementsPerLoad;
    const int row = row_base + local_row;
    const int col = col_base + local_col;

    // Adjacent 16-byte chunks share one rowwise MXFP8 scale. The even chunk
    // loads the E8M0 code, re-emits it at its GEMM-swizzled address (dense
    // indexing equals the per-group layout because group row counts are
    // multiples of the 128-row swizzle tile), and broadcasts it to its
    // partner.
    int scale_code = 0;
    if ((local_chunk % 2) == 0) {
      const int scale_col = col / MXFP8_SCALE_DIM;
      const size_t input_scale_idx = static_cast<size_t>(row) * input_scale_stride + scale_col;
      scale_code = static_cast<int>(input_scale_inv[input_scale_idx]);
      rowwise_scale_inv_swizzled[gemm_swizzled_scale_idx(row, scale_col, rowwise_scale_tiles_x)] =
          static_cast<e8m0_t>(scale_code);
    }
    scale_code = __shfl_sync(0xffffffff, scale_code, lane & ~1);

    const int shared_col = local_col + (local_col / kElementsPerLoad) * kPaddingPerVector;
    Vec<IType, kElementsPerLoad> input_vec;
    input_vec.load_from(input + static_cast<size_t>(row) * num_cols + col);

    constexpr int kDequantizedVecSize = kElementsPerLoad / 2;
    [[maybe_unused]] Vec<bf16, kDequantizedVecSize> dequantized_vec[2];  // kReturnDequantized only
    if constexpr (kUseFastMath) {
#pragma unroll
      for (int i = 0; i < kElementsPerLoad; i += 2) {
        const ptx::FPx2<IType> values = {input_vec.data.elt[i], input_vec.data.elt[i + 1]};
        const ptx::bf16x2 result = dequantize_mxfp8_2x(values, static_cast<e8m0_t>(scale_code));
        *reinterpret_cast<ptx::bf16x2 *>(&dequantized[local_row][shared_col + i]) = result;
        if constexpr (kReturnDequantized) {
          *reinterpret_cast<ptx::bf16x2 *>(
              &dequantized_vec[i / kDequantizedVecSize].data.elt[i % kDequantizedVecSize]) = result;
        }
      }
    } else {
      const float scale = ptx::exp2f(static_cast<e8m0_t>(scale_code));
#pragma unroll
      for (int i = 0; i < kElementsPerLoad; ++i) {
        const float value = scale * static_cast<float>(input_vec.data.elt[i]);
        dequantized[local_row][shared_col + i] = value;
        if constexpr (kReturnDequantized) {
          dequantized_vec[i / kDequantizedVecSize].data.elt[i % kDequantizedVecSize] =
              static_cast<bf16>(value);
        }
      }
    }
    if constexpr (kReturnDequantized) {
      bf16 *const dequantized_out_ptr = dequantized_out + static_cast<size_t>(row) * num_cols + col;
      dequantized_vec[0].store_to(dequantized_out_ptr);
      dequantized_vec[1].store_to(dequantized_out_ptr + kDequantizedVecSize);
    }
  }

  __syncthreads();

  // Phase 2: one thread owns one column and quantizes its 32 values as one
  // MXFP8 block, emitting the scale at its per-group GEMM-swizzled address.
  {
    const int group_start = group_info[0];
    const int group_rows = group_info[1];

    const int dequantized_col = tid + (tid / kElementsPerLoad) * kPaddingPerVector;
    float thread_amax = 0.0f;

    ptx::bf16x4 bf16_values[kTileRows / 4];
    if constexpr (kUseFastMath) {
      ptx::bf16x2 thread_amax_x2 = {static_cast<bf16>(0.0f), static_cast<bf16>(0.0f)};
#pragma unroll
      for (int row = 0; row < kTileRows; row += 4) {
        const ptx::bf16x4 values = {
            dequantized[row][dequantized_col], dequantized[row + 1][dequantized_col],
            dequantized[row + 2][dequantized_col], dequantized[row + 3][dequantized_col]};
        bf16_values[row / 4] = values;
        const ptx::bf16x2 values01 = {values.x1, values.x2};
        const ptx::bf16x2 values23 = {values.x3, values.x4};
        ptx::abs_max_2x(thread_amax_x2, thread_amax_x2, values01);
        ptx::abs_max_2x(thread_amax_x2, thread_amax_x2, values23);
      }
      thread_amax = static_cast<float>(ptx::get_amax(thread_amax_x2.x, thread_amax_x2.y));
    } else {
#pragma unroll
      for (int row = 0; row < kTileRows; ++row) {
        thread_amax = fmaxf(thread_amax, fabsf(dequantized[row][dequantized_col]));
      }
    }

    const e8m0_t biased_exponent =
        ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);

    // Per-group columnwise scale addressing. Mirrors group_quantize's
    // WITH_GEMM_SWIZZLED_SCALES emission (group_quantize_mxfp8.cuh) and the
    // grouped-GEMM consumer's padded cumsum: under the /128 contract the
    // per-group base is exactly group_start/32 * num_cols.
    const size_t colwise_scale_base = static_cast<size_t>(group_start) / MXFP8_SCALE_DIM * num_cols;
    const int local_scale_row = (row_base - group_start) / kTileRows;
    const int scale_col = col_base + tid;
    // One swizzle tile spans GEMM_SWIZZLED_SCALE_TILE_DIM_X = 4 scale rows =
    // 128 data rows, so this matches the producer's DIVUP(rows, 128).
    const int colwise_scale_tiles_x = group_rows / 128;
    colwise_scale_inv[colwise_scale_base +
                      gemm_swizzled_scale_idx(scale_col, local_scale_row, colwise_scale_tiles_x)] =
        biased_exponent;

    if constexpr (kUseFastMath) {
      const bf16 quant_multiplier = ptx::exp2f_rcp<bf16>(biased_exponent);
      const ptx::bf16x2 quant_multiplier_x2 = {quant_multiplier, quant_multiplier};
#pragma unroll
      for (int row = 0; row < kTileRows; row += 4) {
        uint32_t result_data = 0;
        auto &result = *reinterpret_cast<ptx::fp8e4m3x4 *>(&result_data);
        ptx::mul_cvt_4x(result, bf16_values[row / 4], quant_multiplier_x2);
        store_colwise_4x_to_shared(&quantized[row][tid], kTileCols, result_data);
      }
    } else {
      const float quant_multiplier = ptx::exp2f_rcp<float>(biased_exponent);
      const ptx::floatx2 quant_multiplier_x2 = {quant_multiplier, quant_multiplier};
#pragma unroll
      for (int row = 0; row < kTileRows; row += 2) {
        const ptx::floatx2 values = {dequantized[row][dequantized_col],
                                     dequantized[row + 1][dequantized_col]};
        uint32_t result_data = 0;
        auto &result = *reinterpret_cast<ptx::fp8e4m3x2 *>(&result_data);
        ptx::mul_cvt_2x(result, values, quant_multiplier_x2);
        store_colwise_2x_to_shared(&quantized[row][tid], kTileCols, result_data);
      }
    }
  }

  // Make the complete 32x128 FP8 tile visible to the TMA engine and store it.
  // The /128 contract makes every tile full, so no bounds handling is needed.
  ptx::fence_proxy_async_shared_cta();
  __syncthreads();
  if (tid == 0) {
    ptx::cp_async_bulk_tensor_2d_shared_to_global(
        reinterpret_cast<const uint64_t *>(&output_tensor_map), col_base, row_base,
        reinterpret_cast<uint64_t *>(quantized));
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read<0>();
  }
  __syncthreads();
#else
  NVTE_DEVICE_THREAD0_ERROR("Fused grouped requantization requires Blackwell (SM100+) hardware.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <typename IType, bool kUseFastMath, bool kReturnDequantized>
void launch_fused_group_requantize(const Tensor &input, Tensor *output,
                                   const Tensor &tensor_offsets, Tensor *dequantized,
                                   const int num_groups, const int num_rows, const int num_cols,
                                   const int input_scale_stride,
                                   const CUtensorMap &output_tensor_map, cudaStream_t stream) {
  using OType = fp8e4m3;
  const dim3 grid(num_cols / kTileCols, num_rows / kTileRows);
  const dim3 block(kThreads);

  bf16 *dequantized_ptr = nullptr;
  if constexpr (kReturnDequantized) {
    dequantized_ptr = reinterpret_cast<bf16 *>(dequantized->data.dptr);
  }

  fused_group_requantize_kernel<IType, OType, kUseFastMath, kReturnDequantized>
      <<<grid, block, 0, stream>>>(
          output_tensor_map, reinterpret_cast<const IType *>(input.data.dptr),
          reinterpret_cast<const e8m0_t *>(input.scale_inv.dptr),
          reinterpret_cast<e8m0_t *>(output->scale_inv.dptr),
          reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr), dequantized_ptr,
          reinterpret_cast<const int64_t *>(tensor_offsets.data.dptr), num_groups, num_cols,
          input_scale_stride);
}

void fused_group_requantize(const Tensor &input, Tensor *output, const Tensor &tensor_offsets,
                            Tensor *dequantized, const QuantizationConfig *quant_config,
                            cudaStream_t stream) {
  checkCuDriverContext(stream);

  NVTE_CHECK(is_supported_by_CC_100(),
             "Fused grouped requantization requires Blackwell (SM100+) hardware.");
  NVTE_CHECK(input.scaling_mode == NVTE_MXFP8_1D_SCALING, "Input must use MXFP8 1D scaling.");
  NVTE_CHECK(output->scaling_mode == NVTE_MXFP8_1D_SCALING, "Output must use MXFP8 1D scaling.");
  NVTE_CHECK(input.has_data(), "Input must have rowwise MXFP8 data.");
  NVTE_CHECK(input.data.dptr != nullptr, "Input rowwise data must be allocated.");
  NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input rowwise data must have an FP8 type.");
  NVTE_CHECK(input.scale_inv.dptr != nullptr, "Input rowwise scaling tensor must be allocated.");
  NVTE_CHECK(input.scale_inv.dtype == DType::kFloat8E8M0,
             "Input rowwise scaling tensor must have E8M0 type.");
  NVTE_CHECK(!input.with_gemm_swizzled_scales,
             "Input rowwise scales must be unswizzled (compact); dequantization reads them "
             "row-indexed.");
  NVTE_CHECK(output->has_columnwise_data(), "Output must have columnwise MXFP8 data.");
  NVTE_CHECK(output->columnwise_data.dptr != nullptr, "Output columnwise data must be allocated.");
  NVTE_CHECK(output->columnwise_data.dtype == DType::kFloat8E4M3,
             "Output columnwise data must have E4M3 type.");
  NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
             "Output columnwise scaling tensor must be allocated.");
  NVTE_CHECK(output->columnwise_scale_inv.dtype == DType::kFloat8E8M0,
             "Output columnwise scaling tensor must have E8M0 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr,
             "Output rowwise scaling tensor (the swizzled copy of the input scales) must be "
             "allocated.");
  NVTE_CHECK(output->scale_inv.dtype == DType::kFloat8E8M0,
             "Output rowwise scaling tensor must have E8M0 type.");
  NVTE_CHECK(tensor_offsets.has_data(), "tensor_offsets must be allocated.");
  NVTE_CHECK(tensor_offsets.data.dptr != nullptr, "tensor_offsets data must be allocated.");
  NVTE_CHECK(tensor_offsets.data.dtype == DType::kInt64, "tensor_offsets must have Int64 type.");
  NVTE_CHECK(tensor_offsets.data.numel() >= 2, "tensor_offsets must hold num_groups + 1 entries.");

  const int num_groups = static_cast<int>(tensor_offsets.data.numel()) - 1;

  const auto [num_rows_size_t, num_cols_size_t] = input.flat_2d_dims();
  const auto [output_rows_size_t, output_cols_size_t] = output->flat_2d_dims();
  constexpr size_t kMaxInt = static_cast<size_t>(std::numeric_limits<int>::max());
  NVTE_CHECK(num_rows_size_t <= kMaxInt && num_cols_size_t <= kMaxInt,
             "Fused grouped requantization dimensions must fit in int32.");
  const int num_rows = static_cast<int>(num_rows_size_t);
  const int num_cols = static_cast<int>(num_cols_size_t);

  NVTE_CHECK(output_rows_size_t == num_rows_size_t && output_cols_size_t == num_cols_size_t,
             "Input and output shapes must match, but got (", num_rows_size_t, ", ",
             num_cols_size_t, ") and (", output_rows_size_t, ", ", output_cols_size_t, ").");
  // Each group's row count must be a multiple of 128 too, so that every group's
  // scales start on a swizzle-tile boundary. Those counts live on the device,
  // so that half is the caller's contract rather than an assertion.
  NVTE_CHECK(num_rows % 128 == 0 && num_cols % 128 == 0,
             "Fused grouped requantization requires dims that are multiples of 128, but got (",
             num_rows, ", ", num_cols, ").");
  NVTE_CHECK(num_rows / kTileRows <= 65535,
             "The number of rows is too large for the 2D CUDA launch grid.");

  // The input scale tensor may be exactly compact or carry TE's padded
  // allocation; both are row-indexed with this stride. Under the /128 contract
  // the padded shape degenerates to the compact one, so a flat (1-D) scale
  // tensor is also accepted.
  int input_scale_stride = num_cols / static_cast<int>(MXFP8_SCALE_DIM);
  if (input.scale_inv.shape.size() == 2) {
    NVTE_CHECK(input.scale_inv.shape[1] <= kMaxInt, "MXFP8 scale strides must fit in int32.");
    input_scale_stride = static_cast<int>(input.scale_inv.shape[1]);
    NVTE_CHECK(input_scale_stride >= num_cols / static_cast<int>(MXFP8_SCALE_DIM),
               "Input rowwise scale stride is smaller than the number of scale columns.");
  }
  const size_t num_scales = static_cast<size_t>(num_rows) * (num_cols / MXFP8_SCALE_DIM);
  NVTE_CHECK(input.scale_inv.numel() >= num_scales,
             "Input rowwise scale tensor is smaller than rows x cols / 32.");
  NVTE_CHECK(output->scale_inv.numel() >= num_scales,
             "Output rowwise scale tensor is smaller than rows x cols / 32.");
  NVTE_CHECK(output->columnwise_scale_inv.numel() >=
                 static_cast<size_t>(num_rows) / MXFP8_SCALE_DIM * num_cols,
             "Output columnwise scale tensor is smaller than rows / 32 x cols.");

  const bool return_dequantized =
      dequantized != nullptr && dequantized->has_data() && dequantized->data.dptr != nullptr;
  if (return_dequantized) {
    NVTE_CHECK(dequantized->data.dtype == DType::kBFloat16,
               "The dequantized output must have BF16 type.");
    NVTE_CHECK(
        dequantized->data.numel() == static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols),
        "The dequantized output must have rows x cols elements.");
    NVTE_CHECK(is_aligned_ptr(dequantized->data.dptr, 16),
               "The dequantized output pointer must be 16B aligned.");
  }

  NVTE_CHECK(is_aligned_ptr(input.data.dptr, 16), "Input data pointer must be 16B aligned.");
  NVTE_CHECK(is_aligned_ptr(output->columnwise_data.dptr, TMA_GMEM_ALIGNMENT),
             "Output data pointer must be 16B aligned.");

  // Both scale directions come out GEMM-swizzled; make the metadata say so for
  // every caller, not just the PyTorch integration.
  output->with_gemm_swizzled_scales = true;

  if (num_rows == 0) {
    return;
  }

  alignas(64) CUtensorMap output_tensor_map{};
  create_2D_tensor_map(output_tensor_map, output->columnwise_data, num_rows, num_cols, kTileRows,
                       kTileCols, num_cols, 0, typeToNumBits(output->columnwise_data.dtype));

  const bool use_fast_math = quant_config != nullptr && quant_config->use_fast_math;
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_fast_math, USE_FAST_MATH,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          input.data.dtype, IType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_dequantized, RETURN_DEQUANTIZED,
              launch_fused_group_requantize<IType, USE_FAST_MATH, RETURN_DEQUANTIZED>(
                  input, output, tensor_offsets, dequantized, num_groups, num_rows, num_cols,
                  input_scale_stride, output_tensor_map, stream););););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace
}  // namespace requantize
}  // namespace transformer_engine

void nvte_group_requantize(const NVTETensor input, NVTETensor output,
                           const NVTETensor tensor_offsets, NVTETensor dequantized,
                           const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_API_CALL(nvte_group_requantize);

  const Tensor *input_cu = convertNVTETensorCheck(input);
  Tensor *output_cu = convertNVTETensorCheck(output);
  const Tensor *tensor_offsets_cu = convertNVTETensorCheck(tensor_offsets);
  Tensor *dequantized_cu = dequantized != nullptr ? convertNVTETensor(dequantized) : nullptr;
  const auto *quant_config_cu = reinterpret_cast<const QuantizationConfig *>(quant_config);
  requantize::fused_group_requantize(*input_cu, output_cu, *tensor_offsets_cu, dequantized_cu,
                                     quant_config_cu, stream);
}
