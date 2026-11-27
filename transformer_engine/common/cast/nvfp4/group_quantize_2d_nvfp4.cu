/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#include "../../util/ptx_arch_spec.cuh"
#include "../core/grouped_layout.cuh"
#include "core_nvfp4.cuh"
#include "group_quantize_2d_nvfp4.h"
#include "swizzle.cuh"

namespace transformer_engine::dispatch::nvfp4 {
namespace {

__global__ void validate_grouped_2d_splits(const int64_t* offsets, const int64_t* rows,
                                           size_t count, size_t capacity, size_t cols,
                                           const float* noop) {
  if (noop && *noop == 1.0f) return;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  if (rows[i] < 0 || rows[i] % 128 != 0 || offsets[i] < 0 || offsets[i + 1] < offsets[i] ||
      offsets[i + 1] > capacity * cols || offsets[i + 1] - offsets[i] != rows[i] * cols ||
      (i == 0 && offsets[0] != 0)) {
    NVTE_DEVICE_ERROR(
        "Grouped NVFP4 2D requires contiguous, 128-aligned expert rows within capacity.");
  }
}

#if FP4_TYPE_SUPPORTED
template <typename ScaleType, bool ROWWISE, bool COLUMNWISE, bool SWIZZLED>
__global__ __launch_bounds__(128) void grouped_nvfp4_2d_quantize_transpose(
    const __grid_constant__ CUtensorMap input_map, uint8_t* row_data, uint8_t* col_data,
    ScaleType* row_scales, ScaleType* col_scales, const float* amax, const int64_t* offsets,
    const int64_t* rows, size_t count, size_t cols, const float* noop) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  if (noop && *noop == 1.0f) return;
  const size_t global_row = (static_cast<size_t>(blockIdx.x) / (cols / 128)) * 128;
  if (global_row * cols >= offsets[count]) return;
  const size_t expert = common::find_tensor_from_offsets(offsets, count, global_row * cols);
  const size_t start = offsets[expert] / cols;
  const size_t local_row = global_row - start;
  const size_t col = (blockIdx.x % (cols / 128)) * 128;
  const size_t m = rows[expert];
  const float encode = core::compute_global_encode_scaling_factor_FP4<ScaleType>(amax[expert]);
  const float decode = 1.0 / encode;
  const size_t scale_base = offsets[expert] / 16;
  const int tid = threadIdx.x;
  const bool leader = tid == 0;
  __shared__ alignas(128) bf16 input[2][32][128];
  __shared__ alignas(16) uint32_t packed[8][64];
  __shared__ alignas(16) ScaleType scales[8][8];
  __shared__ alignas(8) uint64_t barriers[4];
  initialize_barriers<4, 128>(barriers, leader);
  copy_2d_to_shared(input[0], &input_map, col, global_row, 32 * 128 * sizeof(bf16), &barriers[0],
                    leader);

  // A 128x128 chunk is consumed in four 32x128 stages. Prefetch into the other
  // input buffer while quantizing the current stage; no dynamic tensor maps or D2H.
#pragma unroll
  for (int stage = 0; stage < 4; ++stage) {
    const int buff = stage % 2;
    if (stage + 1 < 4) {
      copy_2d_to_shared(input[(stage + 1) % 2], &input_map, col, global_row + (stage + 1) * 32,
                        32 * 128 * sizeof(bf16), &barriers[stage + 1], leader);
    }
    ptx::mbarrier_wait_parity(&barriers[stage], 0);
    uint32_t col_words[4] = {};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      alignas(8) bf16 values[16];
      float a = 0.0f;
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        values[r] = input[buff][half * 16 + r][tid];
        a = fmaxf(a, fabsf(static_cast<float>(values[r])));
      }
#pragma unroll
      for (int delta = 8; delta > 0; delta /= 2) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, delta, 16));
      }
      const ScaleType scale = core::compute_decoding_scaling_factor<ScaleType>(a, encode);
      const float inverse =
          fminf(1.0f / (static_cast<float>(scale) * decode), detail::TypeExtrema<float>::max);
      const float2 inverse2{inverse, inverse};
      if (tid % 16 == 0) scales[stage * 2 + half][tid / 16] = scale;
        // With deterministic 2D scaling, both directions share the same FP4 values.
        // Quantize once using the single-tensor arithmetic, then transpose the nibbles.
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t bits = ptx::mul_cvt_bf16_to_fp4_4x<false>(
                                  *reinterpret_cast<const uint64_t*>(values + 4 * q), inverse2, 0)
                                  .__x;
        col_words[half * 2 + q / 2] |= bits << ((q % 2) * 16);
        if constexpr (ROWWISE) {
          // Pair adjacent columns into whole words before storing to shared memory.
          const uint32_t adjacent = __shfl_xor_sync(0xffffffff, bits, 1);
          if (tid % 2 == 0) packed[half * 4 + q][tid / 2] = bits | (adjacent << 16);
        }
      }
    }
    if constexpr (COLUMNWISE) {
      *reinterpret_cast<uint4*>(col_data + offsets[expert] / 2 + (col + tid) * (m / 2) +
                                (local_row + stage * 32) / 2) =
          make_uint4(col_words[0], col_words[1], col_words[2], col_words[3]);
    }
    __syncthreads();
    if constexpr (ROWWISE) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = (tid + i * 128) / 16;
        const int c = (tid % 16) * 8;
        const uint4 v = *reinterpret_cast<const uint4*>(&packed[r / 4][c / 2]);
        const int shift = (r % 4) * 4;
        const auto pair = [shift](uint32_t x) {
          return ((x >> shift) & 15u) | ((x >> (shift + 12)) & 240u);
        };
        const uint32_t out = pair(v.x) | (pair(v.y) << 8) | (pair(v.z) << 16) | (pair(v.w) << 24);
        *reinterpret_cast<uint32_t*>(row_data + (global_row + stage * 32 + r) * (cols / 2) +
                                     (col + c) / 2) = out;
      }
    }
    __syncthreads();
  }
  // Emit the replicated scale matrix in 8-byte vectors rather than scattered
  // byte stores per stage. Two 512-byte base blocks cover each swizzled tile.
  static_assert(sizeof(ScaleType) == 1);
  const auto* scale_bytes = reinterpret_cast<const uint8_t*>(scales);
  if constexpr (SWIZZLED) {
    const int r0 = (tid / 2) % 32 + (tid % 2) * 64;
    const int r1 = r0 + 32;
    const int c = (tid / 64) * 4;
    if constexpr (ROWWISE) {
      const uint32_t lo = *reinterpret_cast<const uint32_t*>(&scales[r0 / 16][c]);
      const uint32_t hi = *reinterpret_cast<const uint32_t*>(&scales[r1 / 16][c]);
      const size_t base = swizzle::gemm_swizzled_scale_idx(local_row, col / 16, cols / 16);
      reinterpret_cast<uint64_t*>(row_scales + scale_base + base)[tid] =
          static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }
    if constexpr (COLUMNWISE) {
      uint64_t bits = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        bits |= static_cast<uint64_t>(scale_bytes[(c + j) * 8 + r0 / 16]) << (8 * j);
        bits |= static_cast<uint64_t>(scale_bytes[(c + j) * 8 + r1 / 16]) << (32 + 8 * j);
      }
      const size_t base = swizzle::gemm_swizzled_scale_idx(col, local_row / 16, m / 16);
      reinterpret_cast<uint64_t*>(col_scales + scale_base + base)[tid] = bits;
    }
  } else {
    if constexpr (ROWWISE) {
      *reinterpret_cast<uint64_t*>(row_scales + scale_base + (local_row + tid) * (cols / 16) +
                                   col / 16) =
          *reinterpret_cast<const uint64_t*>(&scales[tid / 16][0]);
    }
    if constexpr (COLUMNWISE) {
      uint64_t bits = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        bits |= static_cast<uint64_t>(scale_bytes[j * 8 + tid / 16]) << (8 * j);
      }
      *reinterpret_cast<uint64_t*>(col_scales + scale_base + (col + tid) * (m / 16) +
                                   local_row / 16) = bits;
    }
  }
  destroy_barriers<4>(barriers, leader);
#else
  NVTE_DEVICE_ERROR("Grouped NVFP4 2D quantization requires Blackwell.");
#endif
}
#endif
}  // namespace

void group_quantize_2d(const GroupedTensor* input, const Tensor* noop, GroupedTensor* output,
                       const QuantizationConfig* config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped NVFP4 2D requires matrix inputs and outputs.");
  NVTE_CHECK(input->dtype() == DType::kBFloat16, "Grouped NVFP4 2D requires BF16 input.");
  NVTE_CHECK(input->num_tensors == output->num_tensors, "Grouped tensor counts must match.");
  NVTE_CHECK(output->all_same_last_dim() || output->all_same_shape(),
             "Grouped NVFP4 2D requires a common hidden dimension.");
  NVTE_CHECK(!config->stochastic_rounding && config->nvfp4_4over6_mode == kNVTENVFP44Over6Disabled,
             "Grouped NVFP4 2D does not support stochastic rounding or 4over6.");
  CheckNoopTensor(*noop, "grouped_2d_noop");
  const size_t capacity = input->logical_shape.data[0];
  const size_t cols = input->logical_shape.data[1];
  const size_t count = input->num_tensors;
  NVTE_CHECK(cols > 0 && cols % 128 == 0 && capacity % 128 == 0,
             "Grouped NVFP4 2D requires 128-aligned capacity and hidden dimension.");
  NVTE_CHECK((capacity == 0 || input->data.dptr) && input->data.numel() >= capacity * cols,
             "Invalid grouped NVFP4 input buffer.");
  NVTE_CHECK(output->logical_shape.data[0] == capacity && output->logical_shape.data[1] == cols,
             "Grouped NVFP4 2D input/output capacities must match.");
  if (count == 0) return;
  // Empty packed storage has a null pointer, but still has one amax per expert.
  const bool rowwise = output->amax.has_data();
  const bool columnwise = output->columnwise_amax.has_data();
  NVTE_CHECK(rowwise || columnwise, "Missing quantized output.");
  const auto validate_direction = [&](bool enabled, const SimpleTensor& data,
                                      const SimpleTensor& scales, const SimpleTensor& amax) {
    if (!enabled) return;
    NVTE_CHECK(data.dtype == DType::kFloat4E2M1 && (capacity == 0 || data.dptr) &&
                   data.numel() >= capacity * cols / 2,
               "Invalid grouped NVFP4 data buffer.");
    NVTE_CHECK((capacity == 0 || scales.dptr) && scales.numel() >= capacity * cols / 16,
               "Invalid grouped NVFP4 scale buffer.");
    NVTE_CHECK(amax.dptr && amax.dtype == DType::kFloat32 && amax.numel() >= count,
               "Grouped NVFP4 2D requires one FP32 amax per expert.");
  };
  validate_direction(rowwise, output->data, output->scale_inv, output->amax);
  validate_direction(columnwise, output->columnwise_data, output->columnwise_scale_inv,
                     output->columnwise_amax);
  const DType scale_type = rowwise ? output->scale_inv.dtype : output->columnwise_scale_inv.dtype;
  NVTE_CHECK(
      !rowwise || !columnwise || output->scale_inv.dtype == output->columnwise_scale_inv.dtype,
      "Grouped NVFP4 directions must use the same scale dtype.");
  NVTE_CHECK(output->tensor_offsets.dtype == DType::kInt64 &&
                 output->first_dims.dtype == DType::kInt64 && output->tensor_offsets.dptr &&
                 output->tensor_offsets.numel() >= count + 1 && output->first_dims.dptr &&
                 output->first_dims.numel() >= count,
             "Grouped NVFP4 2D requires int64 device splits and offsets.");
  const auto* offsets = static_cast<const int64_t*>(output->tensor_offsets.dptr);
  const auto* rows = static_cast<const int64_t*>(output->first_dims.dptr);
  const float* noop_ptr = static_cast<const float*>(noop->data.dptr);
  // Validate on device, including all-zero splits, without introducing a D2H synchronization.
  validate_grouped_2d_splits<<<DIVUP(count, size_t{128}), 128, 0, stream>>>(
      offsets, rows, count, capacity, cols, noop_ptr);
  group_hadamard_transform_amax_graph_safe(input, output, 0, 0, true, stream, noop_ptr);
  if (capacity == 0) return;
  const float* amax =
      static_cast<const float*>(rowwise ? output->amax.dptr : output->columnwise_amax.dptr);
  checkCuDriverContext(stream);
  alignas(64) CUtensorMap input_map{};
  create_2D_tensor_map(input_map, input->data, capacity, cols, 32, 128, cols, 0, 16);
  TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(scale_type, ScaleType, {
    TRANSFORMER_ENGINE_SWITCH_CONDITION(rowwise, ROWWISE, {
      TRANSFORMER_ENGINE_SWITCH_CONDITION(columnwise, COLUMNWISE, {
        TRANSFORMER_ENGINE_SWITCH_CONDITION(output->with_gemm_swizzled_scales, SWIZZLED, {
          grouped_nvfp4_2d_quantize_transpose<ScaleType, ROWWISE, COLUMNWISE, SWIZZLED>
              <<<dim3((cols / 128) * (capacity / 128)), 128, 0, stream>>>(
                  input_map, static_cast<uint8_t*>(output->data.dptr),
                  static_cast<uint8_t*>(output->columnwise_data.dptr),
                  static_cast<ScaleType*>(output->scale_inv.dptr),
                  static_cast<ScaleType*>(output->columnwise_scale_inv.dptr), amax, offsets, rows,
                  count, cols, noop_ptr);
        });
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("Grouped NVFP4 2D requires FP4 support in the CUDA toolkit.");
#endif
}
}  // namespace transformer_engine::dispatch::nvfp4
