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
template <typename ScaleType>
__global__ void grouped_nvfp4_2d_quantize_transpose(const bf16* input, uint8_t* row_data,
                                                    uint8_t* col_data, ScaleType* row_scales,
                                                    ScaleType* col_scales, const float* amax,
                                                    const int64_t* offsets, const int64_t* rows,
                                                    size_t count, size_t cols, bool swizzled,
                                                    const float* noop) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  if (noop && *noop == 1.0f) return;
  const size_t global_row = (static_cast<size_t>(blockIdx.x) / (cols / 128)) * 32;
  if (global_row * cols >= offsets[count]) return;
  const size_t expert = common::find_tensor_from_offsets(offsets, count, global_row * cols);
  const size_t start = offsets[expert] / cols;
  const size_t local_row = global_row - start;
  // Each half-warp owns one 16x16 scale block; a CTA covers a 32x128 tile.
  const int scale_block = threadIdx.x / 16;
  const int block_row = (scale_block / 8) * 16;
  const int block_col = (scale_block % 8) * 16;
  const int lane = threadIdx.x % 16;
  const size_t col = (blockIdx.x % (cols / 128)) * 128 + block_col;
  const size_t m = rows[expert];
  __shared__ bf16 tile[32][129];
  alignas(16) bf16 values[16];
  const bf16* source = input + (global_row + block_row + lane) * cols + col;
  reinterpret_cast<uint4*>(values)[0] = reinterpret_cast<const uint4*>(source)[0];
  reinterpret_cast<uint4*>(values)[1] = reinterpret_cast<const uint4*>(source)[1];
  float a = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    tile[block_row + lane][block_col + j] = values[j];
    a = fmaxf(a, fabsf(static_cast<float>(values[j])));
  }
  for (int delta = 8; delta > 0; delta /= 2) {
    a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, delta, 16));
  }
  __syncwarp();

  // Use the same arithmetic and packed BF16 conversion as single-tensor 2D quantization.
  const float encode = core::compute_global_encode_scaling_factor_FP4<ScaleType>(amax[expert]);
  const float decode = 1.0 / encode;
  const ScaleType scale = core::compute_decoding_scaling_factor<ScaleType>(a, encode);
  const float inverse =
      fminf(1.0f / (static_cast<float>(scale) * decode), detail::TypeExtrema<float>::max);
  const float2 inverse2{inverse, inverse};
  const size_t scale_base = offsets[expert] / 16;
  if (row_data) {
    uint64_t packed_values = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const auto packed = ptx::mul_cvt_bf16_to_fp4_4x<false>(
          *reinterpret_cast<const uint64_t*>(values + 4 * p), inverse2, 0);
      packed_values |= static_cast<uint64_t>(packed.__x) << (16 * p);
    }
    *reinterpret_cast<uint64_t*>(row_data + (global_row + block_row + lane) * (cols / 2) +
                                 col / 2) = packed_values;
    const size_t r = local_row + block_row + lane;
    const size_t c = col / 16;
    const size_t index =
        swizzled ? swizzle::gemm_swizzled_scale_idx(r, c, cols / 16) : r * (cols / 16) + c;
    row_scales[scale_base + index] = scale;
  }
  if (col_data) {
    uint64_t packed_values = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
      for (int j = 0; j < 4; ++j) values[j] = tile[block_row + 4 * p + j][block_col + lane];
      const auto packed = ptx::mul_cvt_bf16_to_fp4_4x<false>(
          *reinterpret_cast<const uint64_t*>(values), inverse2, 0);
      packed_values |= static_cast<uint64_t>(packed.__x) << (16 * p);
    }
    *reinterpret_cast<uint64_t*>(col_data + offsets[expert] / 2 + (col + lane) * (m / 2) +
                                 (local_row + block_row) / 2) = packed_values;
    const size_t r = col + lane;
    const size_t c = (local_row + block_row) / 16;
    const size_t index =
        swizzled ? swizzle::gemm_swizzled_scale_idx(r, c, m / 16) : r * (m / 16) + c;
    col_scales[scale_base + index] = scale;
  }
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
  TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(
      scale_type, ScaleType,
      grouped_nvfp4_2d_quantize_transpose<ScaleType>
      <<<dim3((cols / 128) * (capacity / 32)), 256, 0, stream>>>(
          static_cast<const bf16*>(input->data.dptr), static_cast<uint8_t*>(output->data.dptr),
          static_cast<uint8_t*>(output->columnwise_data.dptr),
          static_cast<ScaleType*>(output->scale_inv.dptr),
          static_cast<ScaleType*>(output->columnwise_scale_inv.dptr), amax, offsets, rows, count,
          cols, output->with_gemm_swizzled_scales, noop_ptr););
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("Grouped NVFP4 2D requires FP4 support in the CUDA toolkit.");
#endif
}
}  // namespace transformer_engine::dispatch::nvfp4
