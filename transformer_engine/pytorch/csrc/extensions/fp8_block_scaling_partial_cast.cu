/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/common.h"
#include "common/utils.cuh"
#include "extensions.h"
#include "type_shim.h"

constexpr int kTileDim = 128;
constexpr int kThreadsPerBlock = 256;

template <typename IType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fp8_block_scaling_compute_partial_amax_kernel(const IType *input, float *amax_ptr,
                                                  const size_t amax_stride_h,
                                                  const size_t amax_stride_w, const size_t h,
                                                  const size_t w, const size_t start_offset,
                                                  const size_t len) {
  constexpr int kThreadsPerWarp = 32;
  constexpr int kLoopsPerRow = kTileDim / kThreadsPerWarp;
  constexpr int kNumWarps = kThreadsPerBlock / kThreadsPerWarp;
  constexpr int kLoopsPerCol = kTileDim / kNumWarps;

  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const size_t end_offset = start_offset + len;
  const IType *input_minus_offset = input - start_offset;

  __shared__ float smem[kNumWarps];
  float amax = 0.0f;

  for (int loop_col = 0; loop_col < kLoopsPerCol; ++loop_col) {
    size_t r = tile_row * kTileDim + loop_col * kNumWarps + threadIdx.x / kThreadsPerWarp;
    for (int loop_row = 0; loop_row < kLoopsPerRow; ++loop_row) {
      size_t c = tile_col * kTileDim + loop_row * kThreadsPerWarp + (threadIdx.x % kThreadsPerWarp);
      size_t idx = r * w + c;
      if (r < h && c < w && idx >= start_offset && idx < end_offset) {
        float other_amax = fabs(static_cast<float>(input_minus_offset[idx]));
        __builtin_assume(amax >= 0);
        __builtin_assume(other_amax >= 0);
        amax = fmaxf(amax, other_amax);
      }
    }
  }

  for (int delta = kThreadsPerWarp / 2; delta > 0; delta /= 2) {
    float other_amax = __shfl_down_sync(0xFFFFFFFF, amax, delta);
    __builtin_assume(amax >= 0);
    __builtin_assume(other_amax >= 0);
    amax = fmaxf(amax, other_amax);
  }

  if (threadIdx.x % kThreadsPerWarp == 0) {
    smem[threadIdx.x / kThreadsPerWarp] = amax;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 0; i < kNumWarps; ++i) {
      float other_amax = smem[i];
      __builtin_assume(amax >= 0);
      __builtin_assume(other_amax >= 0);
      amax = fmaxf(amax, other_amax);
    }
    amax_ptr[tile_row * amax_stride_h + tile_col * amax_stride_w] = amax;
  }
}

template <typename IType, typename OType, bool kWidthAligned>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fp8_block_scaling_partial_cast_kernel(const IType *input, OType *output, const float *scale_ptr,
                                          const size_t scale_stride_h, const size_t scale_stride_w,
                                          const size_t h, const size_t w, const size_t start_offset,
                                          const size_t len) {
  using transformer_engine::Vec;

  static_assert(sizeof(OType) == 1);
  constexpr int kNumOutputElemsPerBank = 4 / sizeof(OType);
  constexpr int kThreadsPerWarp = 32;
  constexpr int kLoopsPerRow = kTileDim / kThreadsPerWarp;
  constexpr int kNumWarps = kThreadsPerBlock / kThreadsPerWarp;
  constexpr int kRowsPerWarp = kTileDim / kNumWarps;

  __shared__ OType smem[kTileDim][kTileDim + kNumOutputElemsPerBank];

  const int tile_w = blockIdx.x;
  const int tile_h = blockIdx.y;
  const size_t end_offset = start_offset + len;
  const IType *input_minus_offset = input - start_offset;
  OType *output_minus_offset = output - start_offset;

  const float scale = scale_ptr[tile_h * scale_stride_h + tile_w * scale_stride_w];

  // Load input data into shared memory
  bool skip_store = true;
  for (int i = 0; i < kRowsPerWarp; ++i) {
    for (int j = 0; j < kLoopsPerRow; ++j) {
      const int h_in_smem = threadIdx.x / kThreadsPerWarp * kRowsPerWarp + i;
      const int w_in_smem = threadIdx.x % kThreadsPerWarp + kThreadsPerWarp * j;
      const int h_in_input = tile_h * kTileDim + h_in_smem;
      const int w_in_input = tile_w * kTileDim + w_in_smem;
      const size_t idx_in_input = static_cast<size_t>(h_in_input) * w + w_in_input;
      if (h_in_input < h && w_in_input < w && idx_in_input >= start_offset &&
          idx_in_input < end_offset) {
        float inp = static_cast<float>(input_minus_offset[idx_in_input]) * scale;
        smem[h_in_smem][w_in_smem] = static_cast<OType>(inp);
        skip_store = false;
      }
    }
  }

  for (int delta = kThreadsPerWarp / 2; delta > 0; delta /= 2) {
    bool other_skip_store = __shfl_down_sync(0xFFFFFFFF, skip_store, delta);
    skip_store = skip_store && other_skip_store;
  }
  skip_store = __shfl_sync(0xFFFFFFFF, skip_store, 0);
  if (skip_store) {
    return;
  }

  // Store casted data into output
  Vec<OType, kNumOutputElemsPerBank> vec_output;
  for (int i = 0; i < kRowsPerWarp; ++i) {
    const int row_in_smem = threadIdx.x / kThreadsPerWarp * kRowsPerWarp + i;
    const int col_in_smem = threadIdx.x % kThreadsPerWarp * kNumOutputElemsPerBank;
    for (int j = 0; j < kNumOutputElemsPerBank; ++j) {
      vec_output.data.elt[j] = smem[row_in_smem][col_in_smem + j];
    }
    const int row_in_output = tile_h * kTileDim + row_in_smem;
    const int col_in_output = tile_w * kTileDim + col_in_smem;
    const size_t idx_in_output = static_cast<size_t>(row_in_output) * w + col_in_output;
    if (row_in_output < h) {
      if constexpr (kWidthAligned) {
        vec_output.store_to(output_minus_offset + idx_in_output);
      } else {
        int num = min(static_cast<size_t>(kNumOutputElemsPerBank),
                      static_cast<size_t>(col_in_output < w ? w - col_in_output : 0));
        vec_output.store_to_elts(output_minus_offset, idx_in_output, num);
      }
    }
  }
}

void fp8_block_scaling_compute_partial_amax(const at::Tensor &tensor, at::Tensor amax, size_t h,
                                            size_t w, size_t start_offset, size_t block_len) {
  TORCH_CHECK(block_len == 128, "Currently only support block_len = 128");
  TORCH_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  TORCH_CHECK(amax.scalar_type() == at::ScalarType::Float, "amax must be a float tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float ||
                  tensor.scalar_type() == at::ScalarType::BFloat16,
              "tensor must be a float or bfloat16 tensor");

  size_t amax_stride_h = amax.stride(0);
  size_t amax_stride_w = amax.stride(1);
  size_t len = tensor.numel();

  assert(h > 0 && w > 0);
  assert(start_offset < h * w);
  assert(start_offset + len <= h * w);

  size_t blocks_x = (w + kTileDim - 1) / kTileDim;
  size_t blocks_y = (h + kTileDim - 1) / kTileDim;
  assert(blocks_x <= std::numeric_limits<unsigned int>::max());
  assert(blocks_y <= std::numeric_limits<unsigned int>::max());
  dim3 grid(blocks_x, blocks_y);

  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_HALF_AND_BFLOAT(tensor.scalar_type(), 0, "compute_partial_amax",
                                 fp8_block_scaling_compute_partial_amax_kernel<scalar_t_0>
                                 <<<grid, kThreadsPerBlock, 0, stream>>>(
                                     tensor.data_ptr<scalar_t_0>(), amax.data_ptr<float>(),
                                     amax_stride_h, amax_stride_w, h, w, start_offset, len);)
}

void fp8_block_scaling_partial_cast(const at::Tensor &inp, at::Tensor out, const at::Tensor &scale,
                                    size_t h, size_t w, size_t start_offset, size_t block_len,
                                    const transformer_engine::DType out_dtype) {
  TORCH_CHECK(block_len == 128, "Currently only support block_len = 128");
  TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "scale must be a float tensor");
  TORCH_CHECK(
      inp.scalar_type() == at::ScalarType::Float || inp.scalar_type() == at::ScalarType::BFloat16,
      "input must be a float or bfloat16 tensor");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Byte, "output must be a uint8 tensor");
  TORCH_CHECK(out_dtype == transformer_engine::DType::kFloat8E4M3 ||
                  out_dtype == transformer_engine::DType::kFloat8E5M2,
              "out_dtype must be kFloat8E4M3 or kFloat8E5M2");

  size_t scale_stride_h = scale.stride(0);
  size_t scale_stride_w = scale.stride(1);
  size_t len = inp.numel();

  assert(h > 0 && w > 0);
  assert(start_offset < h * w);
  assert(start_offset + len <= h * w);

  size_t blocks_x = (w + kTileDim - 1) / kTileDim;
  size_t blocks_y = (h + kTileDim - 1) / kTileDim;
  assert(blocks_x <= std::numeric_limits<unsigned int>::max());
  assert(blocks_y <= std::numeric_limits<unsigned int>::max());
  dim3 grid(blocks_x, blocks_y);

  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      inp.scalar_type(), 0, "partial_cast",
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          out_dtype, fp8_type,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              w % kTileDim == 0, kWidthAligned,
              fp8_block_scaling_partial_cast_kernel<scalar_t_0, fp8_type, kWidthAligned>
              <<<grid, kThreadsPerBlock, 0, stream>>>(inp.data_ptr<scalar_t_0>(),
                                                      reinterpret_cast<fp8_type *>(out.data_ptr()),
                                                      scale.data_ptr<float>(), scale_stride_h,
                                                      scale_stride_w, h, w, start_offset, len);)))
}
