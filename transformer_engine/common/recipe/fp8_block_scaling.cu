/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <cassert>

#include "../common.h"
#include "../utils.cuh"
#include "partial_cast.h"

namespace transformer_engine {
namespace fp8_block_scaling_recipe {

constexpr int kTileDim = 128;
constexpr int kThreadsPerBlock = 256;

__global__ void pack_partial_cast_scales(const float *input, float *output, size_t rows,
                                         size_t cols, size_t stride_h, size_t stride_w) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * cols) {
    output[index] = input[(index / cols) * stride_h + (index % cols) * stride_w];
  }
}

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

  // Store the casted data into the output.
  // Note that this store operation might write "out-of-bounds", but it is intentional:
  //   1. The "out-of-bounds" here only crosses the boundary of the "local shard" (i.e., the region
  //      from start_offset to end_offset), not the boundary of the entire output memory. Therefore,
  //      this out-of-bounds write will not cause illegal memory access.
  //   2. We assume that the subsequent all-gather operation happens in-place, so any parts that
  //      should not be updated here will be overwritten by the all-gather.
  // This tricky approach allows us to avoid checking whether each output index falls within
  // [start, end), resulting in a significant performance improvement.
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

void fp8_block_scaling_compute_partial_amax(const Tensor inp, Tensor amax, size_t h, size_t w,
                                            size_t amax_stride_h, size_t amax_stride_w,
                                            size_t start_offset, size_t block_len,
                                            cudaStream_t stream) {
  NVTE_CHECK(block_len == 128, "Currently only block_len = 128 is supported");

  size_t len = inp.numel();

  assert(h > 0 && w > 0);
  assert(start_offset < h * w);
  assert(start_offset + len <= h * w);

  size_t blocks_x = (w + kTileDim - 1) / kTileDim;
  size_t blocks_y = (h + kTileDim - 1) / kTileDim;
  assert(blocks_x <= std::numeric_limits<unsigned int>::max());
  assert(blocks_y <= std::numeric_limits<unsigned int>::max());
  dim3 grid(blocks_x, blocks_y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      inp.dtype(), inp_dtype,
      fp8_block_scaling_compute_partial_amax_kernel<inp_dtype>
      <<<grid, kThreadsPerBlock, 0, stream>>>(reinterpret_cast<const inp_dtype *>(inp.data.dptr),
                                              reinterpret_cast<float *>(amax.data.dptr),
                                              amax_stride_h, amax_stride_w, h, w, start_offset,
                                              len);)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void partial_cast(const Tensor &inp, Tensor &out, size_t h, size_t w, size_t start_offset,
                  cudaStream_t stream) {
  const auto &scale = out.scale;
  NVTE_CHECK(out.dtype() == DType::kFloat8E4M3 || out.dtype() == DType::kFloat8E5M2,
             "Block FP8 output must be E4M3 or E5M2");
  NVTE_CHECK(scale.dptr != nullptr && scale.dtype == DType::kFloat32,
             "Block FP8 encoding scales must be float32");
  NVTE_CHECK(scale.shape.size() == 2 && scale.shape[0] >= DIVUP<size_t>(h, kTileDim) &&
                 scale.shape[1] >= DIVUP<size_t>(w, kTileDim),
             "Invalid block FP8 scale shape");
  const size_t scale_stride_h = scale.shape[1];
  constexpr size_t scale_stride_w = 1;

  size_t len = inp.numel();

  assert(h > 0 && w > 0);
  assert(start_offset < h * w);
  assert(start_offset + len <= h * w);

  size_t blocks_x = (w + kTileDim - 1) / kTileDim;
  size_t blocks_y = (h + kTileDim - 1) / kTileDim;
  assert(blocks_x <= std::numeric_limits<unsigned int>::max());
  assert(blocks_y <= std::numeric_limits<unsigned int>::max());
  dim3 grid(blocks_x, blocks_y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      inp.dtype(), inp_dtype,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          out.dtype(), fp8_type,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              w % kTileDim == 0, kWidthAligned,
              fp8_block_scaling_partial_cast_kernel<inp_dtype, fp8_type, kWidthAligned>
              <<<grid, kThreadsPerBlock, 0, stream>>>(
                  reinterpret_cast<const inp_dtype *>(inp.data.dptr),
                  reinterpret_cast<fp8_type *>(out.data.dptr),
                  reinterpret_cast<const float *>(scale.dptr), scale_stride_h, scale_stride_w, h, w,
                  start_offset, len);)))
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8_block_scaling_recipe
}  // namespace transformer_engine

void nvte_fp8_block_scaling_compute_partial_amax(const NVTETensor inp, NVTETensor amax, size_t h,
                                                 size_t w, size_t amax_stride_h,
                                                 size_t amax_stride_w, size_t start_offset,
                                                 size_t block_len, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_block_scaling_compute_partial_amax);
  using namespace transformer_engine;
  fp8_block_scaling_recipe::fp8_block_scaling_compute_partial_amax(
      *convertNVTETensorCheck(inp), *convertNVTETensorCheck(amax), h, w, amax_stride_h,
      amax_stride_w, start_offset, block_len, stream);
}

void nvte_fp8_block_scaling_partial_cast(const NVTETensor inp, NVTETensor out,
                                         const NVTETensor scale, size_t h, size_t w,
                                         size_t scale_stride_h, size_t scale_stride_w,
                                         size_t start_offset, size_t block_len,
                                         const NVTEDType out_dtype, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_block_scaling_partial_cast);
  using namespace transformer_engine;
  NVTE_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  const auto &input = *convertNVTETensorCheck(inp);
  const auto &output = *convertNVTETensorCheck(out);
  const auto &scales = *convertNVTETensorCheck(scale);
  TensorWrapper view(NVTE_BLOCK_SCALING_2D);
  view.set_rowwise_data(output.data.dptr, static_cast<DType>(out_dtype),
                        std::vector<size_t>{input.numel()});
  if (input.numel() == 0) {
    nvte_partial_cast(inp, view.data(), h, w, start_offset, stream);
    return;
  }
  NVTE_CHECK(h > 0 && w > 0 && h <= std::numeric_limits<size_t>::max() / w,
             "Invalid logical matrix dimensions");
  NVTE_CHECK(scales.dtype() == DType::kFloat32 && scales.data.dptr != nullptr,
             "Block FP8 encoding scales must be float32");
  const size_t rows = DIVUP<size_t>(h, 128);
  const size_t cols = DIVUP<size_t>(w, 128);
  const bool needs_pack = scale_stride_w != 1 || scale_stride_h < cols;
  detail::PartialCastWorkspace workspace(needs_pack ? rows * cols * sizeof(float) : 0, stream);
  void *scale_ptr = scales.data.dptr;
  size_t row_stride = scale_stride_h;
  if (needs_pack) {
    constexpr size_t threads = 256;
    fp8_block_scaling_recipe::
        pack_partial_cast_scales<<<DIVUP(rows * cols, threads), threads, 0, stream>>>(
            static_cast<const float *>(scales.data.dptr), static_cast<float *>(workspace.data()),
            rows, cols, scale_stride_h, scale_stride_w);
    NVTE_CHECK_CUDA(cudaGetLastError());
    scale_ptr = workspace.data();
    row_stride = cols;
  }
  view.set_scale(scale_ptr, DType::kFloat32, std::vector<size_t>{rows, row_stride});
  nvte_partial_cast(inp, view.data(), h, w, start_offset, stream);
}
