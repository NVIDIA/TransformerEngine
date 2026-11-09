/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "partial_cast.h"

#include <transformer_engine/recipe.h>

#include <limits>

void nvte_partial_cast(const NVTETensor inp, NVTETensor out, size_t h, size_t w,
                       size_t start_offset, cudaStream_t stream) {
  NVTE_API_CALL(nvte_partial_cast);
  using namespace transformer_engine;
  const auto &input = *convertNVTETensorCheck(inp);
  auto &output = *convertNVTETensorCheck(out);
  NVTE_CHECK(h > 0 && w > 0 && h <= std::numeric_limits<size_t>::max() / w,
             "Invalid logical matrix dimensions");
  NVTE_CHECK(start_offset <= h * w && input.numel() <= h * w - start_offset,
             "Input shard exceeds the logical matrix");
  NVTE_CHECK(input.dtype() == DType::kFloat32 || input.dtype() == DType::kBFloat16 ||
                 input.dtype() == DType::kFloat16,
             "Partial cast requires float32, bfloat16, or float16 input");
  NVTE_CHECK(output.numel() == input.numel(),
             "Input and rowwise output must have the same logical size");
  NVTE_CHECK(!output.with_gemm_swizzled_scales, "Partial cast requires unswizzled scales");
  NVTE_CHECK(input.numel() == 0 || (input.data.dptr != nullptr && output.data.dptr != nullptr),
             "Partial cast requires input and output data buffers");
  NVTE_CHECK(output.scaling_mode == NVTE_BLOCK_SCALING_2D ||
                 output.scaling_mode == NVTE_MXFP8_1D_SCALING ||
                 output.scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Unsupported scaling mode for partial cast: ", to_string(output.scaling_mode));
  if (input.numel() == 0) return;

  switch (output.scaling_mode) {
    case NVTE_BLOCK_SCALING_2D:
      fp8_block_scaling_recipe::partial_cast(input, output, h, w, start_offset, stream);
      break;
    case NVTE_MXFP8_1D_SCALING:
      mxfp8_scaling_recipe::partial_cast(input, output, h, w, start_offset, stream);
      break;
    case NVTE_NVFP4_1D_SCALING:
      nvfp4_recipe::partial_cast(input, output, h, w, start_offset, stream);
      break;
    default:
      NVTE_ERROR("Unsupported scaling mode for partial cast: ", to_string(output.scaling_mode));
  }
}
