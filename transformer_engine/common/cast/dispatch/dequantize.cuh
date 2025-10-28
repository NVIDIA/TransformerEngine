/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize.cuh
 *  \brief Dequantize dispatcher.
 */

#ifndef TRANSFORMER_ENGINE_DISPATCH_DEQUANTIZE_CUH_
#define TRANSFORMER_ENGINE_DISPATCH_DEQUANTIZE_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../fp8/dequantize_fp8.cuh"
#include "../mxfp8/dequantize_mxfp8.cuh"
#include "../nvfp4/dequantize_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {

inline void dequantize_helper(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  switch (input.scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
      NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
      NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");
      fp8::dequantize(input, output, stream);
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      if (is_supported_by_CC_100()) {
        mxfp8::dequantize(input, output, stream);
      } else {
        NVTE_ERROR("MXFP8 Dequantization is NOT supported by architectures < 10.0");
      }
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      nvfp4::dequantize(input, output, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(input.scaling_mode) + ".");
  }
}

}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DISPATCH_DEQUANTIZE_CUH_
