/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gated.cuh
 *  \brief Gated dispatcher.
 */

#ifndef TRANSFORMER_ENGINE_DISPATCH_GATED_CUH_
#define TRANSFORMER_ENGINE_DISPATCH_GATED_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../utils.cuh"
#include "../fp8/gated_fp8.cuh"
#include "../mxfp8/gated_mxfp8.cuh"

namespace transformer_engine {
namespace dispatch {

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void quantize_gated_fwd_helper(const NVTETensor nvte_input, NVTETensor nvte_output, ParamOP &p,
                               cudaStream_t stream) {
  const Tensor input = *convertNVTETensorCheck(nvte_input);
  Tensor *output = convertNVTETensorCheck(nvte_output);

  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", /*allow_empty=*/false);

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim() / 2;

  NVTE_CHECK(input.flat_last_dim() % 2 == 0,
             "Wrong input shape. Expected (after flattening) last dimension to be even, ", "got [",
             input.flat_first_dim(), ", ", input.flat_last_dim(), "].");
  NVTE_CHECK(output->flat_last_dim() == cols,
             "Wrong output shape. Expected (after flattening) [*, ", cols, "], got [",
             output->flat_first_dim(), ", ", output->flat_last_dim(), "].");

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  switch (output->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      const bool use_tma_kernels = (cols % 32 == 0) && is_supported_by_CC_100();
      if (use_tma_kernels) {
        Tensor dummy_grad_tensor;
        fp8::cast_gated_tma</*IS_BWD=*/false, ParamOP, ActOP, nullptr>(input, dummy_grad_tensor,
                                                                       output, p, stream);
      } else {
        fp8::cast_gated_fwd<ParamOP, ActOP>(input, output, p, stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      NVTE_CHECK(cols % 32 == 0,
                 "Invalid input shape. Expected the last dimension to be "
                 "divisible by 32, but got ",
                 cols, ".");
      if (output->has_data()) {
        NVTE_CHECK(is_fp8_dtype(output->data.dtype),
                   "The type of the output tensor should be FP8.");
      }
      if (output->has_columnwise_data()) {
        NVTE_CHECK(is_fp8_dtype(output->columnwise_data.dtype),
                   "The type of the columnwise output tensor should be FP8.");
      }
      NVTE_CHECK(is_supported_by_CC_100(),
                 "Gated FWD NVTE_MXFP8_1D_SCALING is only supported on SM 10.0+");
      Tensor dummy_grad_tensor;
      mxfp8::quantize_gated</*IS_BWD=*/false, ParamOP, ActOP, nullptr>(input, dummy_grad_tensor,
                                                                       output, p, stream);
      break;
    }
    default:
      NVTE_ERROR("Not supported scaling mode: " + to_string(output->scaling_mode) + ".");
  }
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated_bwd_helper(const NVTETensor nvte_grad, const NVTETensor nvte_gated_input,
                               NVTETensor nvte_output, ParamOP &p, cudaStream_t stream) {
  const Tensor &grad = *(convertNVTETensorCheck(nvte_grad));
  const Tensor gated_input = *convertNVTETensorCheck(nvte_gated_input);
  Tensor *output = convertNVTETensorCheck(nvte_output);

  CheckInputTensor(grad, "grad");
  CheckInputTensor(gated_input, "gated_input");
  CheckOutputTensor(*output, "output", /*allow_empty=*/false);

  NVTE_CHECK(gated_input.flat_last_dim() % 2 == 0, "Number of columns must be even, but got ",
             gated_input.flat_last_dim(), ".");

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;

  NVTE_CHECK(!is_fp8_dtype(grad.data.dtype), "Grad input must be in higher precision.");
  NVTE_CHECK(grad.data.dtype == gated_input.data.dtype, "Types of both inputs must match.");

  NVTE_CHECK(grad.flat_first_dim() == rows,
             "Wrong Grad shape. Expected first dimension (after flattening) [", rows, ", *], got [",
             grad.flat_first_dim(), ", ", grad.flat_last_dim(), "].");
  NVTE_CHECK(grad.flat_last_dim() == cols,
             "Wrong Grad shape. Expected last dimension (after flattening) [", cols, ", *], got [",
             grad.flat_first_dim(), ", ", grad.flat_last_dim(), "].");

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  NVTE_CHECK(output->flat_first_dim() == rows, "Wrong output shape. Expected (after flattening) [",
             rows, ", *], got [", output->flat_first_dim(), ", ", output->flat_last_dim(), "].");
  NVTE_CHECK(output->flat_last_dim() == cols * 2,
             "Wrong output shape. Expected (after flattening) [*, ", cols * 2, "], got [",
             output->flat_first_dim(), ", ", output->flat_last_dim(), "].");
  NVTE_CHECK(gated_input.data.shape == output->data.shape,
             "Gated input and output shapes must match. Input shape: ", gated_input.data.shape,
             ", output shape: ", output->data.shape, ".");

  switch (output->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      const bool use_tma_kernels = (cols % 32 == 0) && is_supported_by_CC_100();
      if (use_tma_kernels) {
        fp8::cast_gated_tma</*IS_BWD=*/true, ParamOP, ActOP, DActOP>(gated_input, grad, output, p,
                                                                     stream);
      } else {
        fp8::cast_gated_bwd<ParamOP, ActOP, DActOP>(gated_input, grad, output, p, stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      NVTE_CHECK(cols % 32 == 0,
                 "Invalid input shape. Expected the last dimension to be "
                 "divisible by 32, but got ",
                 cols, ".");
      if (output->has_data()) {
        NVTE_CHECK(is_fp8_dtype(output->data.dtype),
                   "The type of the output tensor should be FP8.");
      }
      if (output->has_columnwise_data()) {
        NVTE_CHECK(is_fp8_dtype(output->columnwise_data.dtype),
                   "The type of the columnwise output tensor should be FP8.");
      }
      NVTE_CHECK(is_supported_by_CC_100(),
                 "Gated BWD NVTE_MXFP8_1D_SCALING is only supported on SM 10.0+");

      mxfp8::quantize_gated</*IS_BWD=*/true, ParamOP, ActOP, DActOP>(gated_input, grad, output, p,
                                                                     stream);
      break;
    }
    default:
      NVTE_ERROR("Not supported scaling mode: " + to_string(output->scaling_mode) + ".");
  }
}
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DISPATCH_GATED_CUH_
