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
void quantize_gated_helper(const NVTETensor nvte_input, NVTETensor nvte_output, ParamOP &p,
                           cudaStream_t stream) {
  using namespace dispatch;
  const Tensor input = *convertNVTETensorCheck(nvte_input);
  Tensor *output = convertNVTETensorCheck(nvte_output);

  const auto scaling_mode = output->scaling_mode;
  if ((scaling_mode != NVTE_DELAYED_TENSOR_SCALING) && !is_supported_by_CC_100()) {
    NVTE_ERROR("Not supported by the Arch < 10.0");
  }

  constexpr bool allow_empty = false;
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", allow_empty);

  NVTE_CHECK(input.flat_last_dim() % 2 == 0, "Number of columns must be even.");

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim() / 2;

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  bool is_fp8_rowwise_output = true;
  bool is_fp8_colwise_output = true;
  if (output->has_data()) {
    is_fp8_rowwise_output = is_fp8_dtype(output->data.dtype);
    NVTE_CHECK(output->flat_first_dim() == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->flat_last_dim() == cols, "Wrong dimension of the output.");
  }
  if (output->has_columnwise_data()) {
    is_fp8_colwise_output = is_fp8_dtype(output->columnwise_data.dtype);
    NVTE_CHECK(output->flat_first_dim() == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->flat_last_dim() == cols, "Wrong dimension of the output.");
  }

  const bool use_tma_kernels = is_fp8_rowwise_output && is_fp8_colwise_output && cols % 32 == 0;

  switch(scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      if (use_tma_kernels) {
        Tensor dummy_tensor;  // grad
        fp8::cast_gated_dgated_tma<false, ParamOP, ActOP, nullptr>(dummy_tensor, input, output, p, stream);
      } else {
        fp8::cast_gated<ParamOP, ActOP>(input, output, p, stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      if (use_tma_kernels) {
        Tensor dummy_tensor;  // grad
        mxfp8::quantize_gated_dgated<false, ParamOP, ActOP, nullptr>(dummy_tensor, input, output, p, stream);
      } else {
        NVTE_ERROR("Invalid input shape. Expected the last dimension to be divisible ",
                   "by 32, got input of shape ", input.data.shape);
      }
      break;
    }
    default:
      NVTE_ERROR("Not supported scaling mode: " + to_string(scaling_mode) + ".");
  }
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &), 
          float (*DActOP)(float, const ParamOP &)>
void quantize_dgated_helper(const NVTETensor nvte_grad, const NVTETensor nvte_gated_input,
                            NVTETensor nvte_output, ParamOP &p, cudaStream_t stream) {
  using namespace dispatch;
  const Tensor &grad = *(convertNVTETensorCheck(nvte_grad));
  const Tensor gated_input = *convertNVTETensorCheck(nvte_gated_input);
  Tensor *output = convertNVTETensorCheck(nvte_output);

  const auto scaling_mode = output->scaling_mode;
  if ((scaling_mode != NVTE_DELAYED_TENSOR_SCALING) && !is_supported_by_CC_100()) {
    NVTE_ERROR("Not supported by the Arch < 10.0");
  }

  constexpr bool allow_empty = false;
  CheckInputTensor(gated_input, "gated_input");
  CheckOutputTensor(*output, "output", allow_empty);

  NVTE_CHECK(gated_input.flat_last_dim() % 2 == 0, "Number of columns must be even.");

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = 2 * cols;

  CheckInputTensor(grad, "grad");
  NVTE_CHECK(!is_fp8_dtype(grad.data.dtype), "Grad input must be in higher precision.");
  NVTE_CHECK(grad.data.dtype == gated_input.data.dtype, "Types of both inputs must match.");
  NVTE_CHECK(grad.flat_first_dim() == rows, "Wrong dimension of the grad input.");
  NVTE_CHECK(grad.flat_last_dim() == cols, "Wrong dimension of the grad input.");

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  bool is_fp8_rowwise_output = true;
  bool is_fp8_colwise_output = true;
  if (output->has_data()) {
    is_fp8_rowwise_output = is_fp8_dtype(output->data.dtype);
    NVTE_CHECK(output->flat_first_dim() == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->flat_last_dim() == output_cols, "Wrong dimension of the output.");
  }
  if (output->has_columnwise_data()) {
    is_fp8_colwise_output = is_fp8_dtype(output->columnwise_data.dtype);
    NVTE_CHECK(output->flat_first_dim() == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->flat_last_dim() == output_cols, "Wrong dimension of the output.");
  }

  const bool use_tma_kernels = is_fp8_rowwise_output && is_fp8_colwise_output && (cols % 32 == 0);

  switch(scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      if (use_tma_kernels) {
        fp8::cast_gated_dgated_tma<true, ParamOP, ActOP, DActOP>(grad, gated_input, output, p, stream);
      } else {
        fp8::cast_dgated<ParamOP, ActOP, DActOP>(grad, gated_input, output, p, stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      if (use_tma_kernels) {
        mxfp8::quantize_gated_dgated<true, ParamOP, ActOP, DActOP>(grad, gated_input, output, p, stream);
      } else {
        NVTE_ERROR("Invalid input shape. Expected the last dimension to be divisible ",
                  "by 32, got input of shape ", gated_input.data.shape);
      }
      break;
    }
    default:
      NVTE_ERROR("Not supported scaling mode: " + to_string(scaling_mode) + ".");
  }
}
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DISPATCH_GATED_CUH_
