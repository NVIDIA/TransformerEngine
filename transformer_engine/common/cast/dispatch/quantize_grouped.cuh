/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_grouped.cuh
 *  \brief Quantize Grouped Tensor dispatcher.
 */

#ifndef TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_GROUPED_CUH_
#define TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_GROUPED_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../transpose/cast_transpose.h"
#include "../../util/vectorized_pointwise.h"
#include "../core/common.cuh"
#include "../mxfp8/quantize_grouped_mxfp8.cuh"

namespace transformer_engine {
namespace dispatch {

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_grouped_fwd_helper(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                 const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  using namespace detail;

  NVTEScalingMode scaling_mode = nvte_grouped_tensor_scaling_mode(output);

  // Quantization config
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }

  // Noop flag
  Tensor dummy_tensor;
  Tensor *noop_tensor = &dummy_tensor;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }

  // NVTE_CHECK(output_tensor->has_data() || output_tensor->has_columnwise_data(),
  //            "Either rowwise or columnwise output data need to be allocated.");

  // Dispatch to quantization kernel depending on data format
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      const NVTEGroupedTensor activation = nullptr;
      NVTETensor dbias = nullptr;
      NVTETensor workspace = nullptr;

      const GroupedTensor *input_tensor = convertNVTEGroupedTensorCheck(input);
      GroupedTensor *output_tensor = convertNVTEGroupedTensorCheck(output);
      const GroupedTensor *activations_tensor = convertNVTEGroupedTensor(activation);
      Tensor *dbias_tensor = convertNVTETensor(dbias);
      Tensor *workspace_tensor = convertNVTETensor(workspace);

      mxfp8::quantize_grouped</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, ParamOP, OP>(
          input_tensor, activations_tensor, noop_tensor, output_tensor, dbias_tensor,
          workspace_tensor, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(scaling_mode) + ".");
  }
}

// template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
// void quantize_grouped_bwd_helper(const NVTEGroupedTensor grad, const NVTEGroupedTensor input, NVTEGroupedTensor output,
//                                  NVTEGroupedTensor dbias, NVTEGroupedTensor workspace,
//                                  const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
//   using namespace detail;

//   const Tensor *grad_tensor = convertNVTETensorCheck(grad);
//   const Tensor *input_tensor = convertNVTETensor(input);

//   Tensor *output_tensor = convertNVTETensorCheck(output);
//   Tensor *dbias_tensor = convertNVTETensor(dbias);
//   Tensor *workspace_tensor = convertNVTETensor(workspace);

//   // Quantization config
//   QuantizationConfig quant_config_cpp;
//   if (quant_config != nullptr) {
//     quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
//   }

//   // Noop flag
//   Tensor dummy_tensor;
//   Tensor *noop_tensor = &dummy_tensor;
//   if (quant_config_cpp.noop_tensor != nullptr) {
//     noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
//   }

//   // Check for unsupported options
//   if (quant_config_cpp.stochastic_rounding) {
//     NVTE_CHECK(output_tensor->scaling_mode == NVTE_NVFP4_1D_SCALING,
//                "Stochastic rounding is only supported for NVFP4 quantization.");
//   }

//   NVTE_CHECK(output_tensor->has_data() || output_tensor->has_columnwise_data(),
//              "Either rowwise or columnwise output data need to be allocated.");

//   // Dispatch to quantization kernel depending on data format
//   switch (output_tensor->scaling_mode) {
//     case NVTE_MXFP8_1D_SCALING: {
//       mxfp8::quantize<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, ParamOP, OP>(
//           *grad_tensor, input_tensor, noop_tensor, output_tensor, dbias_tensor, workspace_tensor,
//           stream);
//       break;
//     }
//     default:
//       NVTE_ERROR("Not implemented scaling mode: " + to_string(output_tensor->scaling_mode) + ".");
//   }
// }

}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_GROUPED_CUH_
