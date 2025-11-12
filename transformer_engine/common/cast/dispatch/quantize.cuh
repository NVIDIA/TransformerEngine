/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize.cuh
 *  \brief Quantize dispatcher.
 */

#ifndef TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_CUH_
#define TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../transpose/cast_transpose.h"
#include "../../util/vectorized_pointwise.h"
#include "../core/common.cuh"
#include "../fp8/quantize_fp8.cuh"
#include "../mxfp8/quantize_mxfp8.cuh"
#include "../nvfp4/quantize_nvfp4.cuh"
#include "../nvfp4/quantize_transpose_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_fwd_helper(const NVTETensor input, NVTETensor output,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  using namespace detail;

  const Tensor *input_tensor = convertNVTETensorCheck(input);
  Tensor *output_tensor = convertNVTETensorCheck(output);

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

  // Check for unsupported options
  if (quant_config_cpp.stochastic_rounding) {
    NVTE_CHECK(output_tensor->scaling_mode == NVTE_NVFP4_1D_SCALING,
               "Stochastic rounding is only supported for NVFP4 quantization.");
  }

  NVTE_CHECK(output_tensor->has_data() || output_tensor->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  // Dispatch to quantization kernel depending on data format
  switch (output_tensor->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      const Tensor *dummy_input_tensor = nullptr;
      Tensor *dummy_dbias_tensor = nullptr;
      Tensor *dummy_workspace_tensor = nullptr;
      if (output_tensor->has_columnwise_data()) {
        NVTE_CHECK(output_tensor->has_data(),
                   "Quantizing in only the columnwise direction not supported yet!");
        if constexpr (!IS_ACT) {
          cast_transpose(*input_tensor, *noop_tensor, output_tensor, stream);
        } else {
          cast_transpose_fused</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, float, ParamOP, OP>(
              *input_tensor, dummy_input_tensor, output_tensor, dummy_dbias_tensor,
              dummy_workspace_tensor, stream);
        }
      } else if (output_tensor->has_data()) {
        fp8::quantize</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, ParamOP, OP>(
            *input_tensor, dummy_input_tensor, noop_tensor, output_tensor, dummy_dbias_tensor,
            dummy_workspace_tensor, stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      const Tensor *dummy_input_tensor = nullptr;
      Tensor *dummy_dbias_tensor = nullptr;
      Tensor *dummy_workspace_tensor = nullptr;
      mxfp8::quantize</*IS_DBIAS=*/false, /*IS_DACT=*/false, IS_ACT, ParamOP, OP>(
          *input_tensor, dummy_input_tensor, noop_tensor, output_tensor, dummy_dbias_tensor,
          dummy_workspace_tensor, stream);
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      NVTE_CHECK(!IS_ACT, "IS_ACT is not supported by FWD NVTE_NVFP4_1D_SCALING");

      // Check tensors
      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*input_tensor, "input");
      CheckOutputTensor(*output_tensor, "output", false);

      // Choose kernel
      int32_t rows = input_tensor->flat_first_dim();
      int32_t cols = input_tensor->flat_last_dim();
      auto dtype = input_tensor->dtype();
      bool use_optimized_kernel = (dtype == DType::kBFloat16) && (rows % 32 == 0) &&
                                  (cols % 32 == 0) && output_tensor->has_data();

      // Launch NVFP4 quantize kernel
      if (use_optimized_kernel) {
        if (quant_config_cpp.nvfp4_2d_quantization) {
          nvfp4::quantize_transpose</*use_2d_quantization=*/true>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        } else {
          nvfp4::quantize_transpose</*use_2d_quantization*/ false>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        }
      } else {
        auto &global_amax = (output_tensor->amax.dptr != nullptr) ? output_tensor->amax
                                                                  : output_tensor->columnwise_amax;
        quantize_transpose_vector_blockwise_fp4(
            /*input=*/input_tensor->data, /*global_amax=*/global_amax,
            /*scale_inv=*/output_tensor->scale_inv,
            /*scale_inv_t=*/output_tensor->columnwise_scale_inv,
            /*output=*/output_tensor->data, /*output_t=*/output_tensor->columnwise_data,
            /*epsilon=*/0.0f, /*return_identity=*/output_tensor->has_data(),
            /*return_transpose=*/output_tensor->has_columnwise_data(), /*pow2_scale=*/false,
            /*swizzled_scale=*/false,
            /*use_stochastic_rounding=*/quant_config_cpp.stochastic_rounding,
            /*rng_state=*/quant_config_cpp.rng_state,
            /*use_2d_quantization=*/quant_config_cpp.nvfp4_2d_quantization,
            /*noop_tensor=*/noop_tensor->data, /*stream=*/stream);
      }
      break;
    }
    case NVTE_BLOCK_SCALING_2D: {
      // TODO(kwyss): IS_ACT, ParamOP, OP parameters support.
      NVTE_CHECK(!IS_ACT, "IS_ACT is not implemented for FWD NVTE_BLOCK_SCALING_2D");
      bool force_pow_2_scales = quant_config_cpp.force_pow_2_scales;
      float epsilon = quant_config_cpp.amax_epsilon;
      quantize_transpose_square_blockwise(
          input_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon,
          /*return_transpose=*/output_tensor->has_columnwise_data(), force_pow_2_scales,
          /*noop_tensor=*/noop_tensor->data, stream);
      break;
    }
    case NVTE_BLOCK_SCALING_1D: {
      // TODO(kwyss): IS_ACT, ParamOP, OP parameters support.
      NVTE_CHECK(!IS_ACT, "IS_ACT is not implemented for FWD NVTE_BLOCK_SCALING_1D");
      bool force_pow_2_scales = quant_config_cpp.force_pow_2_scales;
      float epsilon = quant_config_cpp.amax_epsilon;
      FP8BlockwiseRowwiseOption rowwise_option = FP8BlockwiseRowwiseOption::NONE;
      FP8BlockwiseColumnwiseOption columnwise_option = FP8BlockwiseColumnwiseOption::NONE;
      if (output_tensor->has_data()) {
        bool rowwise_compact = (quant_config_cpp.float8_block_scale_tensor_format ==
                                Float8BlockScaleTensorFormat::COMPACT);
        rowwise_option = rowwise_compact ? FP8BlockwiseRowwiseOption::ROWWISE_COMPACT
                                         : FP8BlockwiseRowwiseOption::ROWWISE_GEMM_READY;
      }
      if (output_tensor->has_columnwise_data()) {
        bool columnwise_compact = (quant_config_cpp.float8_block_scale_tensor_format ==
                                   Float8BlockScaleTensorFormat::COMPACT);
        columnwise_option = columnwise_compact
                                ? FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT
                                : FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY;
      }
      quantize_transpose_vector_blockwise(
          input_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon, rowwise_option,
          columnwise_option, force_pow_2_scales, noop_tensor->data, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(output_tensor->scaling_mode) + ".");
  }
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_bwd_helper(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                         NVTETensor dbias, NVTETensor workspace,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  using namespace detail;

  const Tensor *grad_tensor = convertNVTETensorCheck(grad);
  const Tensor *input_tensor = convertNVTETensor(input);

  Tensor *output_tensor = convertNVTETensorCheck(output);
  Tensor *dbias_tensor = convertNVTETensor(dbias);
  Tensor *workspace_tensor = convertNVTETensor(workspace);

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

  // Check for unsupported options
  if (quant_config_cpp.stochastic_rounding) {
    NVTE_CHECK(output_tensor->scaling_mode == NVTE_NVFP4_1D_SCALING,
               "Stochastic rounding is only supported for NVFP4 quantization.");
  }

  NVTE_CHECK(output_tensor->has_data() || output_tensor->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  // Dispatch to quantization kernel depending on data format
  switch (output_tensor->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      if (output_tensor->has_columnwise_data()) {
        NVTE_CHECK(output_tensor->has_data(),
                   "Quantizing in only the columnwise direction not supported yet!");
        if constexpr (!IS_DBIAS && !IS_DACT) {
          cast_transpose(*grad_tensor, *noop_tensor, output_tensor, stream);
        } else {
          cast_transpose_fused<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, float, ParamOP, OP>(
              *grad_tensor, input_tensor, output_tensor, dbias_tensor, workspace_tensor, stream);
        }
      } else if (output_tensor->has_data()) {
        fp8::quantize<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, ParamOP, OP>(
            *grad_tensor, input_tensor, noop_tensor, output_tensor, dbias_tensor, workspace_tensor,
            stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      mxfp8::quantize<IS_DBIAS, IS_DACT, /*IS_ACT=*/false, ParamOP, OP>(
          *grad_tensor, input_tensor, noop_tensor, output_tensor, dbias_tensor, workspace_tensor,
          stream);
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      NVTE_CHECK((!IS_DBIAS && !IS_DACT),
                 "IS_DBIAS and IS_DACT are not supported by BWD NVTE_NVFP4_1D_SCALING");

      // Check tensors
      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*grad_tensor, "input");
      CheckOutputTensor(*output_tensor, "output", false);

      // Choose kernel
      int32_t rows = grad_tensor->flat_first_dim();
      int32_t cols = grad_tensor->flat_last_dim();
      auto dtype = grad_tensor->dtype();
      bool use_optimized_kernel = (dtype == DType::kBFloat16) && (rows % 32 == 0) &&
                                  (cols % 32 == 0) && output_tensor->has_data();

      // Launch NVFP4 quantize kernel
      if (use_optimized_kernel) {
        if (quant_config_cpp.nvfp4_2d_quantization) {
          nvfp4::quantize_transpose</*use_2d_quantization=*/true>(
              *grad_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        } else {
          nvfp4::quantize_transpose</*use_2d_quantization*/ false>(
              *grad_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        }
      } else {
        auto &global_amax = (output_tensor->amax.dptr != nullptr) ? output_tensor->amax
                                                                  : output_tensor->columnwise_amax;
        quantize_transpose_vector_blockwise_fp4(
            /*input=*/grad_tensor->data, /*global_amax=*/global_amax,
            /*scale_inv=*/output_tensor->scale_inv,
            /*scale_inv_t=*/output_tensor->columnwise_scale_inv,
            /*output=*/output_tensor->data, /*output_t=*/output_tensor->columnwise_data,
            /*epsilon=*/0.0f, /*return_identity=*/output_tensor->has_data(),
            /*return_transpose=*/output_tensor->has_columnwise_data(), /*pow2_scale=*/false,
            /*swizzled_scale=*/false,
            /*use_stochastic_rounding=*/quant_config_cpp.stochastic_rounding,
            /*rng_state=*/quant_config_cpp.rng_state,
            /*use_2d_quantization=*/quant_config_cpp.nvfp4_2d_quantization,
            /*noop_tensor=*/noop_tensor->data, /*stream=*/stream);
      }
      break;
    }
    case NVTE_BLOCK_SCALING_2D: {
      // TODO(kwyss): IS_BIAS, IS_DACT, ParamOP, OP parameters support.
      NVTE_CHECK((!IS_DBIAS && !IS_DACT),
                 "IS_DBIAS and IS_DACT are not implemented for BWD NVTE_BLOCK_SCALING_2D");
      bool force_pow_2_scales = quant_config_cpp.force_pow_2_scales;
      float epsilon = quant_config_cpp.amax_epsilon;
      quantize_transpose_square_blockwise(
          grad_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon,
          /*return_transpose=*/output_tensor->has_columnwise_data(), force_pow_2_scales,
          /*noop_tensor=*/noop_tensor->data, stream);
      break;
    }
    case NVTE_BLOCK_SCALING_1D: {
      // TODO(kwyss): IS_BIAS, IS_DACT, ParamOP, OP parameters support.
      NVTE_CHECK((!IS_DBIAS && !IS_DACT),
                 "IS_DBIAS and IS_DACT are not implemented for BWD NVTE_BLOCK_SCALING_1D");
      bool force_pow_2_scales = quant_config_cpp.force_pow_2_scales;
      float epsilon = quant_config_cpp.amax_epsilon;
      FP8BlockwiseRowwiseOption rowwise_option = FP8BlockwiseRowwiseOption::NONE;
      FP8BlockwiseColumnwiseOption columnwise_option = FP8BlockwiseColumnwiseOption::NONE;
      if (output_tensor->has_data()) {
        bool rowwise_compact = (quant_config_cpp.float8_block_scale_tensor_format ==
                                Float8BlockScaleTensorFormat::COMPACT);
        rowwise_option = rowwise_compact ? FP8BlockwiseRowwiseOption::ROWWISE_COMPACT
                                         : FP8BlockwiseRowwiseOption::ROWWISE_GEMM_READY;
      }
      if (output_tensor->has_columnwise_data()) {
        bool columnwise_compact = (quant_config_cpp.float8_block_scale_tensor_format ==
                                   Float8BlockScaleTensorFormat::COMPACT);
        columnwise_option = columnwise_compact
                                ? FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT
                                : FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY;
      }
      quantize_transpose_vector_blockwise(
          grad_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon, rowwise_option,
          columnwise_option, force_pow_2_scales, noop_tensor->data, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(output_tensor->scaling_mode) + ".");
  }
}

}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DISPATCH_QUANTIZE_CUH_
