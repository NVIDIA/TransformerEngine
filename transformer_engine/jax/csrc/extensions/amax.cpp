/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime.h>

#include <iostream>

#include "../extensions.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/hadamard_transform.h"
#include "transformer_engine/recipe.h"
#include "transformer_engine/transformer_engine.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

Error_Type RHTAmaxCalculationFFI(cudaStream_t stream, Buffer_Type input_buf, Result_Type amax_buf,
                                 Result_Type post_rht_amax_buf,
                                 int64_t rht_matrix_random_sign_mask_t, bool produce_regular_amax,
                                 int64_t flatten_axis) {
  NVTE_CHECK(input_buf.untyped_data() != nullptr,
             "Input must be provided for RHT Amax calculation");
  NVTE_CHECK(convert_ffi_datatype_to_te_dtype(input_buf.element_type()) == DType::kBFloat16,
             "Input must be of type bfloat16 for RHT Amax calculation");

  NVTE_CHECK(flatten_axis > 0 && flatten_axis < static_cast<int64_t>(input_buf.dimensions().size()),
             "Flatten axis is out of bounds");
  TensorWrapper input_tensor(input_buf.untyped_data(),
                             std::vector<size_t>{product(input_buf.dimensions(), 0, flatten_axis),
                                                 product(input_buf.dimensions(), flatten_axis,
                                                         input_buf.dimensions().size())},
                             convert_ffi_datatype_to_te_dtype(input_buf.element_type()));

  float *amax_out = nullptr;
  if (produce_regular_amax) {
    amax_out = reinterpret_cast<float *>(amax_buf->untyped_data());
    NVTE_CHECK(amax_out != nullptr, "Amax output must be provided for RHT Amax calculation");
    NVTE_CHECK(convert_ffi_datatype_to_te_dtype(amax_buf->element_type()) == DType::kFloat32,
               "Amax output must be of type float32 for RHT Amax calculation");
    NVTE_CHECK(amax_buf->dimensions().size() == 1 && amax_buf->dimensions()[0] == 1,
               "Amax output must be a single float for RHT Amax calculation");
  }

  float *post_rht_amax_out = reinterpret_cast<float *>(post_rht_amax_buf->untyped_data());
  NVTE_CHECK(post_rht_amax_out != nullptr,
             "Post-RHT Amax output must be provided for RHT Amax calculation");
  NVTE_CHECK(convert_ffi_datatype_to_te_dtype(post_rht_amax_buf->element_type()) == DType::kFloat32,
             "Post-RHT Amax output must be of type float32 for RHT Amax calculation");
  NVTE_CHECK(post_rht_amax_buf->dimensions().size() == 1 && post_rht_amax_buf->dimensions()[0] == 1,
             "Post-RHT Amax output must be a single float for RHT Amax calculation");

  TensorWrapper out_tensor{};
  out_tensor.set_amax(amax_out, DType::kFloat32, std::vector<size_t>{1});
  out_tensor.set_columnwise_amax(post_rht_amax_out, DType::kFloat32, std::vector<size_t>{1});

  // Zero'ing of amaxes is handled by TE common inside nvte_hadamard_transform_amax
  nvte_hadamard_transform_amax(input_tensor.data(), out_tensor.data(),
                               0,  // Regular amax for rowwise does not apply RHT so mask is 0
                               rht_matrix_random_sign_mask_t, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RHTAmaxCalculationHandler, RHTAmaxCalculationFFI,
    FFI::Bind()
        .Ctx<FFI_Stream_Type>()                          // stream
        .Arg<Buffer_Type>()                              // input
        .Ret<Buffer_Type>()                              // amax
        .Ret<Buffer_Type>()                              // post_rht_amax
        .Attr<int64_t>("rht_matrix_random_sign_mask_t")  // rht_matrix_random_sign_mask_t
        .Attr<bool>("produce_regular_amax")              // produce_regular_amax
        .Attr<int64_t>("flatten_axis"),                  // flatten_axis
    FFI_CudaGraph_Traits);

Error_Type RHTAmaxCalculationInitializeFFI(cudaStream_t stream, Buffer_Type input_buf,
                                           Result_Type amax_buf, Result_Type post_rht_amax_buf,
                                           int64_t rht_matrix_random_sign_mask_t,
                                           bool produce_regular_amax, int64_t flatten_axis) {
  return wrapInStreamCapture(std::function(RHTAmaxCalculationFFI), stream, input_buf, amax_buf,
                             post_rht_amax_buf, rht_matrix_random_sign_mask_t, produce_regular_amax,
                             flatten_axis);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RHTAmaxCalculationInitializeHandler, RHTAmaxCalculationInitializeFFI,
    FFI::Bind<FFI_Initialize>()
        .Ctx<FFI_Stream_Type>()                          // stream
        .Arg<Buffer_Type>()                              // input
        .Ret<Buffer_Type>()                              // amax
        .Ret<Buffer_Type>()                              // post_rht_amax
        .Attr<int64_t>("rht_matrix_random_sign_mask_t")  // rht_matrix_random_sign_mask_t
        .Attr<bool>("produce_regular_amax")              // produce_regular_amax
        .Attr<int64_t>("flatten_axis"));                 // flatten_axis

}  // namespace jax
}  // namespace transformer_engine
