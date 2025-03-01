/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include "transformer_engine/cast.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

void Quantize(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *amax = reinterpret_cast<float *>(buffers[1]);
  auto *scale = reinterpret_cast<float *>(buffers[2]);
  auto *scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto *output = buffers[4];
  auto *amax_out = reinterpret_cast<float *>(buffers[5]);
  NVTE_CHECK(amax == amax_out, "amax not bound to amax_out in TE/JAX Quantize primitive.");

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  auto shape = desc.shape.to_vector();
  auto input_tensor = TensorWrapper(input, shape, desc.in_dtype);
  auto output_tensor = TensorWrapper(output, shape, desc.out_dtype, amax_out, scale, scale_inv);

  nvte_quantize(input_tensor.data(), output_tensor.data(), stream);
}

Error_Type QuantizeFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type amax_buf,
                       Buffer_Type scale_buf, Buffer_Type scale_inv_buf, Result_Type output_buf,
                       Result_Type amax_out_buf) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  auto *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  auto *scale_inv = reinterpret_cast<float *>(scale_inv_buf.untyped_data());

  auto *output = output_buf->untyped_data();
  auto *amax_out = reinterpret_cast<float *>(amax_out_buf->untyped_data());
  NVTE_CHECK(amax == amax_out, "amax not bound to amax_out in TE/JAX Quantize primitive.");

  auto input_dims = input_buf.dimensions();
  std::vector<size_t> shape(input_dims.begin(), input_dims.end());
  auto input_tensor = TensorWrapper(input, shape, in_dtype);
  auto output_tensor = TensorWrapper(output, shape, out_dtype, amax_out, scale, scale_inv);

  nvte_quantize(input_tensor.data(), output_tensor.data(), stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(QuantizeHandler, QuantizeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>(),     // amax_out
                              FFI_CudaGraph_Traits);

void Dequantize(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *amax = reinterpret_cast<float *>(buffers[1]);
  auto *scale = reinterpret_cast<float *>(buffers[2]);
  auto *scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto *output = buffers[4];

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);

  auto shape = desc.shape.to_vector();
  auto input_tensor = TensorWrapper(input, shape, desc.in_dtype, amax, scale, scale_inv);
  auto output_tensor = TensorWrapper(output, shape, desc.out_dtype);

  nvte_dequantize(input_tensor.data(), output_tensor.data(), stream);
}

Error_Type DequantizeFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type amax_buf,
                         Buffer_Type scale_buf, Buffer_Type scale_inv_buf, Result_Type output_buf) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  auto *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  auto *scale_inv = reinterpret_cast<float *>(scale_inv_buf.untyped_data());

  auto *output = output_buf->untyped_data();

  auto input_dims = input_buf.dimensions();
  std::vector<size_t> shape(input_dims.begin(), input_dims.end());
  auto input_tensor = TensorWrapper(input, shape, in_dtype, amax, scale, scale_inv);
  auto output_tensor = TensorWrapper(output, shape, out_dtype);

  nvte_dequantize(input_tensor.data(), output_tensor.data(), stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DequantizeHandler, DequantizeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>(),     // output
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
