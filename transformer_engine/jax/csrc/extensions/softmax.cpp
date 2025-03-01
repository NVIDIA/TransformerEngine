/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/softmax.h"

#include "extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

void ScaledSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                          size_t opaque_len) {
  auto *input = buffers[0];
  auto *output = buffers[1];

  const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
  auto shape = std::vector<size_t>{desc.batch_size, desc.head_dim, desc.q_seqlen, desc.k_seqlen};
  auto dtype = desc.dtype;

  auto input_tensor = TensorWrapper(input, shape, dtype);
  auto output_tensor = TensorWrapper(output, shape, dtype);

  nvte_scaled_softmax_forward(input_tensor.data(), output_tensor.data(), desc.scale_factor, stream);
}

void ScaledSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                           size_t opaque_len) {
  auto *grad_output = buffers[0];
  auto *softmax_output = buffers[1];
  auto *dgrad = buffers[2];

  const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
  auto shape = std::vector<size_t>{desc.batch_size, desc.head_dim, desc.q_seqlen, desc.k_seqlen};
  auto dtype = desc.dtype;

  auto grad_output_tensor = TensorWrapper(grad_output, shape, dtype);
  auto softmax_output_tensor = TensorWrapper(softmax_output, shape, dtype);
  auto dgrad_tensor = TensorWrapper(dgrad, shape, dtype);

  nvte_scaled_softmax_backward(grad_output_tensor.data(), softmax_output_tensor.data(),
                               dgrad_tensor.data(), desc.scale_factor, stream);
}

void ScaledMaskedSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                                size_t opaque_len) {
  auto *input = buffers[0];
  auto *mask = buffers[1];
  auto *output = buffers[2];

  const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
  auto io_shape = std::vector<size_t>{desc.batch_size, desc.head_dim, desc.q_seqlen, desc.k_seqlen};
  auto mask_shape = std::vector<size_t>{desc.padding_size, 1, desc.q_seqlen, desc.k_seqlen};
  auto dtype = desc.dtype;

  auto input_tensor = TensorWrapper(input, io_shape, dtype);
  // Mask would be casted to uint8_t
  auto mask_tensor = TensorWrapper(mask, mask_shape, DType::kByte);
  auto output_tensor = TensorWrapper(output, io_shape, dtype);

  nvte_scaled_masked_softmax_forward(input_tensor.data(), mask_tensor.data(), output_tensor.data(),
                                     desc.scale_factor, stream);
}

void ScaledMaskedSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                                 size_t opaque_len) {
  // The backward of ScaledMaskedSoftmax is equivalent to ScaledSoftmax.
  ScaledSoftmaxBackward(stream, buffers, opaque, opaque_len);
}

void ScaledUpperTriangMaskedSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                                           size_t opaque_len) {
  auto *input = buffers[0];
  auto *output = buffers[1];

  const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
  auto attn_batch = desc.batch_size * desc.head_dim;
  auto shape = std::vector<size_t>{attn_batch, desc.q_seqlen, desc.k_seqlen};
  auto dtype = desc.dtype;

  auto input_tensor = TensorWrapper(input, shape, dtype);

  auto output_tensor = TensorWrapper(output, shape, dtype);

  nvte_scaled_upper_triang_masked_softmax_forward(input_tensor.data(), output_tensor.data(),
                                                  desc.scale_factor, stream);
}

void ScaledUpperTriangMaskedSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                                            size_t opaque_len) {
  auto *grad_output = buffers[0];
  auto *softmax_output = buffers[1];
  auto *dgrad = buffers[2];

  const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
  auto attn_batch = desc.batch_size * desc.head_dim;
  auto shape = std::vector<size_t>{attn_batch, desc.q_seqlen, desc.k_seqlen};
  auto dtype = desc.dtype;

  auto grad_output_tensor = TensorWrapper(grad_output, shape, dtype);
  auto softmax_output_tensor = TensorWrapper(softmax_output, shape, dtype);
  auto dgrad_tensor = TensorWrapper(dgrad, shape, dtype);

  nvte_scaled_upper_triang_masked_softmax_backward(grad_output_tensor.data(),
                                                   softmax_output_tensor.data(),
                                                   dgrad_tensor.data(), desc.scale_factor, stream);
}

#define SOFTMAX_COMMON_BLOCK(tensor_buf)                                      \
  auto dtype = convert_ffi_datatype_to_te_dtype((tensor_buf).element_type()); \
  auto tensor_dims = (tensor_buf).dimensions();                               \
  auto tensor_ranks = tensor_dims.size();                                     \
  auto batch_size = product(tensor_dims, 0, tensor_ranks - 3);                \
  auto head_dim = product(tensor_dims, tensor_ranks - 3, tensor_ranks - 2);   \
  auto q_seqlen = product(tensor_dims, tensor_ranks - 2, tensor_ranks - 1);   \
  auto k_seqlen = product(tensor_dims, tensor_ranks - 1, tensor_ranks);       \
  float scale_factor = static_cast<float>(scale_factor_);

#define SOFTMAX_FORWARD_COMMON_BLOCK                      \
  auto *input = input_buf.untyped_data();                 \
  auto *output = output_buf->untyped_data();              \
  auto input_tensor = TensorWrapper(input, shape, dtype); \
  auto output_tensor = TensorWrapper(output, shape, dtype);

Error_Type ScaledSoftmaxForwardFFI(cudaStream_t stream, Buffer_Type input_buf,
                                   Result_Type output_buf, double scale_factor_) {
  SOFTMAX_COMMON_BLOCK(input_buf);
  auto shape = std::vector<size_t>{batch_size, head_dim, q_seqlen, k_seqlen};
  SOFTMAX_FORWARD_COMMON_BLOCK;
  nvte_scaled_softmax_forward(input_tensor.data(), output_tensor.data(), scale_factor, stream);
  return ffi_with_cuda_error_check();
}

Error_Type ScaledMaskedSoftmaxForwardFFI(cudaStream_t stream, Buffer_Type input_buf,
                                         Buffer_Type mask_buf, Result_Type output_buf,
                                         double scale_factor_) {
  SOFTMAX_COMMON_BLOCK(input_buf);

  // Mask would be casted to uint8_t
  auto *mask = mask_buf.untyped_data();
  auto mask_dims = mask_buf.dimensions();
  auto padding_size = product(mask_dims, mask_dims.size() - 3);
  auto mask_shape = std::vector<size_t>{padding_size, 1, q_seqlen, k_seqlen};
  auto mask_tensor = TensorWrapper(mask, mask_shape, DType::kByte);

  auto shape = std::vector<size_t>{batch_size, head_dim, q_seqlen, k_seqlen};
  SOFTMAX_FORWARD_COMMON_BLOCK;
  nvte_scaled_masked_softmax_forward(input_tensor.data(), mask_tensor.data(), output_tensor.data(),
                                     scale_factor, stream);
  return ffi_with_cuda_error_check();
}

Error_Type ScaledUpperTriangMaskedSoftmaxForwardFFI(cudaStream_t stream, Buffer_Type input_buf,
                                                    Result_Type output_buf, double scale_factor_) {
  SOFTMAX_COMMON_BLOCK(input_buf);
  auto shape = std::vector<size_t>{batch_size * head_dim, q_seqlen, k_seqlen};
  SOFTMAX_FORWARD_COMMON_BLOCK;
  nvte_scaled_upper_triang_masked_softmax_forward(input_tensor.data(), output_tensor.data(),
                                                  scale_factor, stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ScaledSoftmaxForwardHandler, ScaledSoftmaxForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Ret<Buffer_Type>()      // output
                                  .Attr<double>("scale_factor"),
                              FFI_CudaGraph_Traits);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ScaledMaskedSoftmaxForwardHandler, ScaledMaskedSoftmaxForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // mask
                                  .Ret<Buffer_Type>()      // output
                                  .Attr<double>("scale_factor"),
                              FFI_CudaGraph_Traits);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ScaledUpperTriangMaskedSoftmaxForwardHandler,
                              ScaledUpperTriangMaskedSoftmaxForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Ret<Buffer_Type>()      // output
                                  .Attr<double>("scale_factor"),
                              FFI_CudaGraph_Traits);

#define SOFTMAX_BACKWARD_COMMON_BLOCK                                       \
  auto *grad_output = grad_output_buf.untyped_data();                       \
  auto *softmax_output = softmax_output_buf.untyped_data();                 \
  auto *dgrad = dgrad_buf->untyped_data();                                  \
  auto grad_output_tensor = TensorWrapper(grad_output, shape, dtype);       \
  auto softmax_output_tensor = TensorWrapper(softmax_output, shape, dtype); \
  auto dgrad_tensor = TensorWrapper(dgrad, shape, dtype);

Error_Type ScaledSoftmaxBackwardFFI(cudaStream_t stream, Buffer_Type grad_output_buf,
                                    Buffer_Type softmax_output_buf, Result_Type dgrad_buf,
                                    double scale_factor_) {
  SOFTMAX_COMMON_BLOCK(grad_output_buf);
  auto shape = std::vector<size_t>{batch_size, head_dim, q_seqlen, k_seqlen};
  SOFTMAX_BACKWARD_COMMON_BLOCK;
  nvte_scaled_softmax_backward(grad_output_tensor.data(), softmax_output_tensor.data(),
                               dgrad_tensor.data(), scale_factor, stream);
  return ffi_with_cuda_error_check();
}

Error_Type ScaledUpperTriangMaskedSoftmaxBackwardFFI(cudaStream_t stream,
                                                     Buffer_Type grad_output_buf,
                                                     Buffer_Type softmax_output_buf,
                                                     Result_Type dgrad_buf, double scale_factor_) {
  SOFTMAX_COMMON_BLOCK(grad_output_buf);
  auto shape = std::vector<size_t>{batch_size * head_dim, q_seqlen, k_seqlen};
  SOFTMAX_BACKWARD_COMMON_BLOCK;
  nvte_scaled_upper_triang_masked_softmax_backward(grad_output_tensor.data(),
                                                   softmax_output_tensor.data(),
                                                   dgrad_tensor.data(), scale_factor, stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ScaledSoftmaxBackwardHandler, ScaledSoftmaxBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // grad_output
                                  .Arg<Buffer_Type>()      // softmax_output
                                  .Ret<Buffer_Type>()      // dgrad
                                  .Attr<double>("scale_factor"),
                              FFI_CudaGraph_Traits);

// The backward of ScaledMaskedSoftmax is equivalent to ScaledSoftmax
XLA_FFI_DEFINE_HANDLER_SYMBOL(ScaledMaskedSoftmaxBackwardHandler, ScaledSoftmaxBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // grad_output
                                  .Arg<Buffer_Type>()      // softmax_output
                                  .Ret<Buffer_Type>()      // dgrad
                                  .Attr<double>("scale_factor"),
                              FFI_CudaGraph_Traits);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ScaledUpperTriangMaskedSoftmaxBackwardHandler,
                              ScaledUpperTriangMaskedSoftmaxBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // grad_output
                                  .Arg<Buffer_Type>()      // softmax_output
                                  .Ret<Buffer_Type>()      // dgrad
                                  .Attr<double>("scale_factor"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
