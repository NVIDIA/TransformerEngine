/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/softmax.h"

#include "jax/csrc/extensions.h"

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

}  // namespace jax
}  // namespace transformer_engine
