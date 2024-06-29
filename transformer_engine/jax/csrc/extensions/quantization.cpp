/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "jax/csrc/extensions.h"
#include "transformer_engine/cast.h"

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

  nvte_fp8_quantize(input_tensor.data(), output_tensor.data(), stream);
}

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

  nvte_fp8_dequantize(input_tensor.data(), output_tensor.data(), stream);
}

}  // namespace jax
}  // namespace transformer_engine
