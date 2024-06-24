/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/cast.h>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"

namespace transformer_engine {

namespace detail {

struct Empty {};

__device__ inline fp32 identity(fp32 value, const Empty &) { return value; }

struct DequantizeParam {
  const fp32 *scale_inv;
};

__device__ inline fp32 dequantize_func(fp32 value, const DequantizeParam &param) {
  return value * (*(param.scale_inv));
}

}  // namespace detail

void fp8_quantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  NVTE_CHECK(!is_fp8_dtype(input.data.dtype), "Input must be in higher precision.");

  NVTE_CHECK(is_fp8_dtype(output->data.dtype), "Output must have FP8 type.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OType, constexpr int nvec = 32 / sizeof(IType);
          VectorizedUnaryKernelLauncher<nvec, detail::Empty, detail::identity>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr),
              reinterpret_cast<const fp32 *>(output->scale.dptr),
              reinterpret_cast<fp32 *>(output->amax.dptr), N, {},
              stream););  // NOLINT(*)
  );                      // NOLINT(*)
}

void fp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");
  NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");

  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
          output->data.dtype, OType, constexpr int nvec = 32 / sizeof(OType);
          detail::DequantizeParam p;
          p.scale_inv = reinterpret_cast<const fp32 *>(input.scale_inv.dptr);
          VectorizedUnaryKernelLauncher<nvec, detail::DequantizeParam, detail::dequantize_func>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr), nullptr, nullptr, N, p,
              stream););  // NOLINT(*)
  );                      // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_fp8_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize);
  using namespace transformer_engine;
  fp8_quantize(*reinterpret_cast<const Tensor *>(input), reinterpret_cast<Tensor *>(output),
               stream);
}

void nvte_fp8_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_dequantize);
  using namespace transformer_engine;
  fp8_dequantize(*reinterpret_cast<const Tensor *>(input), reinterpret_cast<Tensor *>(output),
                 stream);
}
