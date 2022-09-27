/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/cast.h>
#include "../common.h"
#include "../utils.cuh"
#include "../util/vectorized_pointwise.h"

namespace transformer_engine {

namespace detail {

struct Empty {};

__device__ inline fp32 identity(fp32 value, const Empty&) {
  return value;
}

struct DequantizeParam {
  const fp32 *scale_inv;
};

__device__ inline fp32 dequantize_func(fp32 value, const DequantizeParam &param) {
  return value * (*(param.scale_inv));
}

}  // namespace detail

void fp8_quantize(const Tensor &input,
                  const Tensor &scale,
                  Tensor *output,
                  Tensor *amax,
                  Tensor *scale_inv,
                  cudaStream_t stream) {
    NVTE_CHECK(input.dtype != DType::kFloat8E4M3 &&
               input.dtype != DType::kFloat8E5M2,
               "Input must be in higher precision.");
    NVTE_CHECK(input.dptr != nullptr, "Input is not allocated.");

    NVTE_CHECK(output->dptr != nullptr, "Output is not allocated.");
    NVTE_CHECK(output->dtype == DType::kFloat8E4M3 ||
               output->dtype == DType::kFloat8E5M2,
               "Output must have FP8 type.");
    NVTE_CHECK(output->shape == input.shape, "Input and output shapes need to match.");

    NVTE_CHECK(scale.dptr != nullptr, "Scale is not allocated.");
    NVTE_CHECK(scale.dtype == DType::kFloat32, "Scale must have FP32 type.");
    NVTE_CHECK(scale.shape == std::vector<size_t>{ 1 }, "Scale must have 1 element.");

    NVTE_CHECK(amax->dptr != nullptr, "AMAX is not allocated.");
    NVTE_CHECK(amax->dtype == DType::kFloat32, "AMAX must have FP32 type.");
    NVTE_CHECK(amax->shape == std::vector<size_t>{ 1 }, "AMAX must have 1 element.");

    NVTE_CHECK(scale_inv->dptr != nullptr, "Inverted scale is not allocated.");
    NVTE_CHECK(scale_inv->dtype == DType::kFloat32, "Inverted scale must have FP32 type.");
    NVTE_CHECK(scale_inv->shape == std::vector<size_t>{ 1 }, "Inverted scale must have 1 element.");

    const size_t N = product(input.shape);
    TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.dtype, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output->dtype, OType,
          constexpr int nvec = 32 / sizeof(IType);
          VectorizedUnaryKernelLauncher<nvec, detail::Empty, detail::identity>(
                    reinterpret_cast<const IType*>(input.dptr),
                    reinterpret_cast<OType*>(output->dptr),
                    reinterpret_cast<const fp32*>(scale.dptr),
                    reinterpret_cast<fp32*>(scale_inv->dptr),
                    reinterpret_cast<fp32*>(amax->dptr),
                    N,
                    {},
                    stream);
        );  // NOLINT(*)
    );  // NOLINT(*)
}

void fp8_dequantize(const Tensor &input,
                    const Tensor &scale_inv,
                    Tensor *output,
                    cudaStream_t stream) {
    NVTE_CHECK(input.dtype == DType::kFloat8E4M3 ||
               input.dtype == DType::kFloat8E5M2,
               "Input must have FP8 type.");
    NVTE_CHECK(input.dptr != nullptr, "Input is not allocated.");

    NVTE_CHECK(output->dptr != nullptr, "Output is not allocated.");
    NVTE_CHECK(output->dtype != DType::kFloat8E4M3 &&
               output->dtype != DType::kFloat8E5M2,
               "Output must be in higher precision.");
    NVTE_CHECK(output->shape == input.shape, "Input and output shapes need to match.");

    NVTE_CHECK(scale_inv.dptr != nullptr, "Inverted scale is not allocated.");
    NVTE_CHECK(scale_inv.dtype == DType::kFloat32, "Inverted scale must have FP32 type.");
    NVTE_CHECK(scale_inv.shape == std::vector<size_t>{ 1 }, "Inverted scale must have 1 element.");

    const size_t N = product(input.shape);
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(input.dtype, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(output->dtype, OType,
          constexpr int nvec = 32 / sizeof(OType);
          detail::DequantizeParam p;
          p.scale_inv = reinterpret_cast<const fp32*>(scale_inv.dptr);
          VectorizedUnaryKernelLauncher<nvec, detail::DequantizeParam, detail::dequantize_func>(
                    reinterpret_cast<const IType*>(input.dptr),
                    reinterpret_cast<OType*>(output->dptr),
                    nullptr,
                    nullptr,
                    nullptr,
                    N,
                    p,
                    stream);
        );  // NOLINT(*)
    );  // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_fp8_quantize(const NVTETensor input,
                       const NVTETensor scale,
                       NVTETensor output,
                       NVTETensor amax,
                       NVTETensor scale_inv,
                       cudaStream_t stream) {
  using namespace transformer_engine;
  fp8_quantize(*reinterpret_cast<const Tensor*>(input),
               *reinterpret_cast<const Tensor*>(scale),
               reinterpret_cast<Tensor*>(output),
               reinterpret_cast<Tensor*>(amax),
               reinterpret_cast<Tensor*>(scale_inv),
               stream);
}

void nvte_fp8_dequantize(const NVTETensor input,
                         const NVTETensor scale_inv,
                         NVTETensor output,
                         cudaStream_t stream) {
  using namespace transformer_engine;
  fp8_dequantize(*reinterpret_cast<const Tensor*>(input),
                 *reinterpret_cast<const Tensor*>(scale_inv),
                 reinterpret_cast<Tensor*>(output),
                 stream);
}
