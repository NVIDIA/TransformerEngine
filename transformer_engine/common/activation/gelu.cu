/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include "../utils.cuh"
#include "../common.h"
#include <cstdlib>
#include <../util/vectorized_pointwise.h>

namespace transformer_engine {

namespace detail {

struct GELUParam {};

__device__ inline fp32 gelu(fp32 value, const GELUParam &) {
  return value * (0.5F + 0.5F * tanhf(value * (0.79788456F + 0.03567741F * value * value)));
}

}

void gelu_cast(const Tensor &input,
               const Tensor &scale,
               Tensor *output,
               Tensor *amax,
               Tensor *scale_inv,
               cudaStream_t stream) {
  NVTE_CHECK(input.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.shape == output->shape, "Input and output shapes must match.");
  const size_t tot_elts = input.shape[1] * input.shape[0];

  NVTE_CHECK(amax->shape == std::vector<size_t>{ 1 }, "AMAX tensor must have 1 element.");
  NVTE_CHECK(amax->dtype == DType::kFloat32, "AMAX tensor must have Float32 type.");
  NVTE_CHECK(scale.shape == std::vector<size_t>{ 1 }, "Scale tensor must have 1 element.");
  NVTE_CHECK(scale.dtype == DType::kFloat32, "Scale tensor must have Float32 type.");
  NVTE_CHECK(scale_inv->shape == std::vector<size_t>{ 1 },
      "scale_inv tensor must have 1 element.");
  NVTE_CHECK(scale_inv->dtype == DType::kFloat32, "scale_inv tensor must have Float32 type.");

  NVTE_CHECK(input.dptr != nullptr, "Input is not allocated.");
  NVTE_CHECK(scale.dptr != nullptr, "Scale is not allocated.");
  NVTE_CHECK(output->dptr != nullptr, "Output is not allocated.");
  NVTE_CHECK(amax->dptr != nullptr, "AMAX tensor is not allocated.");
  NVTE_CHECK(scale_inv->dptr != nullptr, "scale_inv tensor is not allocated.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      VectorizedUnaryKernelLauncher<nvec, detail::GELUParam, detail::gelu>(
        reinterpret_cast<const IType*>(input.dptr),
        reinterpret_cast<OType*>(output->dptr),
        reinterpret_cast<const fp32*>(scale.dptr),
        reinterpret_cast<fp32*>(scale_inv->dptr),
        reinterpret_cast<fp32*>(amax->dptr),
        tot_elts,
        {},
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_gelu(const NVTETensor input,
               NVTETensor output,
               const NVTETensor scale,
               NVTETensor amax,
               NVTETensor scale_inv,
               cudaStream_t stream) {
  using namespace transformer_engine;
  gelu_cast(*reinterpret_cast<const Tensor*>(input),
            *reinterpret_cast<const Tensor*>(scale),
            reinterpret_cast<Tensor*>(output),
            reinterpret_cast<Tensor*>(amax),
            reinterpret_cast<Tensor*>(scale_inv),
            stream);
}
