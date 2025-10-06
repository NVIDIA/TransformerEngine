/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize_fp8.cuh
 *  \brief FP8 dequantize.
 */

#ifndef TRANSFORMER_ENGINE_DEQUANTIZE_FP8_CUH_
#define TRANSFORMER_ENGINE_DEQUANTIZE_FP8_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/vectorized_pointwise.h"
#include "../../utils.cuh"
#include "../../util/math.h"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
struct DequantizeParam {
  const float *scale_inv;
};

__device__ inline float dequantize_func(float value, const DequantizeParam &param) {
  return value * (*(param.scale_inv));
}

inline void dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          output->data.dtype, OType,

          constexpr int nvec = 32 / sizeof(OType);
          DequantizeParam p;
          p.scale_inv = reinterpret_cast<const fp32 *>(input.scale_inv.dptr);
          VectorizedUnaryKernelLauncher<nvec, DequantizeParam, dequantize_func>(
              reinterpret_cast<const IType *>(input.data.dptr), nullptr,
              reinterpret_cast<OType *>(output->data.dptr), nullptr, nullptr, nullptr, N, p,
              stream););  // NOLINT(*)
  );                      // NOLINT(*)
}
} // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DEQUANTIZE_FP8_CUH_

