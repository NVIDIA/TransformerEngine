/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <cfloat>
#include <limits>
#include <string>

#include "../common.h"
#include "../transpose/cast_transpose.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "cast_kernels.cuh"
#include "dequantize_kernels.cuh"
#include "math.h"
#include "ptx.cuh"
#include "transformer_engine/activation.h"
#include "transformer_engine/transpose.h"


void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = false;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor grad = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, nullptr>(input, grad, nullptr, output,
                                                                     dbias, workspace, stream);
}

void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_noop);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = false;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor grad = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, nullptr>(input, grad, noop, output,
                                                                     dbias, workspace, stream);
}

void nvte_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                         NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = false;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, nullptr>(
      activation_input, input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_dgelu(const NVTETensor input, const NVTETensor activation_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, dgelu<fp32, fp32>>(
      activation_input, input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor activation_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dsilu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, dsilu<fp32, fp32>>(
      activation_input, input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_drelu(const NVTETensor input, const NVTETensor activation_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_drelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, drelu<fp32, fp32>>(
      activation_input, input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_dqgelu(const NVTETensor input, const NVTETensor activation_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dqgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, dqgelu<fp32, fp32>>(
      activation_input, input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_dsrelu(const NVTETensor input, const NVTETensor activation_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dsrelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, dsrelu<fp32, fp32>>(
      activation_input, input, nullptr, output, dbias, workspace, stream);
}

void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dequantize);
  using namespace transformer_engine;
  detail::dequantize_helper(*reinterpret_cast<const Tensor *>(input),
                            reinterpret_cast<Tensor *>(output), stream);
}
