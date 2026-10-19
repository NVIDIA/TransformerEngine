/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "utils.h"

#include <cuda_runtime_api.h>
#include <cudnn_frontend_version.h>

#include <cassert>

#include "common/util/cuda_runtime.h"

namespace transformer_engine {
namespace jax {

int GetCudaRuntimeVersion() {
  int ver = 0;
  NVTE_CHECK_CUDA(cudaRuntimeGetVersion(&ver));
  return ver;
}

size_t GetCudnnRuntimeVersion() { return cudnnGetVersion(); }

size_t GetCudnnFrontendVersion() { return CUDNN_FRONTEND_VERSION; }

int GetDeviceComputeCapability(int gpu_id) { return transformer_engine::cuda::sm_arch(gpu_id); }

}  // namespace jax
}  // namespace transformer_engine
