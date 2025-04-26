/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/util/cuda_runtime.h"
#include "extensions.h"

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }

int get_device_compute_capability(int device_id) {
  return transformer_engine::cuda::sm_arch(device_id);
}
