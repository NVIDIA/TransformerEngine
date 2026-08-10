/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_COMPAT_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_COMPAT_H_

#include <cuda_runtime.h>

#ifdef NVTE_WITH_TORCH_STABLE
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>
#else
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#endif

/* Compatibility layer for the incremental migration to the torch stable ABI.
 * Migrated code is written against this surface, which is restricted to what
 * torch::stable provides. With NVTE_WITH_TORCH_STABLE it maps to torch::stable
 * (requires torch >= 2.14); without it (the default) it maps to the full torch
 * ABI, keeping support for older torch versions intact. */
namespace transformer_engine::pytorch::torch_compat {

#ifdef NVTE_WITH_TORCH_STABLE
using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
#else
using Tensor = at::Tensor;
using ScalarType = at::ScalarType;
#endif

inline Tensor contiguous(const Tensor &tensor) {
#ifdef NVTE_WITH_TORCH_STABLE
  return torch::stable::contiguous(tensor);
#else
  return tensor.contiguous();
#endif
}

inline cudaStream_t getCurrentCUDAStream() {
#ifdef NVTE_WITH_TORCH_STABLE
  return static_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(
          torch::stable::accelerator::getCurrentDeviceIndex())
          .nativeHandle());
#else
  return at::cuda::getCurrentCUDAStream();
#endif
}

}  // namespace transformer_engine::pytorch::torch_compat

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_COMPAT_H_
