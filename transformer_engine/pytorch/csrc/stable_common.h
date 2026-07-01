/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_STABLE_COMMON_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_STABLE_COMMON_H_

// Ensure CUDA-specific APIs are available from PyTorch's shim headers
#ifndef USE_CUDA
#define USE_CUDA
#endif

// PyTorch Stable ABI headers
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

// CUDA headers
#include <cuda_runtime.h>

// Transformer Engine C API headers
#include <transformer_engine/transformer_engine.h>

#include <vector>

#include "common/util/logging.h"

namespace transformer_engine::pytorch::stable {

using torch::headeronly::ScalarType;

// ============================================================================
// DType converter (ScalarType -> TE DType)
// ============================================================================

inline transformer_engine::DType GetTransformerEngineDType(ScalarType t) {
  switch (t) {
    case ScalarType::Float8_e4m3fn:
      return transformer_engine::DType::kFloat8E4M3;
    case ScalarType::Float8_e5m2:
      return transformer_engine::DType::kFloat8E5M2;
    case ScalarType::Half:
      return transformer_engine::DType::kFloat16;
    case ScalarType::Float:
      return transformer_engine::DType::kFloat32;
    case ScalarType::BFloat16:
      return transformer_engine::DType::kBFloat16;
    case ScalarType::Bool:
    case ScalarType::Byte:
      return transformer_engine::DType::kByte;
    case ScalarType::Short:
      return transformer_engine::DType::kInt16;
    case ScalarType::Int:
      return transformer_engine::DType::kInt32;
    case ScalarType::Long:
      return transformer_engine::DType::kInt64;
    default:
      NVTE_ERROR("Invalid ScalarType (", static_cast<int>(t), ").");
  }
}

// ============================================================================
// CUDA stream utility
// ============================================================================

inline cudaStream_t getCurrentCUDAStreamRaw(int32_t device_index = -1) {
  if (device_index < 0) {
    device_index = torch::stable::accelerator::getCurrentDeviceIndex();
  }
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

// ============================================================================
// Shape utility
// ============================================================================

inline std::vector<size_t> getStableTensorShape(const torch::stable::Tensor& t) {
  auto sizes = t.sizes();
  std::vector<size_t> shape;
  shape.reserve(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    shape.push_back(static_cast<size_t>(sizes[i]));
  }
  return shape;
}

// ============================================================================
// TensorWrapper construction from stable::Tensor
// ============================================================================

inline transformer_engine::TensorWrapper makeTransformerEngineTensor(
    const torch::stable::Tensor& tensor) {
  transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
  std::vector<size_t> shape = getStableTensorShape(tensor);
  return transformer_engine::TensorWrapper(tensor.data_ptr(), shape, dtype);
}

// ============================================================================
// Tensor allocation via stable ABI
// ============================================================================

inline torch::stable::Tensor allocateStableTensor(const std::vector<int64_t>& shape,
                                                  ScalarType dtype, int32_t device_index = -1) {
  if (device_index < 0) {
    device_index = torch::stable::accelerator::getCurrentDeviceIndex();
  }
  torch::headeronly::IntHeaderOnlyArrayRef size_ref(shape.data(), shape.size());
  torch::stable::Device device(torch::headeronly::DeviceType::CUDA, device_index);
  return torch::stable::empty(size_ref, dtype,
                              std::nullopt,  // layout
                              device,
                              std::nullopt,  // pin_memory
                              std::nullopt   // memory_format
  );
}

// ============================================================================
// Input validation helpers
// ============================================================================

inline void check_fp16_bf16(const torch::stable::Tensor& t, const char* name) {
  auto st = t.scalar_type();
  NVTE_CHECK(st == ScalarType::Half || st == ScalarType::BFloat16, name,
             ": only fp16 and bf16 are supported");
}

}  // namespace transformer_engine::pytorch::stable

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_STABLE_COMMON_H_
