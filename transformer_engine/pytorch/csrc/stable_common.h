/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_STABLE_COMMON_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_STABLE_COMMON_H_

// PyTorch Stable ABI headers
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>

// CUDA headers
#include <cuda_runtime.h>

// Transformer Engine C API headers
#include <transformer_engine/transformer_engine.h>

#include <cassert>
#include <cstring>
#include <vector>

#include "common/util/logging.h"

namespace transformer_engine::pytorch::stable {

using torch::headeronly::ScalarType;

// ============================================================================
// DType <-> ScalarType converters
// ============================================================================

inline ScalarType GetStableScalarType(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt16:
      return ScalarType::Short;
    case transformer_engine::DType::kInt32:
      return ScalarType::Int;
    case transformer_engine::DType::kInt64:
      return ScalarType::Long;
    case transformer_engine::DType::kFloat32:
      return ScalarType::Float;
    case transformer_engine::DType::kFloat16:
      return ScalarType::Half;
    case transformer_engine::DType::kBFloat16:
      return ScalarType::BFloat16;
    case transformer_engine::DType::kByte:
      return ScalarType::Byte;
    case transformer_engine::DType::kFloat8E4M3:
      return ScalarType::Float8_e4m3fn;
    case transformer_engine::DType::kFloat8E5M2:
      return ScalarType::Float8_e5m2;
    case transformer_engine::DType::kFloat8E8M0:
      return ScalarType::Byte;  // e8m0 not natively supported
    default:
      NVTE_ERROR("Invalid DType (", static_cast<int>(t), ").");
  }
}

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
      return transformer_engine::DType::kByte;
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
// CUDA stream utilities
// ============================================================================

/// Get the current CUDA stream as a raw cudaStream_t.
/// Uses the stable ABI's aoti_torch_get_current_cuda_stream.
inline cudaStream_t getCurrentCUDAStreamRaw(int32_t device_index = -1) {
  if (device_index < 0) {
    device_index = torch::stable::accelerator::getCurrentDeviceIndex();
  }
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

// ============================================================================
// Device properties
// ============================================================================

/// Get SM count for the given CUDA device (or current device if -1).
/// Replaces at::cuda::getCurrentDeviceProperties()->multiProcessorCount.
inline int getSMCount(int device_index = -1) {
  if (device_index < 0) {
    device_index = static_cast<int>(
        torch::stable::accelerator::getCurrentDeviceIndex());
  }
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_index);
  NVTE_CHECK(err == cudaSuccess, "cudaGetDeviceProperties failed: ",
             cudaGetErrorString(err));
  return prop.multiProcessorCount;
}

// ============================================================================
// Shape utilities
// ============================================================================

/// Convert stable tensor sizes (int64_t array) to vector<size_t>.
inline std::vector<size_t> getStableTensorShape(
    const torch::stable::Tensor& t) {
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

/// Create a TensorWrapper from a torch::stable::Tensor.
/// Extracts data_ptr, shape, and dtype.
inline transformer_engine::TensorWrapper makeTransformerEngineTensor(
    const torch::stable::Tensor& tensor) {
  transformer_engine::DType dtype =
      GetTransformerEngineDType(tensor.scalar_type());
  std::vector<size_t> shape = getStableTensorShape(tensor);
  return transformer_engine::TensorWrapper(tensor.data_ptr(), shape, dtype);
}

/// Create a TensorWrapper from raw components (same as unstable version).
inline transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const std::vector<size_t>& shape,
    const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}

// ============================================================================
// Tensor allocation via stable ABI
// ============================================================================

/// Allocate an empty tensor on CUDA via the stable ABI.
inline torch::stable::Tensor allocateStableTensor(
    const std::vector<int64_t>& shape,
    ScalarType dtype,
    int32_t device_index = -1) {
  if (device_index < 0) {
    device_index = torch::stable::accelerator::getCurrentDeviceIndex();
  }
  torch::headeronly::IntHeaderOnlyArrayRef size_ref(shape.data(), shape.size());
  torch::stable::Device device(torch::headeronly::DeviceType::CUDA,
                               device_index);
  return torch::stable::empty(
      size_ref,
      dtype,
      std::nullopt,  // layout
      device,
      std::nullopt,  // pin_memory
      std::nullopt   // memory_format
  );
}

/// Allocate an empty tensor on CUDA, using TE DType.
inline torch::stable::Tensor allocateStableTensor(
    const std::vector<int64_t>& shape,
    transformer_engine::DType te_dtype,
    int32_t device_index = -1) {
  return allocateStableTensor(shape, GetStableScalarType(te_dtype),
                              device_index);
}

/// Allocate a zero-filled tensor on CUDA via the stable ABI.
inline torch::stable::Tensor allocateStableTensorZeros(
    const std::vector<int64_t>& shape,
    ScalarType dtype,
    int32_t device_index = -1) {
  auto t = allocateStableTensor(shape, dtype, device_index);
  torch::stable::zero_(t);
  return t;
}

/// Allocate a zero-filled tensor on CUDA, using TE DType.
inline torch::stable::Tensor allocateStableTensorZeros(
    const std::vector<int64_t>& shape,
    transformer_engine::DType te_dtype,
    int32_t device_index = -1) {
  return allocateStableTensorZeros(shape, GetStableScalarType(te_dtype),
                                   device_index);
}

// ============================================================================
// TensorWrapper construction with quantization metadata
// ============================================================================

/// Build a TensorWrapper with rowwise quantization metadata.
/// The output_data tensor holds the quantized data.
/// amax, scale, scale_inv are optional quantization parameters.
/// If scale_inv_dtype is -1, defaults to kFloat32 (use kFloat8E8M0=10 for
/// MXFP8, kFloat8E4M3=8 for NVFP4).
inline transformer_engine::TensorWrapper makeQuantizedTensorWrapper(
    const torch::stable::Tensor& output_data,
    transformer_engine::DType te_dtype,
    const std::vector<size_t>& shape,
    const std::optional<torch::stable::Tensor>& amax,
    const std::optional<torch::stable::Tensor>& scale,
    const std::optional<torch::stable::Tensor>& scale_inv,
    NVTEScalingMode scaling_mode) {
  TensorWrapper out(scaling_mode);
  out.set_rowwise_data(output_data.data_ptr(), te_dtype, shape);

  const std::vector<size_t> scalar_shape{1};
  if (amax.has_value() && amax->numel() > 0) {
    out.set_amax(amax->data_ptr(), DType::kFloat32, scalar_shape);
  }
  if (scale.has_value() && scale->numel() > 0) {
    out.set_scale(scale->data_ptr(), DType::kFloat32, scalar_shape);
  }
  if (scale_inv.has_value() && scale_inv->numel() > 0) {
    // Determine scale_inv dtype from scaling mode
    DType si_dtype = DType::kFloat32;
    if (scaling_mode == NVTE_MXFP8_1D_SCALING) {
      si_dtype = DType::kFloat8E8M0;
    } else if (scaling_mode == NVTE_NVFP4_1D_SCALING) {
      si_dtype = DType::kFloat8E4M3;
    }
    auto si_shape = getStableTensorShape(scale_inv.value());
    out.set_rowwise_scale_inv(scale_inv->data_ptr(), si_dtype, si_shape);
  }
  return out;
}

/// Helper to run the two-phase workspace pattern for any NVTE function.
/// The callable should have signature: void(NVTETensor workspace)
/// First call queries workspace size, second call runs the kernel.
template <typename Fn>
inline void runWithWorkspace(Fn&& fn, int32_t device_idx) {
  TensorWrapper workspace;
  fn(workspace.data());

  auto ws_shape = workspace.shape();
  auto ws_dtype = workspace.dtype();
  if (ws_shape.ndim > 0 && workspace.numel() > 0) {
    auto workspace_data = allocateStableTensor(
        std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
        ws_dtype, device_idx);
    workspace = makeTransformerEngineTensor(
        workspace_data.data_ptr(),
        std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
        ws_dtype);
  }

  fn(workspace.data());
}

}  // namespace transformer_engine::pytorch::stable

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_STABLE_COMMON_H_
