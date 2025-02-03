/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/cast_transpose_noop.h>
#include <transformer_engine/comm_gemm_overlap.h>
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/fused_rope.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/normalization.h>
#include <transformer_engine/padding.h>
#include <transformer_engine/permutation.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/softmax.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "c10/util/ArrayRef.h"
#include "common/util/logging.h"

namespace transformer_engine::pytorch {

// Each tensor here is shape (N, ) holding all scaling
// data for a single FP8 block, e.g. LayerNormLinear
class FP8TensorMeta {
 public:
  at::Tensor scale;
  at::Tensor scale_inv;
  at::Tensor amax_history;
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8FwdTensors {
  GEMM1_INPUT = 0,
  GEMM1_WEIGHT = 1,
  GEMM1_OUTPUT = 2,
  GEMM2_INPUT = 3,
  GEMM2_WEIGHT = 4,
  GEMM2_OUTPUT = 5,
  GEMM3_INPUT = 6,
  GEMM3_WEIGHT = 7,
  GEMM3_OUTPUT = 8
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors {
  GRAD_OUTPUT1 = 0,
  GRAD_INPUT1 = 1,
  GRAD_OUTPUT2 = 2,
  GRAD_INPUT2 = 3,
  GRAD_OUTPUT3 = 4,
  GRAD_INPUT3 = 5
};

class Quantizer {
 public:
  virtual NVTEScalingMode get_scaling_mode() const = 0;

  virtual void set_quantization_params(TensorWrapper* tensor) const = 0;

  virtual std::pair<TensorWrapper, py::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<at::Tensor> rowwise_data = std::nullopt) const = 0;

  virtual ~Quantizer() = default;

  bool rowwise_usage = true;
  bool columnwise_usage = true;
  bool internal = false;
  py::handle quantizer;

 protected:
  explicit Quantizer(const py::handle& quantizer);
};

class NoneQuantizer : public Quantizer {
 public:
  explicit NoneQuantizer(const py::handle& quantizer) : Quantizer(quantizer) {}

  NVTEScalingMode get_scaling_mode() const override { return NVTE_DELAYED_TENSOR_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override {}

  std::pair<TensorWrapper, py::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<at::Tensor> rowwise_data = std::nullopt) const override;
};

class Float8Quantizer : public Quantizer {
 public:
  at::Tensor scale;
  at::Tensor scale_inv;
  at::Tensor amax;
  DType dtype;

  explicit Float8Quantizer(const py::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_DELAYED_TENSOR_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, py::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<at::Tensor> rowwise_data = std::nullopt) const override;
};

class MXFP8Quantizer : public Quantizer {
 public:
  DType dtype;

  explicit MXFP8Quantizer(const py::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_MXFP8_1D_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, py::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<at::Tensor> rowwise_data = std::nullopt) const override;
};

std::unique_ptr<Quantizer> convert_quantizer(py::handle quantizer);

std::vector<size_t> getTensorShape(at::Tensor t);

transformer_engine::DType getTransformerEngineFP8Type(bool e4m3_if_hybrid,
                                                      const std::string& fp8_recipe);

inline at::ScalarType GetATenDType(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt32:
      return torch::kInt32;
    case transformer_engine::DType::kInt64:
      return torch::kInt64;
    case transformer_engine::DType::kFloat32:
      return at::kFloat;
    case transformer_engine::DType::kFloat16:
      return at::kHalf;
    case transformer_engine::DType::kBFloat16:
      return at::kBFloat16;
    case transformer_engine::DType::kByte:
      return at::kByte;
    case transformer_engine::DType::kFloat8E4M3:
      return at::kFloat8_e4m3fn;
    case transformer_engine::DType::kFloat8E5M2:
      return at::kFloat8_e5m2;
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(at::ScalarType t) {
  switch (t) {
    case at::kFloat8_e4m3fn:
      return transformer_engine::DType::kFloat8E4M3;
    case at::kFloat8_e5m2:
      return transformer_engine::DType::kFloat8E5M2;
    case at::kHalf:
      return transformer_engine::DType::kFloat16;
    case at::kFloat:
      return transformer_engine::DType::kFloat32;
    case at::kBFloat16:
      return transformer_engine::DType::kBFloat16;
    case at::kBool:
      return transformer_engine::DType::kByte;
    case torch::kByte:
      return transformer_engine::DType::kByte;
    case torch::kInt32:
      return transformer_engine::DType::kInt32;
    case torch::kInt64:
      return transformer_engine::DType::kInt64;
    default:
      std::cout << "Type: " << static_cast<int>(t) << std::endl;
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(int DType_value) {
  return static_cast<transformer_engine::DType>(DType_value);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const std::vector<size_t>& shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const std::vector<size_t>& shape, const transformer_engine::DType type,
    void* amax_ptr, void* scale_ptr, void* scale_inv_ptr, std::vector<size_t> scale_inv_shape = {1},
    NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, void* columnwise_data_ptr, const std::vector<size_t>& shape,
    const std::vector<size_t>& columnwise_shape, const transformer_engine::DType type,
    void* amax_ptr, void* scale_ptr, void* scale_inv_ptr, void* columnwise_scale_inv_ptr,
    const std::vector<size_t>& scale_inv_shape = {1},
    const std::vector<size_t>& columnwise_scale_inv_shape = {1},
    NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const NVTEShape& shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor);

TensorWrapper makeTransformerEngineTensor(py::handle tensor, py::handle quantizer);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    at::Tensor tensor, at::Tensor amax, const at::Tensor scale, at::Tensor scale_inv,
    NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

template <typename T>
T product(const std::vector<T>& shape);

size_t product(const NVTEShape& shape, size_t begin, size_t end);

std::vector<size_t> nvte_shape_to_vector(const NVTEShape& nvte_shape);

at::Tensor allocateSpace(const std::vector<size_t>& shape, const transformer_engine::DType type,
                         bool init_to_zeros);

at::Tensor allocateSpace(const NVTEShape& shape, const transformer_engine::DType type,
                         bool init_to_zeros = false);

at::Tensor allocateTorchTensor(int M, int N, transformer_engine::DType dtype);

at::Tensor allocateTorchTensor(int M, transformer_engine::DType dtype);

void* getDataPtr(at::Tensor tensor, int offset = 0);

std::vector<size_t> convertShape(const NVTEShape& shape);

int roundup(const int value, const int multiple);

}  // namespace transformer_engine::pytorch

namespace std {
template <typename T>
string to_string(const vector<T>& vec) {
  string ret = "[";
  for (const auto& val : vec) {
    ret += to_string(val) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}

// Torch shape -> string
template <typename T>
string to_string(const c10::ArrayRef<T>& vec) {
  string ret = "[";
  for (const auto& val : vec) {
    ret += to_string(val) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}

inline string to_string(const NVTEShape& s) {
  string ret = "[";
  for (size_t i = 0; i < s.ndim; ++i) {
    ret += to_string(s.data[i]) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}
}  // namespace std

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
