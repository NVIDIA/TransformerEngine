/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/fused_rope.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/layer_norm.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/rmsnorm.h>
#include <transformer_engine/softmax.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/util/logging.h"

namespace transformer_engine {

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

}  // namespace transformer_engine

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
    case transformer_engine::DType::kFloat8E4M3:
    case transformer_engine::DType::kFloat8E5M2:
      return at::kByte;
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(at::ScalarType t) {
  switch (t) {
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
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(int DType_value) {
  return static_cast<transformer_engine::DType>(DType_value);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const std::vector<size_t>& shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const std::vector<size_t>& shape,
                                                              const transformer_engine::DType type,
                                                              void* amax_ptr, void* scale_ptr,
                                                              void* scale_inv_ptr);

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const NVTEShape& shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor);

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor, at::Tensor amax,
                                                              const at::Tensor scale,
                                                              at::Tensor scale_inv);

size_t product(const std::vector<size_t>& shape);

at::Tensor allocateSpace(const std::vector<size_t>& shape, const transformer_engine::DType type,
                         bool init_to_zeros);

at::Tensor allocateSpace(const NVTEShape& shape, const transformer_engine::DType type,
                         bool init_to_zeros = false);

at::Tensor allocateTorchTensor(int M, int N, transformer_engine::DType dtype);

at::Tensor allocateTorchTensor(int M, transformer_engine::DType dtype);

void* getDataPtr(at::Tensor tensor, int offset = 0);

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
