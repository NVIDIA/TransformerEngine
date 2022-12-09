/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Handle.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/layer_norm.h>
#include <transformer_engine/logging.h>
#include <transformer_engine/softmax.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

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
enum FP8FwdTensors { GEMM1_INPUT = 0, GEMM1_WEIGHT = 1, GEMM2_INPUT = 2, GEMM2_WEIGHT = 3 };

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors { GRAD_OUTPUT1 = 0, GRAD_OUTPUT2 = 1 };

}  // namespace transformer_engine

transformer_engine::DType getTransformerEngineFP8Type(bool e4m3_if_hybrid,
                                                      const std::string &fp8_recipe);

inline at::ScalarType GetATenDType(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt32:
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
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(int DType_value) {
  return static_cast<transformer_engine::DType>(DType_value);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(void *data_ptr,
                                                              const std::vector<size_t> &shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(void *data_ptr,
                                                              const std::vector<size_t> &shape,
                                                              const transformer_engine::DType type,
                                                              void *amax_ptr, void *scale_ptr,
                                                              void *scale_inv_ptr);

transformer_engine::TensorWrapper makeTransformerEngineTensor(void *data_ptr,
                                                              const NVTEShape &shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor);

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor, at::Tensor amax,
                                                              const at::Tensor scale,
                                                              at::Tensor scale_inv);

size_t product(const std::vector<size_t> &shape);

at::Tensor allocateSpace(const NVTEShape &shape, const transformer_engine::DType type,
                         bool init_to_zeros = false);

at::Tensor allocateTorchTensor(int M, int N, transformer_engine::DType dtype);

at::Tensor allocateTorchTensor(int M, transformer_engine::DType dtype);

void dispatch_layernorm(
    void *input,  // i
    const std::vector<size_t> &input_shape, const transformer_engine::DType input_type,
    void *gamma,  // i
    const std::vector<size_t> &gamma_shape, const transformer_engine::DType gamma_type,
    void *beta,  // i
    const std::vector<size_t> &beta_shape, const transformer_engine::DType beta_type,
    void *scale,  // i
    const std::vector<size_t> &scale_shape, const transformer_engine::DType scale_type,
    const float epsilon,  // i
    void *z,              // o
    const std::vector<size_t> &z_shape, const transformer_engine::DType z_type,
    void *mu,  // o
    const std::vector<size_t> &mu_shape, const transformer_engine::DType mu_type,
    void *rsigma,  // o
    const std::vector<size_t> &rsigma_shape, const transformer_engine::DType rsigma_type,
    void *amax,  // o
    const std::vector<size_t> &amax_shape, const transformer_engine::DType amax_type,
    void *scale_inv,  // o
    const std::vector<size_t> &scale_inv_shape, const transformer_engine::DType scale_inv_type,
    const int multiProcessorCount);

void dispatch_cast_transpose_fusion(
    void *input,  // i
    const std::vector<size_t> &input_shape, const transformer_engine::DType input_type,
    void *scale,  // i
    const std::vector<size_t> &scale_shape, const transformer_engine::DType scale_type,
    void *output_cast,  // o
    const std::vector<size_t> &output_cast_shape, const transformer_engine::DType output_cast_type,
    void *output_transpose,  // o
    const std::vector<size_t> &output_transpose_shape,
    const transformer_engine::DType output_transpose_type,
    void *amax,  // o
    const std::vector<size_t> &amax_shape, const transformer_engine::DType amax_type,
    void *scale_inv,  // o
    const std::vector<size_t> &scale_inv_shape, const transformer_engine::DType scale_inv_type);

void dispatch_gelu(
    void *input,  // i
    const std::vector<size_t> &input_shape, const transformer_engine::DType input_type,
    void *scale,  // i
    const std::vector<size_t> &scale_shape, const transformer_engine::DType scale_type,
    void *output,  // o
    const std::vector<size_t> &output_shape, const transformer_engine::DType output_type,
    void *amax,  // o
    const std::vector<size_t> &amax_shape, const transformer_engine::DType amax_type,
    void *scale_inv,  // o
    const std::vector<size_t> &scale_inv_shape, const transformer_engine::DType scale_inv_type);

void dispatch_transpose(void *input,  // i
                        const std::vector<size_t> &input_shape,
                        const transformer_engine::DType input_type,
                        void *output,  // o
                        const std::vector<size_t> &output_shape,
                        const transformer_engine::DType output_type);

void dispatch_bgrad_cast_transpose_fusion(
    void *input,  // i
    const std::vector<size_t> &input_shape, const transformer_engine::DType input_type,
    void *scale,  // i
    const std::vector<size_t> &scale_shape, const transformer_engine::DType scale_type,
    void *cast_output,  // o
    const std::vector<size_t> &cast_output_shape, const transformer_engine::DType cast_output_type,
    void *transposed_output,  // o
    const std::vector<size_t> &transposed_output_shape,
    const transformer_engine::DType transposed_output_type,
    void *amax,  // o
    const std::vector<size_t> &amax_shape, const transformer_engine::DType amax_type,
    void *dbias,  // o
    const std::vector<size_t> &dbias_shape, const transformer_engine::DType dbias_type,
    void *scale_inv,  // o
    const std::vector<size_t> &scale_inv_shape, const transformer_engine::DType scale_inv_type);

void dispatch_bgrad_dgelu_cast_transpose_fusion(
    void *input,  // i
    const std::vector<size_t> &input_shape, const transformer_engine::DType input_type,
    void *gelu_input,  // i
    const std::vector<size_t> &gelu_input_shape, const transformer_engine::DType gelu_input_type,
    void *scale,  // i
    const std::vector<size_t> &scale_shape, const transformer_engine::DType scale_type,
    void *cast_output,  // o
    const std::vector<size_t> &cast_output_shape, const transformer_engine::DType cast_output_type,
    void *transposed_output,  // o
    const std::vector<size_t> &transposed_output_shape,
    const transformer_engine::DType transposed_output_type,
    void *amax,  // o
    const std::vector<size_t> &amax_shape, const transformer_engine::DType amax_type,
    void *dbias,  // o
    const std::vector<size_t> &dbias_shape, const transformer_engine::DType dbias_type,
    void *scale_inv,  // o
    const std::vector<size_t> &scale_inv_shape, const transformer_engine::DType scale_inv_type);

void dispatch_multi_cast_transpose(
    std::vector<void *> input_dptr_list,  // i
    const std::vector<std::vector<size_t>> &input_shape_list,
    const std::vector<transformer_engine::DType> &input_type_list,
    std::vector<void *> scale_dptr_list,  // i
    const std::vector<std::vector<size_t>> &scale_shape_list,
    const std::vector<transformer_engine::DType> &scale_type_list,
    std::vector<void *> cast_output_dptr_list,  // o
    const std::vector<std::vector<size_t>> &cast_output_shape_list,
    const std::vector<transformer_engine::DType> &cast_output_type_list,
    std::vector<void *> transposed_output_dptr_list,  // o
    const std::vector<std::vector<size_t>> &transposed_output_shape_list,
    const std::vector<transformer_engine::DType> &transposed_output_type_list,
    std::vector<void *> amax_dptr_list,  // o
    const std::vector<std::vector<size_t>> &amax_shape_list,
    const std::vector<transformer_engine::DType> &amax_type_list,
    std::vector<void *> scale_inv_dptr_list,  // o
    const std::vector<std::vector<size_t>> &scale_inv_shape_list,
    const std::vector<transformer_engine::DType> &scale_inv_type_list);

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
