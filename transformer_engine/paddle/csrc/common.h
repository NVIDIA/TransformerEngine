/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include <cublasLt.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/layer_norm.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/rmsnorm.h>
#include <transformer_engine/softmax.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>

#include <cstdlib>
#include <vector>

#include "common/util/logging.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"

namespace transformer_engine {
namespace paddle_ext {
// Paddle Tensor Utils
template <typename T>
inline const void *GetDataPtr(const paddle::Tensor &x, int64_t index) {
  if (index < 0 || index >= x.numel()) {
    NVTE_ERROR("Index out of bound");
  }
  return reinterpret_cast<const void *>(x.data<T>() + static_cast<size_t>(index));
}

template <typename T>
inline void *GetDataPtr(paddle::Tensor &x, int64_t index) {  // NOLINT
  if (index < 0 || index >= x.numel()) {
    NVTE_ERROR("Index out of bound");
  }
  return reinterpret_cast<void *>(x.data<T>() + static_cast<size_t>(index));
}

template <typename T>
inline const void *GetOptionalDataPtr(const paddle::optional<paddle::Tensor> &x, int64_t index) {
  return x ? GetDataPtr<T>(*x, index) : nullptr;
}

template <typename T>
inline void *GetOptionalDataPtr(paddle::optional<paddle::Tensor> &x, int64_t index) {  // NOLINT
  return x ? GetDataPtr<T>(*x, index) : nullptr;
}

inline const void *GetOptionalDataPtr(const paddle::optional<paddle::Tensor> &x) {
  return x ? x->data() : nullptr;
}

inline void *GetOptionalDataPtr(paddle::optional<paddle::Tensor> &x) {  // NOLINT
  return x ? x->data() : nullptr;
}

inline std::vector<size_t> GetShapeArray(const paddle::Tensor &x) {
  std::vector<size_t> shapes;
  for (auto dim : x.shape()) {
    shapes.push_back(static_cast<size_t>(dim));
  }
  return shapes;
}

inline std::vector<size_t> GetShapeArray(const paddle::optional<paddle::Tensor> &x) {
  if (x) return GetShapeArray(x.get());
  return {0};
}

paddle::Tensor AllocateSpace(const NVTEShape &shape, const DType type, const paddle::Place &place,
                             bool init_to_zeros = 0);

// DType Utils
inline paddle::DataType Nvte2PaddleDType(DType t) {
  switch (t) {
    case DType::kInt32:
    case DType::kFloat32:
      return paddle::DataType::FLOAT32;
    case DType::kFloat16:
      return paddle::DataType::FLOAT16;
    case DType::kBFloat16:
      return paddle::DataType::BFLOAT16;
    case DType::kByte:
    case DType::kFloat8E4M3:
    case DType::kFloat8E5M2:
      return paddle::DataType::UINT8;
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline DType Paddle2NvteDType(paddle::DataType t) {
  switch (t) {
    case paddle::DataType::FLOAT16:
      return DType::kFloat16;
    case paddle::DataType::FLOAT32:
      return DType::kFloat32;
    case paddle::DataType::BFLOAT16:
      return DType::kBFloat16;
    case paddle::DataType::BOOL:
      return DType::kByte;
    case paddle::DataType::UINT8:
      return DType::kByte;
    case paddle::DataType::INT32:
      return DType::kInt32;
    case paddle::DataType::INT64:
      return DType::kInt64;
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline DType Int2NvteDType(int64_t dtype) {
  if (dtype >= 0 && dtype < static_cast<int64_t>(DType::kNumTypes)) {
    return static_cast<DType>(dtype);
  } else {
    NVTE_ERROR("Type not supported.");
  }
}

// get the fused attention backend
inline NVTE_Fused_Attn_Backend get_fused_attn_backend(
    const transformer_engine::DType q_dtype, const transformer_engine::DType kv_dtype,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    float p_dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim) {
  NVTE_Fused_Attn_Backend fused_attention_backend =
      nvte_get_fused_attn_backend(static_cast<NVTEDType>(q_dtype), static_cast<NVTEDType>(kv_dtype),
                                  qkv_layout, bias_type, attn_mask_type, p_dropout, num_attn_heads,
                                  num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim);
  return fused_attention_backend;
}

// CUDA Utils
class cudaDevicePropertiesManager {
 public:
  static cudaDevicePropertiesManager &Instance() {
    static thread_local cudaDevicePropertiesManager instance;
    return instance;
  }

  int GetMultiProcessorCount() {
    if (!prop_queried_) {
      int device_id;
      NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
      cudaGetDeviceProperties(&prop_, device_id);
      prop_queried_ = true;
    }
    return prop_.multiProcessorCount;
  }

  int GetMajor() {
    if (!prop_queried_) {
      int device_id;
      NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
      cudaGetDeviceProperties(&prop_, device_id);
      prop_queried_ = true;
    }
    return prop_.major;
  }

 private:
  bool prop_queried_ = false;
  cudaDeviceProp prop_;
};

// NVTE Tensor Utils
TensorWrapper MakeNvteTensor(const void *data_ptr, const std::vector<size_t> &shape,
                             const DType type);
TensorWrapper MakeNvteTensor(void *data_ptr, const NVTEShape &shape, const DType type);
TensorWrapper MakeNvteTensor(void *data_ptr, const std::vector<size_t> &shape, const DType type,
                             void *amax_ptr, void *scale_ptr, void *scale_inv_ptr);
TensorWrapper MakeNvteTensor(paddle::Tensor &tensor);  // NOLINT
TensorWrapper MakeNvteTensor(const paddle::Tensor &tensor);

NVTE_QKV_Layout get_nvte_qkv_layout(const std::string &qkv_layout);

}  // namespace paddle_ext
}  // namespace transformer_engine
