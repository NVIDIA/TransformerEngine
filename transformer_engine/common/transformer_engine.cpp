/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <iostream>

#include "common.h"

namespace transformer_engine {

size_t typeToSize(const DType type) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
                                     return TypeInfo<T>::size;);  // NOLINT(*)
}

bool is_fp8_dtype(const DType t) { return t == DType::kFloat8E4M3 || t == DType::kFloat8E5M2; }

std::string to_string(const DType type) {
  switch (type) {
    case DType::kByte:
      return "Byte";
    case DType::kBFloat16:
      return "BFloat16";
    case DType::kFloat16:
      return "Float16";
    case DType::kFloat32:
      return "Float32";
    case DType::kFloat8E4M3:
      return "Float8E4M3";
    case DType::kFloat8E5M2:
      return "Float8E5M2";
    case DType::kFloat8E8M0:
      return "Float8E8M0";
    case DType::kInt32:
      return "Int32";
    case DType::kInt64:
      return "Int64";
    default:
      return concat_strings("Invalid type ", static_cast<int>(type));
  }
}

std::string to_string(const NVTEScalingMode &mode) {
  switch (mode) {
    case NVTE_DELAYED_TENSOR_SCALING:
      return "Delayed Tensor Scaling";
    case NVTE_MXFP8_1D_SCALING:
      return "MXFP8 1D Scaling";
    case NVTE_INVALID_SCALING:
      return "Invalid Scaling";
  }
  return "Invalid Scaling";
}

void CheckNoopTensor(const Tensor &t, const std::string &name) {
  if (t.data.dptr != nullptr) {
    NVTE_CHECK(t.numel() == 1, "Expected 1 element for ", name, " noop, but found ", t.numel(),
               ".");
    NVTE_CHECK(t.data.dtype == DType::kFloat32, "Found wrong dtype for ", name,
               " noop. Expected kFloat32.");
  }
}

void CheckScaleTensorShape(const Tensor &t) {
  NVTE_CHECK(t.scaling_mode != NVTE_INVALID_SCALING, "Invalid scaling mode!");
  if (is_tensor_scaling(t.scaling_mode)) {
    // per-tensor scaling
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.numel() == 1, "Tensor has invalid scale_inv shape (expected (1), got ",
                 t.scale_inv.shape, ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.numel() == 1,
                 "Tensor has invalid columnwise_scale_inv shape (expected (1), got ",
                 t.columnwise_scale_inv.shape, ")");
    }
  } else {
    if (t.scaling_mode == NVTE_MXFP8_1D_SCALING) {
      // Need (4, 128) alignment even for e8 scaling factor
      auto block_alignment = std::vector<size_t>{128ul, 4ul};
      size_t expected_x, expected_y, alignment;
      if (t.has_data()) {
        alignment = block_alignment[0];
        expected_x =
            DIVUP(DIVUP(t.flat_first_dim(), static_cast<size_t>(1)), alignment) * alignment;
        alignment = block_alignment[1];
        expected_y =
            DIVUP(DIVUP(t.flat_last_dim(), static_cast<size_t>(32)), alignment) * alignment;
        const auto &expected = std::vector<size_t>{expected_x, expected_y};
        NVTE_CHECK(t.scale_inv.shape == expected, "Tensor has invalid scale_inv shape (expected ",
                   expected, ", got ", t.scale_inv.shape, ")");
      }
      if (t.has_columnwise_data()) {
        alignment = block_alignment[1];
        expected_x =
            DIVUP(DIVUP(t.flat_first_dim(), static_cast<size_t>(32)), alignment) * alignment;
        alignment = block_alignment[0];
        expected_y = DIVUP(DIVUP(t.flat_last_dim(), static_cast<size_t>(1)), alignment) * alignment;
        const auto &expected = std::vector<size_t>{expected_x, expected_y};
        NVTE_CHECK(t.columnwise_scale_inv.shape == expected,
                   "Tensor has invalid columnwise_scale_inv shape (expected ", expected, ", got ",
                   t.columnwise_scale_inv.shape, ")");
      }
    }
  }
}

void CheckInputTensor(const Tensor &t, const std::string &name) {
  const DType type = t.dtype();
  if (is_fp8_dtype(type)) {
    // FP8 input needs to have scale_inv
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.dptr != nullptr, "FP8 scaling factor input ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32 || t.scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor input ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected Float32 or Byte, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.dptr != nullptr, "FP8 scaling factor input ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat32 ||
                     t.columnwise_scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor input ", name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected Float32 or Byte, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr, "Scale is not supported for non-FP8 input ", name);
    NVTE_CHECK(t.amax.dptr == nullptr, "Amax is not supported for non-FP8 input ", name);
    NVTE_CHECK(t.scale_inv.dptr == nullptr, "Scale_inv is not supported for non-FP8 input ", name);
    NVTE_CHECK(t.columnwise_scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 input ", name);
  }
  NVTE_CHECK(t.has_data() || t.has_columnwise_data(), "Input ", name, " is not allocated!");

  CheckScaleTensorShape(t);
}

void CheckOutputTensor(const Tensor &t, const std::string &name, bool allow_empty) {
  const DType type = t.dtype();
  if (is_fp8_dtype(type)) {
    // FP8 output needs to have scale, scale_inv and (if delayed scaling) amax
    if (t.scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
      NVTE_CHECK(t.amax.dptr != nullptr, "FP8 output ", name, " must have amax tensor");
      NVTE_CHECK(t.amax.dtype == DType::kFloat32, "Invalid amax dtype (expected ",
                 to_string(DType::kFloat32), ", got ", to_string(t.amax.dtype), ")");
      NVTE_CHECK(product(t.amax.shape) == 1, "Invalid shape of amax in output ", name,
                 " (expected 1 entry, got shape=", t.amax.shape, ")");
    }
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.dptr != nullptr, "FP8 scaling factor output ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32 || t.scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor output ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected Float32 or Float8E8M0, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.dptr != nullptr, "FP8 scaling factor output ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat32 ||
                     t.columnwise_scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor output ", name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected Float32 or Float8E8M0, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr, "Scale is not supported for non-FP8 output ", name);
    NVTE_CHECK(t.amax.dptr == nullptr, "Amax is not supported for non-FP8 output ", name);
    NVTE_CHECK(t.scale_inv.dptr == nullptr, "Scale_inv is not supported for non-FP8 output ", name);
    NVTE_CHECK(t.columnwise_scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 input ", name);
  }

  if (!allow_empty) {
    NVTE_CHECK(t.has_data() || t.has_columnwise_data(), "Output ", name, " is not allocated!");
  }

  CheckScaleTensorShape(t);
}

}  // namespace transformer_engine

NVTETensor nvte_create_tensor(NVTEScalingMode scaling_mode) {
  transformer_engine::Tensor *ret = new transformer_engine::Tensor;
  ret->scaling_mode = scaling_mode;
  return ret;
}

void nvte_destroy_tensor(NVTETensor tensor) {
  if (tensor == nullptr) return;
  auto *t = reinterpret_cast<transformer_engine::Tensor *>(tensor);
  delete t;
}

NVTEDType nvte_tensor_type(const NVTETensor tensor) {
  if (tensor == nullptr) return kNVTEFloat32;
  return static_cast<NVTEDType>(
      reinterpret_cast<const transformer_engine::Tensor *>(tensor)->dtype());
}

NVTEShape nvte_tensor_shape(const NVTETensor tensor) {
  if (tensor == nullptr) return {nullptr, 0};
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTEShape ret;

  // FP8 tensor keeps shape in rowwise data
  if (t.scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    ret.data = t.data.shape.data();
    ret.ndim = t.data.shape.size();
    return ret;
  }

  // Get shape based on what data is available
  if (t.has_data()) {
    ret.data = t.data.shape.data();
    ret.ndim = t.data.shape.size();
    return ret;
  }
  if (t.has_columnwise_data()) {
    ret.data = t.columnwise_data.shape.data();
    ret.ndim = t.columnwise_data.shape.size();
    return ret;
  }

  // Tensor has no data
  ret.data = t.data.shape.data();
  ret.ndim = t.data.shape.size();
  return ret;
}

NVTEShape nvte_tensor_columnwise_shape(const NVTETensor tensor) {
  if (tensor == nullptr) return {nullptr, 0};
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTEShape ret;
  ret.data = t.columnwise_data.shape.data();
  ret.ndim = t.columnwise_data.shape.size();
  return ret;
}

size_t nvte_tensor_ndim(const NVTETensor tensor) {
  if (tensor == nullptr) return 0;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.data.shape.size();
}

size_t nvte_tensor_size(const NVTETensor tensor, const size_t dim) {
  if (tensor == nullptr) return 0;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTE_CHECK(dim >= 0 && dim < t.data.shape.size(), "Invalid dimension index: ", dim);
  return t.data.shape[dim];
}

size_t nvte_tensor_numel(const NVTETensor tensor) {
  if (tensor == nullptr) return 0;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  size_t numel = 1;
  for (auto size : t.data.shape) {
    numel *= size;
  }
  return numel;
}

size_t nvte_tensor_element_size(const NVTETensor tensor) {
  if (tensor == nullptr) return sizeof(float);
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return transformer_engine::typeToSize(t.data.dtype);
}

void *nvte_tensor_data(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.data.dptr;
}

void *nvte_tensor_columnwise_data(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.columnwise_data.dptr;
}

float *nvte_tensor_amax(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTE_CHECK(t.amax.dtype == transformer_engine::DType::kFloat32,
             "Tensor's amax must have Float32 type!");
  return reinterpret_cast<float *>(t.amax.dptr);
}

float *nvte_tensor_scale(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTE_CHECK(t.scale.dtype == transformer_engine::DType::kFloat32,
             "Tensor's scale must have Float32 type!");
  return reinterpret_cast<float *>(t.scale.dptr);
}

float *nvte_tensor_scale_inv(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return reinterpret_cast<float *>(t.scale_inv.dptr);
}

void *nvte_tensor_columnwise_scale_inv(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.columnwise_scale_inv.dptr;
}

NVTEShape nvte_tensor_scale_inv_shape(const NVTETensor tensor) {
  if (tensor == nullptr) return {nullptr, 0};
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTEShape ret;
  ret.data = t.scale_inv.shape.data();
  ret.ndim = t.scale_inv.shape.size();
  return ret;
}

void nvte_set_tensor_param(NVTETensor *tensor, NVTETensorParam param_name,
                           const NVTEBasicTensor *param) {
  NVTE_CHECK(tensor != nullptr, "Tensor pointer can't be NULL.");
  NVTE_CHECK(*tensor != nullptr, "Tensor is not allocated.");
  auto &t = *reinterpret_cast<transformer_engine::Tensor *>(*tensor);
  switch (param_name) {
    case kNVTERowwiseData:
      t.data = *param;
      break;
    case kNVTEColumnwiseData:
      t.columnwise_data = *param;
      break;
    case kNVTEScale:
      t.scale = *param;
      break;
    case kNVTEAmax:
      t.amax = *param;
      break;
    case kNVTERowwiseScaleInv:
      t.scale_inv = *param;
      break;
    case kNVTEColumnwiseScaleInv:
      t.columnwise_scale_inv = *param;
      break;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEBasicTensor nvte_get_tensor_param(const NVTETensor tensor, NVTETensorParam param_name) {
  if (tensor == nullptr) {
    return {nullptr, kNVTEFloat32, {nullptr, 0}};
  }
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  switch (param_name) {
    case kNVTERowwiseData:
      return t.data;
    case kNVTEColumnwiseData:
      return t.columnwise_data;
    case kNVTEScale:
      return t.scale;
    case kNVTEAmax:
      return t.amax;
    case kNVTERowwiseScaleInv:
      return t.scale_inv;
    case kNVTEColumnwiseScaleInv:
      return t.columnwise_scale_inv;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEScalingMode nvte_tensor_scaling_mode(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.scaling_mode;
}

void nvte_tensor_pack_create(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    pack->tensors[i] = reinterpret_cast<NVTETensor>(new transformer_engine::Tensor);
  }
}

void nvte_tensor_pack_destroy(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    auto *t = reinterpret_cast<transformer_engine::Tensor *>(pack->tensors[i]);
    delete t;
  }
}

void nvte_zero_tensor(const NVTETensor tensor, cudaStream_t stream) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  // Zero out tensor data if allocated
  if (t.data.dptr != nullptr) {
    size_t size_in_bytes = nvte_tensor_element_size(tensor) * nvte_tensor_numel(tensor);
    cudaMemsetAsync(t.data.dptr, 0, size_in_bytes, stream);
  }
  // Set amax to 0 if allocated
  if (t.amax.dptr != nullptr) {
    float zero = 0.0f;
    cudaMemcpyAsync(t.amax.dptr, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
  }
  cudaStreamSynchronize(stream);
}
