/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

NVTETensor nvte_create_tensor(void *dptr,
                              const NVTEShape shape,
                              const NVTEDType dtype,
                              float *amax,
                              float *scale,
                              float *scale_inv) {
  transformer_engine::Tensor *ret = new transformer_engine::Tensor;
  ret->data.dptr = dptr;
  ret->data.shape = std::vector<size_t>(shape.data, shape.data + shape.ndim);
  ret->data.dtype = static_cast<transformer_engine::DType>(dtype);
  ret->amax.dptr = amax;
  ret->scale.dptr = scale;
  ret->scale_inv.dptr = scale_inv;
  return ret;
}

void nvte_destroy_tensor(NVTETensor tensor) {
  if (tensor == nullptr) return;
  auto *t = reinterpret_cast<transformer_engine::Tensor *>(tensor);
  delete t;
}

NVTEDType nvte_tensor_type(const NVTETensor tensor) {
  return static_cast<NVTEDType>(
          reinterpret_cast<const transformer_engine::Tensor*>(tensor)->data.dtype);
}

NVTEShape nvte_tensor_shape(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor*>(tensor);
  NVTEShape ret;
  ret.data = t.data.shape.data();
  ret.ndim = t.data.shape.size();
  return ret;
}

void *nvte_tensor_data(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor*>(tensor);
  return t.data.dptr;
}

float *nvte_tensor_amax(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor*>(tensor);
  NVTE_CHECK(t.amax.dtype == transformer_engine::DType::kFloat32,
             "Tensor's amax must have Float32 type!");
  return reinterpret_cast<float*>(t.amax.dptr);
}

float *nvte_tensor_scale(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor*>(tensor);
  NVTE_CHECK(t.scale.dtype == transformer_engine::DType::kFloat32,
             "Tensor's scale must have Float32 type!");
  return reinterpret_cast<float*>(t.scale.dptr);
}

float *nvte_tensor_scale_inv(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor*>(tensor);
  NVTE_CHECK(t.scale_inv.dtype == transformer_engine::DType::kFloat32,
             "Tensor's inverse of scale must have Float32 type!");
  return reinterpret_cast<float*>(t.scale_inv.dptr);
}

void nvte_tensor_pack_create(NVTETensorPack* pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
     pack->tensors[i] = reinterpret_cast<NVTETensor>(new transformer_engine::Tensor);
  }
}

void nvte_tensor_pack_destroy(NVTETensorPack* pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
     auto *t = reinterpret_cast<transformer_engine::Tensor*>(pack->tensors[i]);
     delete t;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

namespace transformer_engine {

size_t typeToSize(const transformer_engine::DType type) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
        return TypeInfo<T>::size;
    );  // NOLINT(*)
}

bool is_fp8_dtype(const transformer_engine::DType t) {
  return t == transformer_engine::DType::kFloat8E4M3 ||
         t == transformer_engine::DType::kFloat8E5M2;
}

void CheckInputTensor(const Tensor &t, const std::string &name) {
  const DType type = t.data.dtype;
  if (is_fp8_dtype(type)) {
    // FP8 input needs to have scale_inv
    NVTE_CHECK(t.scale_inv.dptr != nullptr,
               "FP8 input " + name + " must have inverse of scale.");
    NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32);
    NVTE_CHECK(t.scale_inv.shape == std::vector<size_t>{ 1 });
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr,
               "Scale is not supported for non-FP8 input " + name + ".");
    NVTE_CHECK(t.amax.dptr == nullptr,
               "Amax is not supported for non-FP8 input " + name + ".");
    NVTE_CHECK(t.scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 input " + name + ".");
  }
  NVTE_CHECK(t.data.dptr != nullptr,
             "Input " + name + " is not allocated!");
}

void CheckOutputTensor(const Tensor &t, const std::string &name, bool allow_empty) {
  const DType type = t.data.dtype;
  if (is_fp8_dtype(type)) {
    // FP8 output needs to have scale, amax and scale_inv
    NVTE_CHECK(t.amax.dptr != nullptr,
               "FP8 output " + name + " must have amax tensor.");
    NVTE_CHECK(t.amax.dtype == DType::kFloat32);
    NVTE_CHECK(t.amax.shape == std::vector<size_t>{ 1 });
    NVTE_CHECK(t.scale_inv.dptr != nullptr,
               "FP8 output " + name + " must have scale.");
    NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32);
    NVTE_CHECK(t.scale_inv.shape == std::vector<size_t>{ 1 });
    NVTE_CHECK(t.scale.dptr != nullptr,
               "FP8 output " + name + " must have inverse of scale.");
    NVTE_CHECK(t.scale.dtype == DType::kFloat32);
    NVTE_CHECK(t.scale.shape == std::vector<size_t>{ 1 });
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr,
               "Scale is not supported for non-FP8 output " + name + ".");
    NVTE_CHECK(t.amax.dptr == nullptr,
               "Amax is not supported for non-FP8 output " + name + ".");
    NVTE_CHECK(t.scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 output " + name + ".");
  }

  if (!allow_empty) {
    NVTE_CHECK(t.data.dptr != nullptr,
               "Output " + name + " is not allocated!");
  }
}

NVTETensor TensorWrapper::data() const noexcept {
  return tensor_;
}

const NVTEShape TensorWrapper::shape() const noexcept {
  if (tensor_ == nullptr) return NVTEShape{nullptr, 0};
  return nvte_tensor_shape(tensor_);
}

DType TensorWrapper::dtype() const noexcept {
  if (tensor_ == nullptr) return DType::kNumTypes;
  return static_cast<DType>(nvte_tensor_type(tensor_));
}

void * TensorWrapper::dptr() const noexcept {
  if (tensor_ == nullptr) return nullptr;
  return nvte_tensor_data(tensor_);
}

float * TensorWrapper::amax() const noexcept {
  if (tensor_ == nullptr) return nullptr;
  return nvte_tensor_amax(tensor_);
}

float * TensorWrapper::scale() const noexcept {
  if (tensor_ == nullptr) return nullptr;
  return nvte_tensor_scale(tensor_);
}

float * TensorWrapper::scale_inv() const noexcept {
  if (tensor_ == nullptr) return nullptr;
  return nvte_tensor_scale_inv(tensor_);
}

size_t TensorWrapper::ndim() const noexcept {
  if (tensor_ == nullptr) return 0;
  return shape().ndim;
}

size_t TensorWrapper::numel() const noexcept {
  if (tensor_ == nullptr) return 0;
  auto shape_ = shape();
  return std::reduce(shape_.data, shape_.data+shape_.ndim, 1, std::multiplies<size_t>{});
}

size_t TensorWrapper::element_size() const noexcept {
  if (tensor_ == nullptr) return 0;
  return typeToSize(dtype());
}

size_t TensorWrapper::bytes() const noexcept {
  if (tensor_ == nullptr) return 0;
  return numel() * element_size();
}

}  // namespace transformer_engine
