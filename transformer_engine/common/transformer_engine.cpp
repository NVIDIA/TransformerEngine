/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include "common.h"

namespace transformer_engine {

size_t typeToSize(const transformer_engine::DType type) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
        return TypeInfo<T>::size;
    );  // NOLINT(*)
}

}  // namespace transformer_engine

NVTETensor nvte_create_tensor(void *dptr,
                              const NVTEShape shape,
                              const NVTEDType dtype) {
  transformer_engine::Tensor *ret = new transformer_engine::Tensor;
  ret->dptr = dptr;
  ret->shape = std::vector<size_t>(shape.data, shape.data + shape.ndim);
  ret->dtype = static_cast<transformer_engine::DType>(dtype);
  return ret;
}

void nvte_destroy_tensor(NVTETensor tensor) {
  if (tensor == nullptr) return;
  auto *t = reinterpret_cast<transformer_engine::Tensor *>(tensor);
  delete t;
}

NVTEDType nvte_tensor_type(const NVTETensor tensor) {
  return static_cast<NVTEDType>(reinterpret_cast<const transformer_engine::Tensor*>(tensor)->dtype);
}

NVTEShape nvte_tensor_shape(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor*>(tensor);
  NVTEShape ret;
  ret.data = t.shape.data();
  ret.ndim = t.shape.size();
  return ret;
}

void *nvte_tensor_data(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor*>(tensor);
  return t.dptr;
}
