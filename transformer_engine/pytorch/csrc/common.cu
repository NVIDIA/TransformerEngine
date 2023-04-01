/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "transformer_engine/transformer_engine.h"


transformer_engine::DType getTransformerEngineFP8Type(bool e4m3_if_hybrid,
                                                      const std::string &fp8_recipe) {
    // if e4m3 or hybrid + forward
    if ( (fp8_recipe == "E4M3") || ( (fp8_recipe == "HYBRID") && e4m3_if_hybrid ) ) {
        return transformer_engine::DType::kFloat8E4M3;
    }
    return transformer_engine::DType::kFloat8E5M2;
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr,
    const NVTEShape& shape,
    const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}


transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr,
    const std::vector<size_t>& shape,
    const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}


transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor) {
    transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
    std::vector<size_t> shape;

    for (auto s : tensor.sizes()) {
        shape.push_back(s);
    }
    return makeTransformerEngineTensor(tensor.data_ptr(), shape, dtype);
}


transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr,
    const std::vector<size_t>& shape,
    const transformer_engine::DType type,
    void* amax_ptr,
    void* scale_ptr,
    void* scale_inv_ptr) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type,
                                           reinterpret_cast<float*>(amax_ptr),
                                           reinterpret_cast<float*>(scale_ptr),
                                           reinterpret_cast<float*>(scale_inv_ptr));
}


transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor,
                                                              at::Tensor amax,
                                                              const at::Tensor scale,
                                                              at::Tensor scale_inv) {
    transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
    std::vector<size_t> shape;

    for (auto s : tensor.sizes()) {
        shape.push_back(s);
    }
    NVTE_CHECK(amax.scalar_type() == at::kFloat);
    NVTE_CHECK(scale.scalar_type() == at::kFloat);
    NVTE_CHECK(scale_inv.scalar_type() == at::kFloat);

    return makeTransformerEngineTensor(tensor.data_ptr(), shape, dtype,
                                       amax.data_ptr(),
                                       scale.data_ptr(),
                                       scale_inv.data_ptr());
}


size_t product(const std::vector<size_t> &shape) {
    size_t ret = 1;
    for (auto s : shape) {
        ret *= s;
    }
    return ret;
}


at::Tensor allocateSpace(const NVTEShape &shape,
                         const transformer_engine::DType type,
                         bool init_to_zeros) {
    auto size = shape.ndim;
    if (size == 2 && init_to_zeros) {
        return at::zeros({static_cast<int64_t>(shape.data[0]),
                          static_cast<int64_t>(shape.data[1])},
                          at::CUDA(GetATenDType(type)));
    } else if (size == 2) {
        return at::empty({static_cast<int64_t>(shape.data[0]),
                          static_cast<int64_t>(shape.data[1])},
                          at::CUDA(GetATenDType(type)));
    } else if (size == 1 && init_to_zeros) {
        return at::zeros({static_cast<int64_t>(shape.data[0])}, at::CUDA(GetATenDType(type)));
    } else if (size == 1) {
        return at::empty({static_cast<int64_t>(shape.data[0])}, at::CUDA(GetATenDType(type)));
    }
    NVTE_CHECK(false, "Should never reach here! func: allocateSpace");
}


at::Tensor allocateTorchTensor(int M,
                               int N,
                               transformer_engine::DType dtype
) {
    return at::empty({static_cast<int64_t>(M), static_cast<int64_t>(N)},
                     at::CUDA(GetATenDType(dtype)));
}


at::Tensor allocateTorchTensor(int M,
                               transformer_engine::DType dtype
) {
    return at::empty({static_cast<int64_t>(M)},
                     at::CUDA(GetATenDType(dtype)));
}
