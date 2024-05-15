/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                                                              const transformer_engine::DType dtype,
                                                              at::Tensor amax,
                                                              const at::Tensor scale,
                                                              at::Tensor scale_inv,
                                                              int64_t fp8_idx) {
    if (tensor.numel() == 0)
        return transformer_engine::TensorWrapper(nullptr, std::vector<size_t>{0}, dtype);

    if (amax.numel())
        NVTE_CHECK(amax.scalar_type() == at::kFloat);
    if (scale.numel())
        NVTE_CHECK(scale.scalar_type() == at::kFloat);
    if (scale_inv.numel()) {
        NVTE_CHECK(scale_inv.scalar_type() == at::kFloat);
        if (fp8_idx >= 0) scale_inv = scale_inv[fp8_idx];
    }

    std::vector<size_t> shape;
    for (auto s : tensor.sizes())
        shape.push_back(s);

    return makeTransformerEngineTensor(tensor.data_ptr(), shape, dtype,
                                       getDataPtr(amax), getDataPtr(scale), getDataPtr(scale_inv));
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor,
                                                              at::Tensor amax,
                                                              const at::Tensor scale,
                                                              at::Tensor scale_inv,
                                                              int64_t fp8_idx) {
    transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
    return makeTransformerEngineTensor(tensor, dtype, amax, scale, scale_inv, fp8_idx);
}

size_t product(const std::vector<size_t> &shape) {
    size_t ret = 1;
    for (auto s : shape) {
        ret *= s;
    }
    return ret;
}


at::Tensor allocateSpace(const std::vector<size_t>& shape,
                         const transformer_engine::DType type,
                         bool init_to_zeros) {
    std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    c10::IntArrayRef ar_shape(shape_int64);
    if (init_to_zeros) {
        return at::zeros(ar_shape, at::CUDA(GetATenDType(type)));
    } else {
        return at::empty(ar_shape, at::CUDA(GetATenDType(type)));
    }
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

void *getDataPtr(at::Tensor t) {
    if (t.numel() > 0) {
        return t.data_ptr();
    } else {
        return nullptr;
    }
}
