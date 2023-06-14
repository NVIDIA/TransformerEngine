/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"

namespace transformer_engine {
namespace paddle_ext {

TensorWrapper MakeNvteTensor(void *data_ptr, const std::vector<size_t> &shape, const DType type) {
    return TensorWrapper(data_ptr, shape, type);
}

TensorWrapper MakeNvteTensor(void *data_ptr, const std::vector<size_t> &shape, const DType type,
                             void *amax_ptr, void *scale_ptr, void *scale_inv_ptr) {
    return TensorWrapper(data_ptr, shape, type, reinterpret_cast<float *>(amax_ptr),
                         reinterpret_cast<float *>(scale_ptr),
                         reinterpret_cast<float *>(scale_inv_ptr));
}

TensorWrapper MakeNvteTensor(const paddle::Tensor &tensor) {
    return MakeNvteTensor(const_cast<void *>(tensor.data()), GetShapeArray(tensor),
                          Paddle2NvteDType(tensor.dtype()));
}

}  // namespace paddle_ext
}  // namespace transformer_engine
