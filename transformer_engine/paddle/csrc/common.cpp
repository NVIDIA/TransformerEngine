/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"

namespace transformer_engine {
namespace paddle_ext {

TensorWrapper MakeNvteTensor(const void *data_ptr, const std::vector<size_t> &shape,
                             const DType type) {
    return TensorWrapper(const_cast<void *>(data_ptr), shape, type);
}

TensorWrapper MakeNvteTensor(void *data_ptr, const NVTEShape &shape, const DType type) {
    return TensorWrapper(data_ptr, shape, type);
}

TensorWrapper MakeNvteTensor(void *data_ptr, const std::vector<size_t> &shape, const DType type,
                             void *amax_ptr, void *scale_ptr, void *scale_inv_ptr) {
    return TensorWrapper(data_ptr, shape, type, reinterpret_cast<float *>(amax_ptr),
                         reinterpret_cast<float *>(scale_ptr),
                         reinterpret_cast<float *>(scale_inv_ptr));
}

TensorWrapper MakeNvteTensor(paddle::Tensor &tensor) {  // NOLINT
    return MakeNvteTensor(tensor.data(), GetShapeArray(tensor), Paddle2NvteDType(tensor.dtype()));
}

TensorWrapper MakeNvteTensor(const paddle::Tensor &tensor) {
    return MakeNvteTensor(const_cast<void *>(tensor.data()), GetShapeArray(tensor),
                          Paddle2NvteDType(tensor.dtype()));
}

paddle::Tensor AllocateSpace(const NVTEShape &shape, const DType type, const paddle::Place &place,
                             bool init_to_zeros) {
    auto size = shape.ndim;
    if (size == 2 && init_to_zeros) {
        return paddle::zeros(
            {static_cast<int64_t>(shape.data[0]), static_cast<int64_t>(shape.data[1])},
            Nvte2PaddleDType(type), place);
    } else if (size == 2) {
        return paddle::empty(
            {static_cast<int64_t>(shape.data[0]), static_cast<int64_t>(shape.data[1])},
            Nvte2PaddleDType(type), place);
    } else if (size == 1 && init_to_zeros) {
        return paddle::zeros({static_cast<int64_t>(shape.data[0])}, Nvte2PaddleDType(type), place);
    } else if (size == 1) {
        return paddle::empty({static_cast<int64_t>(shape.data[0])}, Nvte2PaddleDType(type), place);
    }
    NVTE_CHECK(false, "Should never reach here! func: AllocateSpace");
}

}  // namespace paddle_ext
}  // namespace transformer_engine
