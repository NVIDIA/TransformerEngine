/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_

#include <torch/extension.h>

#include <optional>
#include <tuple>
#include <vector>

#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace pytorch {

/*! \brief Convert tensor block scales into GEMM swizzled format.
 *
 *  The returned swizzled scales should be kept alive during the GEMM.
 */
std::tuple<std::optional<at::Tensor>, std::optional<at::Tensor>> swizzle_scales_for_gemm(
    TensorWrapper& tensor, bool rowwise_usage, bool columnwise_usage);

/*! \brief Convert multiple tensor block scales into GEMM swizzled format.
 *
 *  The returned swizzled scales should be kept alive during the GEMMs.
 */
std::optional<at::Tensor> multi_tensor_swizzle_scales_for_gemm(std::vector<TensorWrapper>& tensors,
                                                               bool rowwise_usage,
                                                               bool columnwise_usage);

/*! \brief Convert a block scaling tensor to an mxfp8 tensor in-place.
 *
 *  If rowwise==false, the columnwise data will be reinterpreted as
 *  rowwise data to avoid transposing it in memory. Due to differences
 *  in how block scaling and mxfp8 store data, this requires the
 *  calling code to treat the output tensor as having been transposed
 *  in this case.
 *
 *  Returns the swizzled scaling factor of the converted mxfp8 tensor.
 *  The returned swizzled scaling factor tensor should be kept alive
 *  during the GEMM.
 */
at::Tensor convert_block_scaling_to_mxfp8_tensor(TensorWrapper& input, bool rowwise);

}  // namespace pytorch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_UTIL_H_
