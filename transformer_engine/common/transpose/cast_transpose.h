/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_TRANSPOSE_CAST_TRANSPOSE_H_
#define TRANSFORMER_ENGINE_COMMON_TRANSPOSE_CAST_TRANSPOSE_H_

#include "../common.h"

namespace transformer_engine::detail {

void cast_transpose(const Tensor &input, const Tensor &noop, Tensor *output_, cudaStream_t stream);

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ComputeType, typename ParamOP,
          ComputeType (*OP)(ComputeType, const ParamOP &)>
void cast_transpose_fused(const Tensor &input, const Tensor *act_input, Tensor *output,
                          Tensor *dbias, Tensor *workspace, cudaStream_t stream);

template <typename ComputeType, typename ParamOP, ComputeType (*OP1)(ComputeType, const ParamOP &),
          ComputeType (*OP2)(ComputeType, const ParamOP &)>
void dgated_act_cast_transpose(const Tensor &input, const Tensor &gated_act_input, Tensor *output,
                               cudaStream_t stream);

}  // namespace transformer_engine::detail

#endif  // TRANSFORMER_ENGINE_COMMON_TRANSPOSE_CAST_TRANSPOSE_H_
