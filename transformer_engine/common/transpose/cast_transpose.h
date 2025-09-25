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

void quantize_transpose_square_blockwise(const SimpleTensor &input, SimpleTensor &scale_inv,
                                         SimpleTensor &scale_inv_t, SimpleTensor &output,
                                         SimpleTensor &output_t, const float epsilon,
                                         const bool return_transpose, const bool pow_2_scale,
                                         const SimpleTensor &noop_tensor, const bool pdl_sync,
                                         const bool pdl_trigger, cudaStream_t stream);

// enum class for rowwise usage
enum class FP8BlockwiseRowwiseOption {
  // No rowwise data, skip rowwise quantization
  NONE,
  // Rowwise data, scales in GEMM format
  ROWWISE_GEMM_READY,
  // Rowwise data, scales in compact format, needs extra processing (padding, transposing) before GEMM
  ROWWISE_COMPACT
};

// enum class for columnwise usage
// For Hopper sm90 with only TN fp8 gemm, there is need to do columnwise transpose when doing 1D block scaling
enum class FP8BlockwiseColumnwiseOption {
  // No columnwise data, skip columnwise quantization
  NONE,
  // Columnwise data transposed from original shape.
  // Scales in GEMM format corresponding to GEMM ingesting transposed column data.
  // On Hopper sm90, GEMM_READY means that columnwise quantization also fuses transpose op
  // On higher sm versions with TN,NT,NN fp8 gemm, GEMM_READY doesn't fuse transpose
  COLUMNWISE_GEMM_READY,
  // Columnwise data in original shape
  // Scales in compact format, needs extra processing (padding, transposing) before GEMM
  COLUMNWISE_COMPACT
};

void quantize_transpose_vector_blockwise(const SimpleTensor &input, SimpleTensor &scale_inv,
                                         SimpleTensor &scale_inv_t, SimpleTensor &output,
                                         SimpleTensor &output_t, const float epsilon,
                                         FP8BlockwiseRowwiseOption rowwise_option,
                                         FP8BlockwiseColumnwiseOption columnwise_option,
                                         const bool pow_2_scale, const SimpleTensor &noop_tensor,
                                         const bool pdl_sync, const bool pdl_trigger,
                                         cudaStream_t stream);

}  // namespace transformer_engine::detail

#endif  // TRANSFORMER_ENGINE_COMMON_TRANSPOSE_CAST_TRANSPOSE_H_
