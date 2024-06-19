/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "jax/csrc/extensions.h"

namespace transformer_engine {
namespace jax {

pybind11::bytes PackCustomCallCommonDescriptor(const std::vector<size_t> &shape, DType in_dtype,
                                               DType out_dtype, size_t act_enum) {
  CustomCallCommonDescriptor desc{};
  desc.shape.from_vector(shape);
  desc.in_dtype = in_dtype;
  desc.out_dtype = out_dtype;
  desc.act_enum = act_enum;
  return PackOpaque(desc);
}

pybind11::bytes PackCustomCallCommonWkDescriptor(const std::vector<size_t> &shape,
                                                 const std::vector<size_t> &wkshape, DType in_dtype,
                                                 DType out_dtype, DType wk_dtype, size_t act_enum) {
  CustomCallCommonWkDescriptor desc{};
  desc.shape.from_vector(shape);
  desc.wkshape.from_vector(wkshape);
  desc.in_dtype = in_dtype;
  desc.out_dtype = out_dtype;
  desc.wk_dtype = wk_dtype;
  desc.act_enum = act_enum;
  return PackOpaque(desc);
}

pybind11::bytes PackCustomCallNormDescriptor(
    size_t batch_size, size_t hidden_size, size_t wkspace_size, size_t barrier_size,
    const std::vector<size_t> &dgamma_part_shape, const std::vector<size_t> &dbeta_part_shape,
    DType x_dtype, DType w_dtype, DType wkspace_dtype, DType barrier_dtype, DType dgamma_part_dtype,
    DType dbeta_part_dtype, bool zero_centered_gamma, float eps, int sm_margin) {
  CustomCallNormDescriptor desc{};
  desc.batch_size = batch_size;
  desc.hidden_size = hidden_size;
  desc.wkspace_size = wkspace_size;
  desc.barrier_size = barrier_size;
  desc.dgamma_part_shape.from_vector(dgamma_part_shape);
  desc.dbeta_part_shape.from_vector(dbeta_part_shape);
  desc.x_dtype = x_dtype;
  desc.w_dtype = w_dtype;
  desc.wkspace_dtype = wkspace_dtype;
  desc.barrier_dtype = barrier_dtype;
  desc.dgamma_part_dtype = dgamma_part_dtype;
  desc.dbeta_part_dtype = dbeta_part_dtype;
  desc.zero_centered_gamma = zero_centered_gamma;
  desc.eps = eps;
  desc.sm_margin = sm_margin;
  return PackOpaque(desc);
}

pybind11::bytes PackCustomCallSoftmaxDescriptor(size_t batch_size, size_t padding_size,
                                                size_t head_dim, size_t q_seqlen, size_t k_seqlen,
                                                DType dtype, float scale_factor) {
  return PackOpaque(SoftmaxDescriptor{batch_size, padding_size, head_dim, q_seqlen, k_seqlen, dtype,
                                      scale_factor});
}

pybind11::bytes PackCustomCallFusedAttnDescriptor(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    size_t wkspace_size, float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, DType wkspace_dtype,
    bool is_training) {
  return PackOpaque(CustomCallFusedAttnDescriptor{
      input_batch, bias_batch, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, bias_heads,
      head_dim, wkspace_size, scaling_factor, dropout_probability, bias_type, mask_type, qkv_layout,
      dtype, wkspace_dtype, is_training});
}

}  // namespace jax
}  // namespace transformer_engine
