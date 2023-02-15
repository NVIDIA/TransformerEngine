/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/script.h>
#include "extensions.h"

namespace {
  transformer_engine::DType reverse_map_dtype(int64_t dtype) {
    if (dtype >= 0 && dtype < static_cast<int64_t>(transformer_engine::DType::kNumTypes)) {
      return static_cast<transformer_engine::DType>(dtype);
    } else {
      NVTE_ERROR("Type not supported.");
    }
  }
}  // namespace


at::Tensor cast_to_fp8_ts(const at::Tensor &input,
                          const at::Tensor &scale,
                          const at::Tensor &amax,
                          const at::Tensor &scale_inv,
                          int64_t fp8_tensor,
                          int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  at::Tensor output =  cast_to_fp8(input,
                                   scale[fp8_tensor],
                                   amax[0][fp8_tensor],
                                   scale_inv[fp8_tensor],
                                   otype_arg);
  return output;
}

at::Tensor cast_from_fp8_ts(const at::Tensor &input,
                            const at::Tensor &scale_inv,
                            int64_t fp8_tensor,
                            int64_t itype,
                            int64_t otype) {
  transformer_engine::DType itype_arg = reverse_map_dtype(itype);
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  at::Tensor output = cast_from_fp8(input,
                                    scale_inv[fp8_tensor],
                                    itype_arg,
                                    otype_arg);
  return output;
}

at::Tensor fp8_gelu_ts(at::Tensor input,
                       at::Tensor scale,
                       at::Tensor amax,
                       at::Tensor scale_inv,
                       int64_t fp8_tensor,
                       int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  at::Tensor output = fp8_gelu(input,
                               scale[fp8_tensor],
                               amax[0][fp8_tensor],
                               scale_inv[fp8_tensor],
                               otype_arg);
  return output;
}

at::Tensor te_gemm_ts(at::Tensor A,
                      at::Tensor A_scale_inverse,
                      int64_t A_fp8_tensor,
                      int64_t A_type,
                      int64_t transa,
                      at::Tensor B,
                      at::Tensor B_scale_inverse,
                      int64_t B_fp8_tensor,
                      int64_t B_type,
                      int64_t transb,
                      at::Tensor D,
                      at::Tensor D_scale,
                      int64_t D_type,
                      at::Tensor D_amax,
                      at::Tensor bias,
                      int64_t bias_type,
                      at::Tensor pre_gelu_out,
                      int64_t grad,
                      at::Tensor workspace,
                      int64_t workspaceSize,
                      int64_t accumulate,
                      int64_t use_split_accumulator) {
  // cast inputs to types accepted by te_gemm
  transformer_engine::DType A_type_arg = reverse_map_dtype(A_type);
  bool transa_arg = static_cast<bool>(transa);
  transformer_engine::DType B_type_arg = reverse_map_dtype(B_type);
  bool transb_arg = static_cast<bool>(transb);
  transformer_engine::DType D_type_arg = reverse_map_dtype(D_type);
  transformer_engine::DType bias_type_arg = reverse_map_dtype(bias_type);
  bool grad_arg = static_cast<bool>(grad);
  size_t workspaceSize_arg = static_cast<size_t>(workspaceSize);
  bool accumulate_arg = static_cast<bool>(accumulate);
  bool use_split_accumulator_arg = static_cast<bool>(use_split_accumulator);

  if (A_scale_inverse.numel())
    A_scale_inverse = A_scale_inverse[A_fp8_tensor];

  if (B_scale_inverse.numel())
    B_scale_inverse = B_scale_inverse[B_fp8_tensor];

  te_gemm(A,
          A_scale_inverse,
          A_type_arg,
          transa_arg,
          B,
          B_scale_inverse,
          B_type_arg,
          transb_arg,
          D,
          D_scale,
          D_type_arg,
          D_amax,
          bias,
          bias_type_arg,
          pre_gelu_out,
          grad_arg,
          workspace,
          workspaceSize_arg,
          accumulate_arg,
          use_split_accumulator_arg);
  return D;
}

at::Tensor layernorm_fwd_fp8_inf_ts(const at::Tensor &input,
                                    const at::Tensor &weight,
                                    const at::Tensor &bias,
                                    double eps,
                                    at::Tensor scale,
                                    at::Tensor amax,
                                    at::Tensor scale_inv,
                                    int64_t fp8_tensor,
                                    int64_t otype,
                                    const bool zero_centered_gamma) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  float eps_float = static_cast<float>(eps);

  at::Tensor output = layernorm_fwd_fp8_inf(input,
                                            weight,
                                            bias,
                                            eps_float,
                                            scale,
                                            amax,
                                            scale_inv,
                                            otype_arg,
                                            zero_centered_gamma);

  return output;
}

at::Tensor layernorm_fwd_inf_ts(const at::Tensor &input,
                                const at::Tensor &weight,
                                const at::Tensor &bias,
                                double eps,
                                const bool zero_centered_gamma) {
  float eps_float = static_cast<float>(eps);

  at::Tensor output = layernorm_fwd_inf(input,
                                        weight,
                                        bias,
                                        eps_float,
                                        zero_centered_gamma);

  return output;
}

TORCH_LIBRARY(tex_ts, m) {
  m.def("cast_to_fp8_ts", &cast_to_fp8_ts);
  m.def("cast_from_fp8_ts", &cast_from_fp8_ts);
  m.def("fp8_gelu_ts", &fp8_gelu_ts);
  m.def("te_gemm_ts", &te_gemm_ts);
  m.def("layernorm_fwd_fp8_inf_ts", &layernorm_fwd_fp8_inf_ts);
  m.def("layernorm_fwd_inf_ts", &layernorm_fwd_inf_ts);
}
