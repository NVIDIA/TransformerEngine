/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cuda_fp8.h>
#include <torch/script.h>

#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
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

at::Tensor cast_to_fp8_ts(const at::Tensor &input, const at::Tensor &scale, at::Tensor amax,
                          at::Tensor scale_inv, int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  at::Tensor output =
      cast_to_fp8(input, scale[fp8_tensor], amax[0][fp8_tensor], scale_inv[fp8_tensor], otype_arg);
  return output;
}

at::Tensor cast_to_fp8_noalloc_ts(const at::Tensor &input, const at::Tensor &scale,
                                  at::Tensor output, at::Tensor amax, at::Tensor scale_inv,
                                  int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  cast_to_fp8_noalloc(input, scale[fp8_tensor], output, amax[0][fp8_tensor], scale_inv[fp8_tensor],
                      otype_arg);
  return output;
}

at::Tensor cast_from_fp8_ts(const at::Tensor &input, const at::Tensor &scale_inv,
                            int64_t fp8_tensor, int64_t itype, int64_t otype) {
  transformer_engine::DType itype_arg = reverse_map_dtype(itype);
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  at::Tensor output = cast_from_fp8(input, scale_inv[fp8_tensor], itype_arg, otype_arg);
  return output;
}

at::Tensor gelu_ts(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                   int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  at::Tensor s, a, s_inv;
  if (scale.numel()) {
    s = scale[fp8_tensor];
  } else {
    s = scale;
  }

  if (amax.numel()) {
    a = amax[0][fp8_tensor];
  } else {
    a = amax;
  }

  if (scale_inv.numel()) {
    s_inv = scale_inv[fp8_tensor];
  } else {
    s_inv = scale_inv;
  }

  at::Tensor output = gelu(input, s, a, s_inv, otype_arg);
  return output;
}

at::Tensor relu_ts(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                   int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  at::Tensor s, a, s_inv;
  if (scale.numel()) {
    s = scale[fp8_tensor];
  } else {
    s = scale;
  }

  if (amax.numel()) {
    a = amax[0][fp8_tensor];
  } else {
    a = amax;
  }

  if (scale_inv.numel()) {
    s_inv = scale_inv[fp8_tensor];
  } else {
    s_inv = scale_inv;
  }

  at::Tensor output = relu(input, s, a, s_inv, otype_arg);
  return output;
}

at::Tensor reglu_ts(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                    int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  at::Tensor s, a, s_inv;
  if (scale.numel()) {
    s = scale[fp8_tensor];
  } else {
    s = scale;
  }

  if (amax.numel()) {
    a = amax[0][fp8_tensor];
  } else {
    a = amax;
  }

  if (scale_inv.numel()) {
    s_inv = scale_inv[fp8_tensor];
  } else {
    s_inv = scale_inv;
  }

  at::Tensor output = reglu(input, s, a, s_inv, otype_arg);
  return output;
}

at::Tensor geglu_ts(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                    int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  at::Tensor s, a, s_inv;
  if (scale.numel()) {
    s = scale[fp8_tensor];
  } else {
    s = scale;
  }

  if (amax.numel()) {
    a = amax[0][fp8_tensor];
  } else {
    a = amax;
  }

  if (scale_inv.numel()) {
    s_inv = scale_inv[fp8_tensor];
  } else {
    s_inv = scale_inv;
  }

  at::Tensor output = geglu(input, s, a, s_inv, otype_arg);
  return output;
}

at::Tensor swiglu_ts(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                     int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  at::Tensor s, a, s_inv;
  if (scale.numel()) {
    s = scale[fp8_tensor];
  } else {
    s = scale;
  }

  if (amax.numel()) {
    a = amax[0][fp8_tensor];
  } else {
    a = amax;
  }

  if (scale_inv.numel()) {
    s_inv = scale_inv[fp8_tensor];
  } else {
    s_inv = scale_inv;
  }

  at::Tensor output = swiglu(input, s, a, s_inv, otype_arg);
  return output;
}

at::Tensor qgelu_ts(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                    int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  at::Tensor s, a, s_inv;
  if (scale.numel()) {
    s = scale[fp8_tensor];
  } else {
    s = scale;
  }

  if (amax.numel()) {
    a = amax[0][fp8_tensor];
  } else {
    a = amax;
  }

  if (scale_inv.numel()) {
    s_inv = scale_inv[fp8_tensor];
  } else {
    s_inv = scale_inv;
  }

  at::Tensor output = qgelu(input, s, a, s_inv, otype_arg);
  return output;
}

at::Tensor srelu_ts(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                    int64_t fp8_tensor, int64_t otype) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);

  at::Tensor s, a, s_inv;
  if (scale.numel()) {
    s = scale[fp8_tensor];
  } else {
    s = scale;
  }

  if (amax.numel()) {
    a = amax[0][fp8_tensor];
  } else {
    a = amax;
  }

  if (scale_inv.numel()) {
    s_inv = scale_inv[fp8_tensor];
  } else {
    s_inv = scale_inv;
  }

  at::Tensor output = srelu(input, s, a, s_inv, otype_arg);
  return output;
}

at::Tensor te_gemm_ts(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                      int64_t A_type, int64_t transa, at::Tensor B, at::Tensor B_scale_inverse,
                      int64_t B_fp8_tensor, int64_t B_type, int64_t transb, at::Tensor D,
                      at::Tensor D_scale, int64_t D_type, at::Tensor D_amax, at::Tensor bias,
                      int64_t bias_type, at::Tensor pre_gelu_out, int64_t grad,
                      at::Tensor workspace, int64_t workspaceSize, int64_t accumulate,
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

  // Set an external SM Margin to all the GEMMs.
  // This comes in handy when DP is overlapped with GEMMs

  const int sm_count = transformer_engine::cuda::sm_count();
  int num_math_sms = sm_count - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);

  if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

  if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

  te_gemm(A, A_scale_inverse, A_type_arg, transa_arg, B, B_scale_inverse, B_type_arg, transb_arg, D,
          D_scale, D_type_arg, D_amax, bias, bias_type_arg, pre_gelu_out, grad_arg, workspace,
          workspaceSize_arg, accumulate_arg, use_split_accumulator_arg, num_math_sms);
  return D;
}

at::Tensor layernorm_fwd_fp8_inf_ts(const at::Tensor &input, const at::Tensor &weight,
                                    const at::Tensor &bias, double eps, at::Tensor scale,
                                    at::Tensor amax, at::Tensor scale_inv, int64_t fp8_tensor,
                                    int64_t otype, const int64_t sm_margin,
                                    const bool zero_centered_gamma) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  float eps_float = static_cast<float>(eps);

  at::Tensor output = layernorm_fwd_fp8_inf(input, weight, bias, eps_float, scale, amax, scale_inv,
                                            otype_arg, sm_margin, zero_centered_gamma,
                                            fp8_tensor,   // scale_offset
                                            fp8_tensor,   // amax_offset
                                            fp8_tensor);  // scale_inv_offset

  return output;
}

at::Tensor layernorm_fwd_inf_ts(const at::Tensor &input, const at::Tensor &weight,
                                const at::Tensor &bias, double eps, const int64_t sm_margin,
                                const bool zero_centered_gamma) {
  float eps_float = static_cast<float>(eps);

  at::Tensor output =
      layernorm_fwd_inf(input, weight, bias, eps_float, sm_margin, zero_centered_gamma);

  return output;
}

at::Tensor rmsnorm_fwd_fp8_inf_ts(const at::Tensor &input, const at::Tensor &weight, double eps,
                                  at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                                  int64_t fp8_tensor, int64_t otype, const int64_t sm_margin,
                                  const bool zero_centered_gamma) {
  transformer_engine::DType otype_arg = reverse_map_dtype(otype);
  float eps_float = static_cast<float>(eps);

  at::Tensor output = rmsnorm_fwd_fp8_inf(input, weight, eps_float, scale, amax, scale_inv,
                                          otype_arg, sm_margin, zero_centered_gamma,
                                          fp8_tensor,   // scale_offset
                                          fp8_tensor,   // amax_offset
                                          fp8_tensor);  // scale_inv_offset

  return output;
}

at::Tensor rmsnorm_fwd_inf_ts(const at::Tensor &input, const at::Tensor &weight, double eps,
                              const int64_t sm_margin, const bool zero_centered_gamma) {
  float eps_float = static_cast<float>(eps);

  at::Tensor output = rmsnorm_fwd_inf(input, weight, eps_float, sm_margin, zero_centered_gamma);

  return output;
}

TORCH_LIBRARY(tex_ts, m) {
  m.def("cast_to_fp8_ts", &cast_to_fp8_ts);
  m.def("cast_to_fp8_noalloc_ts", &cast_to_fp8_noalloc_ts);
  m.def("cast_from_fp8_ts", &cast_from_fp8_ts);
  m.def("gelu_ts", &gelu_ts);
  m.def("relu_ts", &relu_ts);
  m.def("geglu_ts", &geglu_ts);
  m.def("reglu_ts", &reglu_ts);
  m.def("swiglu_ts", &swiglu_ts);
  m.def("qgelu_ts", &qgelu_ts);
  m.def("srelu_ts", &srelu_ts);
  m.def("te_gemm_ts", &te_gemm_ts);
  m.def("layernorm_fwd_fp8_inf_ts", &layernorm_fwd_fp8_inf_ts);
  m.def("layernorm_fwd_inf_ts", &layernorm_fwd_inf_ts);
  m.def("rmsnorm_fwd_fp8_inf_ts", &rmsnorm_fwd_fp8_inf_ts);
  m.def("rmsnorm_fwd_inf_ts", &rmsnorm_fwd_inf_ts);
}
