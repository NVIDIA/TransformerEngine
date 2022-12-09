/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"

void te_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
             bool transa, at::Tensor B, at::Tensor B_scale_inverse,
             transformer_engine::DType B_type, bool transb, at::Tensor D,
             transformer_engine::DType D_type, at::Tensor bias, at::Tensor pre_gelu_out, bool grad,
             at::Tensor workspace, size_t workspaceSize, bool accumulate,
             bool use_split_accumulator);

void fused_cast_transpose(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                          at::Tensor input_cast, at::Tensor input_transpose,
                          transformer_engine::DType otype);

std::vector<at::Tensor> fused_cast_transpose_bgrad(at::Tensor grad_output, at::Tensor scale,
                                                   at::Tensor amax, at::Tensor scale_inv,
                                                   transformer_engine::DType otype);

std::vector<at::Tensor> fused_cast_transpose_bgrad_dgelu(at::Tensor grad_output,
                                                         at::Tensor gelu_input, at::Tensor scale,
                                                         at::Tensor amax, at::Tensor scale_inv,
                                                         transformer_engine::DType otype);

void fused_multi_cast_transpose(std::vector<at::Tensor> input_list,
                                std::vector<at::Tensor> scale_list,
                                std::vector<at::Tensor> cast_output_list,
                                std::vector<at::Tensor> transposed_output_list,
                                std::vector<at::Tensor> amax_output_list,
                                std::vector<at::Tensor> scale_inv_output_list,
                                transformer_engine::DType otype);

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype);

at::Tensor fp8_gelu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                    transformer_engine::DType otype);

std::vector<at::Tensor> layernorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                      const at::Tensor &mu, const at::Tensor &rsigma,
                                      const at::Tensor &gamma);

std::vector<at::Tensor> layernorm_fwd_fp8(const at::Tensor &input, const at::Tensor &weight,
                                          const at::Tensor &bias, float eps, at::Tensor scale,
                                          at::Tensor amax, at::Tensor scale_inv,
                                          transformer_engine::DType otype);

std::vector<at::Tensor> layernorm_fwd(const at::Tensor &input, const at::Tensor &weight,
                                      const at::Tensor &bias, float eps);

at::Tensor cast_to_fp8(const at::Tensor &input, const at::Tensor &scale, at::Tensor amax,
                       at::Tensor scale_inv, transformer_engine::DType otype);

at::Tensor cast_from_fp8(const at::Tensor &input, const at::Tensor &scale_inv,
                         transformer_engine::DType itype, transformer_engine::DType otype);

at::Tensor scaled_softmax_forward(at::Tensor input, float scale_factor);

at::Tensor scaled_softmax_backward(at::Tensor output_grad_, at::Tensor softmax_results_,
                                   float scale_factor);

at::Tensor scaled_masked_softmax_forward(at::Tensor input, at::Tensor mask, float scale_factor);

at::Tensor scaled_masked_softmax_backward(at::Tensor output_grad_, at::Tensor softmax_results_,
                                          float scale_factor);

at::Tensor scaled_upper_triang_masked_softmax_forward(at::Tensor input, float scale_factor);

at::Tensor scaled_upper_triang_masked_softmax_backward(at::Tensor output_grads_,
                                                       at::Tensor softmax_results_,
                                                       float scale_factor);
