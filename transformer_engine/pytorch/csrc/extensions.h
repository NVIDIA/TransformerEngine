/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "../common.h"

NVTE_QKV_Layout get_nvte_qkv_layout(const std::string qkv_layout);

NVTE_Bias_Type get_nvte_bias_type(const std::string bias_type);

NVTE_Mask_Type get_nvte_mask_type(const std::string mask_type);

std::vector<at::Tensor> fused_attn_fwd_qkvpacked(
                size_t b, size_t max_seqlen, size_t total_seqs,
                size_t h, size_t d,
                bool is_training, float attn_scale, float p_dropout, bool set_zero,
                std::string qkv_layout, std::string bias_type, std::string attn_mask_type,
                const at::Tensor cu_seqlens,
                const at::Tensor QKV,
                const transformer_engine::DType qkv_type,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_O,
                c10::optional<at::Tensor> amax_S,
                c10::optional<at::Tensor> amax_O,
                const c10::optional<at::Tensor> Bias,
                const c10::optional<at::Generator> rng_gen,
                bool return_softmax,
                int num_split,
                int fused_attention_backend);

std::vector<at::Tensor> fused_attn_bwd_qkvpacked(
                size_t b, size_t max_seqlen, size_t total_seqs,
                size_t h, size_t d,
                float attn_scale, float p_dropout, bool set_zero,
                std::string qkv_layout, std::string bias_type, std::string attn_mask_type,
                const at::Tensor cu_seqlens,
                const at::Tensor QKV,
                const at::Tensor O,
                const at::Tensor dO,
                const transformer_engine::DType qkv_type,
                const std::vector<at::Tensor> Aux_CTX_Tensors,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> descale_O,
                const c10::optional<at::Tensor> descale_dO,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_dP,
                const c10::optional<at::Tensor> scale_dQKV,
                c10::optional<at::Tensor> amax_dP,
                c10::optional<at::Tensor> amax_dQKV,
                int num_split,
                int fused_attention_backend); 

std::vector<at::Tensor> fused_attn_fwd_kvpacked(
                size_t b, size_t max_seqlen_q, size_t max_seqlen_kv,
                size_t total_seqs_q, size_t total_seqs_kv,
                size_t h, size_t d,
                bool is_training, float attn_scale, float p_dropout, bool set_zero,
                std::string qkv_layout, std::string bias_type, std::string attn_mask_type,
                const at::Tensor cu_seqlens_q,
                const at::Tensor cu_seqlens_kv,
                const at::Tensor Q,
                const at::Tensor KV,
                const transformer_engine::DType qkv_type,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_O,
                c10::optional<at::Tensor> amax_S,
                c10::optional<at::Tensor> amax_O,
                const c10::optional<at::Tensor> Bias,
                const c10::optional<at::Generator> rng_gen,
                bool return_softmax,
                int num_split,
                int fused_attention_backend);

std::vector<at::Tensor> fused_attn_bwd_kvpacked(
                size_t b, size_t max_seqlen_q, size_t max_seqlen_kv,
                size_t total_seqs_q, size_t total_seqs_kv,
                size_t h, size_t d,
                float attn_scale, float p_dropout, bool set_zero,
                std::string qkv_layout, std::string bias_type, std::string attn_mask_type,
                const at::Tensor cu_seqlens_q,
                const at::Tensor cu_seqlens_kv,
                const at::Tensor Q,
                const at::Tensor KV,
                const at::Tensor O,
                const at::Tensor dO,
                const transformer_engine::DType qkv_type,
                const std::vector<at::Tensor> Aux_CTX_Tensors,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> descale_O,
                const c10::optional<at::Tensor> descale_dO,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_dP,
                const c10::optional<at::Tensor> scale_dQKV,
                c10::optional<at::Tensor> amax_dP,
                c10::optional<at::Tensor> amax_dQKV,
                int num_split,
                int fused_attention_backend); 

void te_gemm(at::Tensor A,
             at::Tensor A_scale_inverse,
             transformer_engine::DType A_type,
             bool transa,
             at::Tensor B,
             at::Tensor B_scale_inverse,
             transformer_engine::DType B_type,
             bool transb,
             at::Tensor D,
             at::Tensor D_scale,
             transformer_engine::DType D_type,
             at::Tensor D_amax,
             at::Tensor bias,
             transformer_engine::DType bias_type,
             at::Tensor pre_gelu_out,
             bool grad,
             at::Tensor workspace,
             size_t workspaceSize,
             bool accumulate,
             bool use_split_accumulator,
             int math_sm_count
);


void fused_cast_transpose(at::Tensor input,
                          at::Tensor scale,
                          at::Tensor amax,
                          at::Tensor scale_inv,
                          at::Tensor input_cast,
                          at::Tensor input_transpose,
                          transformer_engine::DType otype
);


std::vector<at::Tensor> fused_cast_transpose_bgrad(at::Tensor grad_output,
                                                   at::Tensor scale,
                                                   at::Tensor amax,
                                                   at::Tensor scale_inv,
                                                   transformer_engine::DType otype
);


std::vector<at::Tensor> fused_fp8_transpose_bgrad(at::Tensor grad_output,
                                              at::Tensor scale,
                                              at::Tensor amax,
                                              at::Tensor scale_inv,
                                              transformer_engine::DType otype,
                                              transformer_engine::DType grad_bias_type
);


std::vector<at::Tensor> fused_cast_transpose_bgrad_dgelu(at::Tensor grad_output,
                                                         at::Tensor gelu_input,
                                                         at::Tensor scale,
                                                         at::Tensor amax,
                                                         at::Tensor scale_inv,
                                                         transformer_engine::DType otype
);


void fused_multi_cast_transpose(std::vector<at::Tensor> input_list,
                                std::vector<at::Tensor> scale_list,
                                std::vector<at::Tensor> cast_output_list,
                                std::vector<at::Tensor> transposed_output_list,
                                std::vector<at::Tensor> amax_output_list,
                                std::vector<at::Tensor> scale_inv_output_list,
                                transformer_engine::DType otype
);


at::Tensor fp8_transpose(at::Tensor input,
                         transformer_engine::DType otype
);


at::Tensor fp8_gelu(at::Tensor input,
                    at::Tensor scale,
                    at::Tensor amax,
                    at::Tensor scale_inv,
                    transformer_engine::DType otype
);


std::vector<at::Tensor> layernorm_bwd(const at::Tensor &dz,
                                      const at::Tensor &x,
                                      const at::Tensor &mu,
                                      const at::Tensor &rsigma,
                                      const at::Tensor &gamma,
                                      const int sm_margin,
                                      const bool zero_centered_gamma
);


std::vector<at::Tensor> layernorm_fwd_fp8(const at::Tensor &input,
                                          const at::Tensor &weight,
                                          const at::Tensor &bias,
                                          float eps,
                                          at::Tensor scale,
                                          at::Tensor amax,
                                          at::Tensor scale_inv,
                                          transformer_engine::DType otype,
                                          const int sm_margin,
                                          const bool zero_centered_gamma
);

std::vector<at::Tensor> layernorm_fwd_fp8_noalloc(const at::Tensor &input,
                                                  const at::Tensor &weight,
                                                  const at::Tensor &bias,
                                                  float eps,
                                                  at::Tensor scale,
                                                  at::Tensor ln_out,
                                                  at::Tensor amax,
                                                  at::Tensor scale_inv,
                                                  transformer_engine::DType otype,
                                                  const int sm_margin,
                                                  const bool zero_centered_gamma
);

at::Tensor layernorm_fwd_fp8_inf(const at::Tensor &input,
                                 const at::Tensor &weight,
                                 const at::Tensor &bias,
                                 float eps,
                                 at::Tensor scale,
                                 at::Tensor amax,
                                 at::Tensor scale_inv,
                                 transformer_engine::DType otype,
                                 const bool zero_centered_gamma
);

std::vector<at::Tensor> layernorm_fwd(const at::Tensor &input,
                                      const at::Tensor &weight,
                                      const at::Tensor &bias,
                                      float eps,
                                      const int sm_margin,
                                      const bool zero_centered_gamma
);

std::vector<at::Tensor> layernorm_fwd_noalloc(const at::Tensor &input,
                                      const at::Tensor &weight,
                                      const at::Tensor &bias,
                                      at::Tensor ln_out,
                                      float eps,
                                      const int sm_margin,
                                      const bool zero_centered_gamma
);

at::Tensor layernorm_fwd_inf(const at::Tensor &input,
                             const at::Tensor &weight,
                             const at::Tensor &bias,
                             float eps,
                             const bool zero_centered_gamma
);

at::Tensor cast_to_fp8(const at::Tensor &input,
                       const at::Tensor &scale,
                       at::Tensor amax,
                       at::Tensor scale_inv,
                       transformer_engine::DType otype
);


void cast_to_fp8_noalloc(const at::Tensor &input,
                         const at::Tensor &scale,
                         at::Tensor output,
                         at::Tensor amax,
                         at::Tensor scale_inv,
                         transformer_engine::DType otype
);


at::Tensor cast_from_fp8(const at::Tensor &input,
                         const at::Tensor &scale_inv,
                         transformer_engine::DType itype,
                         transformer_engine::DType otype
);


at::Tensor scaled_softmax_forward(at::Tensor input,
                                  float scale_factor
);


at::Tensor scaled_softmax_backward(at::Tensor output_grad_,
                                   at::Tensor softmax_results_,
                                   float scale_factor
);


at::Tensor scaled_masked_softmax_forward(at::Tensor input,
                                         at::Tensor mask,
                                         float scale_factor
);


at::Tensor scaled_masked_softmax_backward(at::Tensor output_grad_,
                                          at::Tensor softmax_results_,
                                          float scale_factor
);


at::Tensor scaled_upper_triang_masked_softmax_forward(at::Tensor input,
                                                      float scale_factor
);


at::Tensor scaled_upper_triang_masked_softmax_backward(at::Tensor output_grads_,
                                                       at::Tensor softmax_results_,
                                                       float scale_factor
);
