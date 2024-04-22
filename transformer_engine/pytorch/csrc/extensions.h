/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "common/common.h"

/***************************************************************************************************
 * Attention
 **************************************************************************************************/

NVTE_Fused_Attn_Backend get_fused_attn_backend(
                const transformer_engine::DType q_dtype,
                const transformer_engine::DType kv_dtype,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type,
                NVTE_Mask_Type attn_mask_type,
                float p_dropout,
                size_t num_attn_heads, size_t num_gqa_groups,
                size_t max_seqlen_q, size_t max_seqlen_kv,
                size_t head_dim);

std::vector<at::Tensor> fused_attn_fwd_qkvpacked(
                size_t max_seqlen, bool is_training,
                float attn_scale, float p_dropout, bool set_zero,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type,
                NVTE_Mask_Type attn_mask_type,
                const at::Tensor cu_seqlens,
                const at::Tensor QKV,
                const transformer_engine::DType qkv_type,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_O,
                c10::optional<at::Tensor> amax_S,
                c10::optional<at::Tensor> amax_O,
                const c10::optional<at::Tensor> Bias,
                const c10::optional<at::Generator> rng_gen,
                size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd_qkvpacked(
                size_t max_seqlen, float attn_scale,
                float p_dropout, bool set_zero,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type,
                NVTE_Mask_Type attn_mask_type,
                const at::Tensor cu_seqlens,
                const at::Tensor QKV,
                const at::Tensor O,
                const at::Tensor dO,
                const transformer_engine::DType qkv_type,
                const transformer_engine::DType dqkv_type,
                const std::vector<at::Tensor> Aux_CTX_Tensors,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> descale_O,
                const c10::optional<at::Tensor> descale_dO,
                const c10::optional<at::Tensor> descale_dP,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_dP,
                const c10::optional<at::Tensor> scale_dQKV,
                c10::optional<at::Tensor> amax_dP,
                c10::optional<at::Tensor> amax_dQKV);

std::vector<at::Tensor> fused_attn_fwd_kvpacked(
                size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                float attn_scale, float p_dropout, bool set_zero,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type,
                NVTE_Mask_Type attn_mask_type,
                const at::Tensor cu_seqlens_q,
                const at::Tensor cu_seqlens_kv,
                const at::Tensor Q,
                const at::Tensor KV,
                const transformer_engine::DType qkv_type,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_O,
                c10::optional<at::Tensor> amax_S,
                c10::optional<at::Tensor> amax_O,
                const c10::optional<at::Tensor> Bias,
                const c10::optional<at::Generator> rng_gen,
                size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd_kvpacked(
                size_t max_seqlen_q, size_t max_seqlen_kv,
                float attn_scale, float p_dropout, bool set_zero,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type,
                NVTE_Mask_Type attn_mask_type,
                const at::Tensor cu_seqlens_q,
                const at::Tensor cu_seqlens_kv,
                const at::Tensor Q,
                const at::Tensor KV,
                const at::Tensor O,
                const at::Tensor dO,
                const transformer_engine::DType qkv_type,
                const transformer_engine::DType dqkv_type,
                const std::vector<at::Tensor> Aux_CTX_Tensors,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> descale_O,
                const c10::optional<at::Tensor> descale_dO,
                const c10::optional<at::Tensor> descale_dP,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_dP,
                const c10::optional<at::Tensor> scale_dQKV,
                c10::optional<at::Tensor> amax_dP,
                c10::optional<at::Tensor> amax_dQKV);

std::vector<at::Tensor> fused_attn_fwd(
                size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                float attn_scale, float p_dropout, bool set_zero,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type,
                NVTE_Mask_Type attn_mask_type,
                const at::Tensor cu_seqlens_q,
                const at::Tensor cu_seqlens_kv,
                const at::Tensor Q,
                const at::Tensor K,
                const at::Tensor V,
                const transformer_engine::DType qkv_type,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_O,
                c10::optional<at::Tensor> amax_S,
                c10::optional<at::Tensor> amax_O,
                const c10::optional<at::Tensor> Bias,
                const c10::optional<at::Generator> rng_gen,
                size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd(
                size_t max_seqlen_q, size_t max_seqlen_kv,
                float attn_scale, float p_dropout, bool set_zero,
                NVTE_QKV_Layout qkv_layout,
                NVTE_Bias_Type bias_type,
                NVTE_Mask_Type attn_mask_type,
                const at::Tensor cu_seqlens_q,
                const at::Tensor cu_seqlens_kv,
                const at::Tensor Q,
                const at::Tensor K,
                const at::Tensor V,
                const at::Tensor O,
                const at::Tensor dO,
                const transformer_engine::DType qkv_type,
                const transformer_engine::DType dqkv_type,
                const std::vector<at::Tensor> Aux_CTX_Tensors,
                const c10::optional<at::Tensor> descale_QKV,
                const c10::optional<at::Tensor> descale_S,
                const c10::optional<at::Tensor> descale_O,
                const c10::optional<at::Tensor> descale_dO,
                const c10::optional<at::Tensor> descale_dP,
                const c10::optional<at::Tensor> scale_S,
                const c10::optional<at::Tensor> scale_dP,
                const c10::optional<at::Tensor> scale_dQKV,
                c10::optional<at::Tensor> amax_dP,
                c10::optional<at::Tensor> amax_dQKV);

at::Tensor fa_prepare_fwd(at::Tensor qkvi);
at::Tensor fa_prepare_bwd(at::Tensor q, at::Tensor k, at::Tensor v);

/***************************************************************************************************
 * GEMM
 **************************************************************************************************/

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

void te_atomic_gemm(at::Tensor A,
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
                    int math_sm_count,
                    int m_split,
                    int n_split,
                    bool gemm_producer,
                    at::Tensor counter
);

/***************************************************************************************************
 * Transpose
 **************************************************************************************************/

void fused_cast_transpose(at::Tensor input,
                          at::Tensor scale,
                          at::Tensor amax,
                          at::Tensor scale_inv,
                          at::Tensor input_cast,
                          at::Tensor input_transpose,
                          transformer_engine::DType otype
);


void fused_cast_transpose_noop(at::Tensor input,
                               at::Tensor noop,
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

void fp8_transpose_noalloc(at::Tensor input,
                           at::Tensor output,
                           transformer_engine::DType otype
);

void fp8_transpose_noalloc_noop(at::Tensor input,
                                at::Tensor output,
                                at::Tensor noop,
                                transformer_engine::DType otype
);

/***************************************************************************************************
 * Activations
 **************************************************************************************************/

at::Tensor gelu(at::Tensor input,
                at::Tensor scale,
                at::Tensor amax,
                at::Tensor scale_inv,
                transformer_engine::DType otype
);

at::Tensor relu(at::Tensor input,
                at::Tensor scale,
                at::Tensor amax,
                at::Tensor scale_inv,
                transformer_engine::DType otype
);

at::Tensor geglu(at::Tensor input,
                 at::Tensor scale,
                 at::Tensor amax,
                 at::Tensor scale_inv,
                 transformer_engine::DType otype
);

at::Tensor reglu(at::Tensor input,
                 at::Tensor scale,
                 at::Tensor amax,
                 at::Tensor scale_inv,
                 transformer_engine::DType otype
);

at::Tensor swiglu(at::Tensor input,
                  at::Tensor scale,
                  at::Tensor amax,
                  at::Tensor scale_inv,
                  transformer_engine::DType otype
);

at::Tensor qgelu(at::Tensor input,
                  at::Tensor scale,
                  at::Tensor amax,
                  at::Tensor scale_inv,
                  transformer_engine::DType otype
);

at::Tensor dgelu(at::Tensor grad,
                 at::Tensor input,
                 transformer_engine::DType otype
);

at::Tensor drelu(at::Tensor grad,
                 at::Tensor input,
                 transformer_engine::DType otype
);

at::Tensor dgeglu(at::Tensor grad,
                  at::Tensor input,
                  transformer_engine::DType otype
);

at::Tensor dreglu(at::Tensor grad,
                  at::Tensor input,
                  transformer_engine::DType otype
);

at::Tensor dswiglu(at::Tensor grad,
                   at::Tensor input,
                   transformer_engine::DType otype
);

at::Tensor dqgelu(at::Tensor grad,
                   at::Tensor input,
                   transformer_engine::DType otype
);

/***************************************************************************************************
 * LayerNorm
 **************************************************************************************************/

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
                                 const int sm_margin,
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
                             const int sm_margin,
                             const bool zero_centered_gamma
);

/***************************************************************************************************
 * RMSNorm
 **************************************************************************************************/

std::vector<at::Tensor> rmsnorm_bwd(const at::Tensor &dz,
                                    const at::Tensor &x,
                                    const at::Tensor &rsigma,
                                    const at::Tensor &gamma,
                                    const int sm_margin,
                                    const bool zero_centered_gamma
);


std::vector<at::Tensor> rmsnorm_fwd_fp8(const at::Tensor &input,
                                        const at::Tensor &weight,
                                        float eps,
                                        at::Tensor scale,
                                        at::Tensor amax,
                                        at::Tensor scale_inv,
                                        transformer_engine::DType otype,
                                        const int sm_margin,
                                        const bool zero_centered_gamma
);

std::vector<at::Tensor> rmsnorm_fwd_fp8_noalloc(const at::Tensor &input,
                                                const at::Tensor &weight,
                                                float eps,
                                                at::Tensor scale,
                                                at::Tensor ln_out,
                                                at::Tensor amax,
                                                at::Tensor scale_inv,
                                                transformer_engine::DType otype,
                                                const int sm_margin,
                                                const bool zero_centered_gamma
);

at::Tensor rmsnorm_fwd_fp8_inf(const at::Tensor &input,
                               const at::Tensor &weight,
                               float eps,
                               at::Tensor scale,
                               at::Tensor amax,
                               at::Tensor scale_inv,
                               transformer_engine::DType otype,
                               const int sm_margin,
                               const bool zero_centered_gamma
);

std::vector<at::Tensor> rmsnorm_fwd(const at::Tensor &input,
                                    const at::Tensor &weight,
                                    float eps,
                                    const int sm_margin,
                                    const bool zero_centered_gamma
);

std::vector<at::Tensor> rmsnorm_fwd_noalloc(const at::Tensor &input,
                                    const at::Tensor &weight,
                                    at::Tensor ln_out,
                                    float eps,
                                    const int sm_margin,
                                    const bool zero_centered_gamma
);

at::Tensor rmsnorm_fwd_inf(const at::Tensor &input,
                           const at::Tensor &weight,
                           float eps,
                           const int sm_margin,
                           const bool zero_centered_gamma
);

/***************************************************************************************************
 * Cast
 **************************************************************************************************/

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

/***************************************************************************************************
 * Softmax
 **************************************************************************************************/

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


at::Tensor scaled_aligned_causal_masked_softmax_forward(at::Tensor input,
                                                        float scale_factor
);


at::Tensor scaled_aligned_causal_masked_softmax_backward(at::Tensor output_grads_,
                                                         at::Tensor softmax_results_,
                                                         float scale_factor
);

/***************************************************************************************************
 * FP8 recipe
 **************************************************************************************************/

void fused_amax_and_scale_update_after_reduction(const at::Tensor &amax_reduction_buffer,
                                                 std::vector<at::Tensor> amax_histories,
                                                 std::vector<at::Tensor> scales,
                                                 std::vector<at::Tensor> scale_invs,
                                                 const std::string &amax_compute_algo,
                                                 transformer_engine::DType fp8_dtype,
                                                 float margin);

/***************************************************************************************************
 * Rotary positional embedding
 **************************************************************************************************/

at::Tensor fused_rope_forward(const at::Tensor &input,
                              const at::Tensor &freqs,
                              const bool transpose_output_memory
);

at::Tensor fused_rope_backward(const at::Tensor &output_grads,
                               const at::Tensor &freqs,
                               const bool transpose_output_memory
);

at::Tensor fused_rope_thd_forward(const at::Tensor &input,
                                  const at::Tensor &cu_seqlens,
                                  const at::Tensor &freqs
);

at::Tensor fused_rope_thd_backward(const at::Tensor &output_grads,
                                   const at::Tensor &cu_seqlens,
                                   const at::Tensor &freqs
);

/***************************************************************************************************
 * Miscellaneous
 **************************************************************************************************/

size_t get_cublasLt_version();

size_t get_cudnn_version();

bool userbuf_comm_available();

void placeholder();
