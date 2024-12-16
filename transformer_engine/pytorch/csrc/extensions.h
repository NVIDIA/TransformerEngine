/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_

#include <optional>

#include "common.h"
#include "common/common.h"

/***************************************************************************************************
 * Permutation
 **************************************************************************************************/

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> moe_permute_fwd(
    at::Tensor input, const transformer_engine::DType dtype, at::Tensor indices,
    int64_t num_out_tokens, std::vector<at::Tensor> workspace, int64_t max_expanded_token_num);

at::Tensor moe_permute_bwd(at::Tensor input, const transformer_engine::DType dtype,
                           at::Tensor row_id_map, at::Tensor prob, int64_t num_tokens,
                           int64_t topK);

at::Tensor moe_unpermute_fwd(at::Tensor input, const transformer_engine::DType dtype,
                             at::Tensor row_id_map, at::Tensor prob, int64_t num_tokens,
                             int64_t topK);

std::tuple<at::Tensor, at::Tensor> moe_unpermute_bwd(at::Tensor input_bwd, at::Tensor input_fwd,
                                                     const transformer_engine::DType dtype,
                                                     at::Tensor row_id_map, at::Tensor prob);

/***************************************************************************************************
 * Attention
 **************************************************************************************************/

NVTE_Fused_Attn_Backend get_fused_attn_backend(const transformer_engine::DType q_dtype,
                                               const transformer_engine::DType kv_dtype,
                                               NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                               NVTE_Mask_Type attn_mask_type, float p_dropout,
                                               size_t num_attn_heads, size_t num_gqa_groups,
                                               size_t max_seqlen_q, size_t max_seqlen_kv,
                                               size_t head_dim_qk, size_t head_dim_v,
                                               int64_t window_size_left, int64_t window_size_right);

std::vector<at::Tensor> fused_attn_fwd_qkvpacked(
    size_t max_seqlen, bool is_training, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const std::vector<int64_t> window_size, bool bottom_right_diagonal, const at::Tensor cu_seqlens, const at::Tensor QKV,
    const transformer_engine::DType qkv_type, const c10::optional<at::Tensor> cu_seqlens_padded,
    const c10::optional<at::Tensor> descale_QKV, const int descale_QKV_offset,
    const c10::optional<at::Tensor> descale_S, const int descale_S_offset,
    const c10::optional<at::Tensor> scale_S, const int scale_S_offset,
    const c10::optional<at::Tensor> scale_O, const int scale_O_offset,
    c10::optional<at::Tensor> amax_S, const int amax_S_offset, c10::optional<at::Tensor> amax_O,
    const int amax_O_offset, const c10::optional<at::Tensor> Bias,
    const c10::optional<at::Generator> rng_gen, size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd_qkvpacked(
    size_t max_seqlen, float attn_scale, float p_dropout, bool set_zero, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, const std::vector<int64_t> window_size, bool bottom_right_diagonal,
    bool deterministic, const at::Tensor cu_seqlens, const at::Tensor QKV, const at::Tensor O,
    const at::Tensor dO, const transformer_engine::DType qkv_type,
    const transformer_engine::DType dqkv_type, const std::vector<at::Tensor> Aux_CTX_Tensors,
    const c10::optional<at::Tensor> cu_seqlens_padded, const c10::optional<at::Tensor> descale_QKV,
    const c10::optional<at::Tensor> descale_S, const c10::optional<at::Tensor> descale_O,
    const c10::optional<at::Tensor> descale_dO, const c10::optional<at::Tensor> descale_dP,
    const c10::optional<at::Tensor> scale_S, const c10::optional<at::Tensor> scale_dP,
    const c10::optional<at::Tensor> scale_dQKV, c10::optional<at::Tensor> amax_dP,
    c10::optional<at::Tensor> amax_dQKV);

std::vector<at::Tensor> fused_attn_fwd_kvpacked(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, const std::vector<int64_t> window_size, bool bottom_right_diagonal,
    const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv, const at::Tensor Q,
    const at::Tensor KV, const transformer_engine::DType qkv_type,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const int descale_QKV_offset,
    const c10::optional<at::Tensor> descale_S, const int descale_S_offset,
    const c10::optional<at::Tensor> scale_S, const int scale_S_offset,
    const c10::optional<at::Tensor> scale_O, const int scale_O_offset,
    c10::optional<at::Tensor> amax_S, const int amax_S_offset, c10::optional<at::Tensor> amax_O,
    const int amax_O_offset, const c10::optional<at::Tensor> Bias,
    const c10::optional<at::Generator> rng_gen, size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd_kvpacked(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const std::vector<int64_t> window_size, bool bottom_right_diagonal, bool deterministic, const at::Tensor cu_seqlens_q,
    const at::Tensor cu_seqlens_kv, const at::Tensor Q, const at::Tensor KV, const at::Tensor O,
    const at::Tensor dO, const transformer_engine::DType qkv_type,
    const transformer_engine::DType dqkv_type, const std::vector<at::Tensor> Aux_CTX_Tensors,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> descale_O, const c10::optional<at::Tensor> descale_dO,
    const c10::optional<at::Tensor> descale_dP, const c10::optional<at::Tensor> scale_S,
    const c10::optional<at::Tensor> scale_dP, const c10::optional<at::Tensor> scale_dQKV,
    c10::optional<at::Tensor> amax_dP, c10::optional<at::Tensor> amax_dQKV);

std::vector<at::Tensor> fused_attn_fwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, const std::vector<int64_t> window_size, bool bottom_right_diagonal,
    const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv, const at::Tensor Q,
    const at::Tensor K, const at::Tensor V, const transformer_engine::DType qkv_type,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const int descale_QKV_offset,
    const c10::optional<at::Tensor> descale_S, const int descale_S_offset,
    const c10::optional<at::Tensor> scale_S, const int scale_S_offset,
    const c10::optional<at::Tensor> scale_O, const int scale_O_offset,
    c10::optional<at::Tensor> amax_S, const int amax_S_offset, c10::optional<at::Tensor> amax_O,
    const int amax_O_offset, const c10::optional<at::Tensor> Bias,
    const c10::optional<at::Generator> rng_gen, size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const std::vector<int64_t> window_size, bool bottom_right_diagonal, bool deterministic, const at::Tensor cu_seqlens_q,
    const at::Tensor cu_seqlens_kv, const at::Tensor Q, const at::Tensor K, const at::Tensor V,
    const at::Tensor O, const at::Tensor dO, const transformer_engine::DType qkv_type,
    const transformer_engine::DType dqkv_type, const std::vector<at::Tensor> Aux_CTX_Tensors,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> descale_O, const c10::optional<at::Tensor> descale_dO,
    const c10::optional<at::Tensor> descale_dP, const c10::optional<at::Tensor> scale_S,
    const c10::optional<at::Tensor> scale_dP, const c10::optional<at::Tensor> scale_dQKV,
    c10::optional<at::Tensor> amax_dP, c10::optional<at::Tensor> amax_dQKV);

at::Tensor fa_prepare_fwd(at::Tensor qkvi);
at::Tensor fa_prepare_bwd(at::Tensor q, at::Tensor k, at::Tensor v);

/***************************************************************************************************
 * GEMM
 **************************************************************************************************/

void te_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
             bool transa, at::Tensor B, at::Tensor B_scale_inverse,
             transformer_engine::DType B_type, bool transb, at::Tensor D, at::Tensor D_scale,
             transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
             transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
             at::Tensor workspace, size_t workspaceSize, bool accumulate,
             bool use_split_accumulator, int math_sm_count);

void te_atomic_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
                    bool transa, at::Tensor B, at::Tensor B_scale_inverse,
                    transformer_engine::DType B_type, bool transb, at::Tensor D, at::Tensor D_scale,
                    transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
                    transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
                    at::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, at::Tensor counter);

void te_grouped_gemm(std::vector<at::Tensor> A, at::Tensor A_scale_inverse, int A_offset,
                     transformer_engine::DType A_type, bool transa, std::vector<at::Tensor> B,
                     at::Tensor B_scale_inverse, int B_offset, transformer_engine::DType B_type,
                     bool transb, std::vector<at::Tensor> D, int D_offset, at::Tensor D_scale,
                     transformer_engine::DType D_type, at::Tensor D_amax,
                     std::vector<at::Tensor> bias, transformer_engine::DType bias_type,
                     std::vector<at::Tensor> pre_gelu_out, bool grad,
                     std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
                     bool use_split_accumulator, int math_sm_count);

void te_grouped_gemm_single_output(
    std::vector<at::Tensor> A, std::vector<at::Tensor> A_scale_inverse, int A_offset,
    transformer_engine::DType A_type, bool transa, std::vector<at::Tensor> B,
    at::Tensor B_scale_inverse, int B_offset, transformer_engine::DType B_type, bool transb,
    std::vector<int64_t> m_splits, at::Tensor D, int D_offset, at::Tensor D_scale,
    transformer_engine::DType D_type, at::Tensor D_amax, std::vector<at::Tensor> bias,
    transformer_engine::DType bias_type, std::vector<at::Tensor> pre_gelu_out, bool grad,
    std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, int math_sm_count);

/***************************************************************************************************
 * Transpose
 **************************************************************************************************/

void fused_cast_transpose(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                          at::Tensor input_cast, at::Tensor input_transpose,
                          transformer_engine::DType otype);

void fused_cast_transpose_noop(at::Tensor input, at::Tensor noop, at::Tensor scale, at::Tensor amax,
                               at::Tensor scale_inv, at::Tensor input_cast,
                               at::Tensor input_transpose, transformer_engine::DType otype,
                               int scale_offset = 0, int amax_offset = 0, int scale_inv_offset = 0);

std::vector<at::Tensor> fused_cast_transpose_bgrad(at::Tensor grad_output, at::Tensor scale,
                                                   at::Tensor amax, at::Tensor scale_inv,
                                                   transformer_engine::DType otype,
                                                   int scale_offset = 0, int amax_offset = 0,
                                                   int scale_inv_offset = 0);

std::vector<at::Tensor> fused_fp8_transpose_bgrad(at::Tensor grad_output, at::Tensor scale,
                                                  at::Tensor amax, at::Tensor scale_inv,
                                                  transformer_engine::DType otype,
                                                  transformer_engine::DType grad_bias_type,
                                                  int scale_offset = 0, int amax_offset = 0,
                                                  int scale_inv_offset = 0);

std::vector<at::Tensor> fused_cast_transpose_bgrad_dgelu(at::Tensor grad_output,
                                                         at::Tensor gelu_input, at::Tensor scale,
                                                         at::Tensor amax, at::Tensor scale_inv,
                                                         transformer_engine::DType otype,
                                                         int scale_offset = 0, int amax_offset = 0,
                                                         int scale_inv_offset = 0);

void fused_dswiglu_cast_transpose(at::Tensor grad_output, at::Tensor input, at::Tensor grad_input,
                                  at::Tensor grad_input_transpose, at::Tensor scale,
                                  at::Tensor amax, at::Tensor scale_inv,
                                  transformer_engine::DType otype, int scale_offset = 0,
                                  int amax_offset = 0, int scale_inv_offset = 0);

void fused_multi_cast_transpose(std::vector<at::Tensor> input_list,
                                std::vector<at::Tensor> scale_list,
                                std::vector<at::Tensor> cast_output_list,
                                std::vector<at::Tensor> transposed_output_list,
                                std::vector<at::Tensor> amax_output_list,
                                std::vector<at::Tensor> scale_inv_output_list,
                                transformer_engine::DType otype);

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> fused_multi_cast_transpose_alloc(
    std::vector<at::Tensor> input_list, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
    std::vector<int> scale_indices, std::vector<int> amax_indices,
    std::vector<int> scale_inv_indices, transformer_engine::DType otype);

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype);

void fp8_transpose_noalloc(at::Tensor input, at::Tensor output, transformer_engine::DType otype);

void fp8_transpose_noalloc_noop(at::Tensor input, at::Tensor output, at::Tensor noop,
                                transformer_engine::DType otype);

/***************************************************************************************************
 * Activations
 **************************************************************************************************/

at::Tensor gelu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                transformer_engine::DType otype);

at::Tensor relu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                transformer_engine::DType otype);

at::Tensor geglu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                 transformer_engine::DType otype);

at::Tensor reglu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                 transformer_engine::DType otype);

at::Tensor swiglu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                  transformer_engine::DType otype);

at::Tensor qgelu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                 transformer_engine::DType otype);

at::Tensor srelu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                 transformer_engine::DType otype);

at::Tensor dgelu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype);

at::Tensor drelu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype);

at::Tensor dgeglu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype);

at::Tensor dreglu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype);

at::Tensor dswiglu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype);

at::Tensor dqgelu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype);

at::Tensor dsrelu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype);

/***************************************************************************************************
 * LayerNorm
 **************************************************************************************************/

std::vector<at::Tensor> layernorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                      const at::Tensor &mu, const at::Tensor &rsigma,
                                      const at::Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma);

std::vector<at::Tensor> layernorm_fwd_fp8(const at::Tensor &input, const at::Tensor &weight,
                                          const at::Tensor &bias, float eps, at::Tensor scale,
                                          at::Tensor amax, at::Tensor scale_inv,
                                          transformer_engine::DType otype, const int sm_margin,
                                          const bool zero_centered_gamma,
                                          const int scale_offset = 0, const int amax_offset = 0,
                                          const int scale_inv_offset = 0);

std::vector<at::Tensor> layernorm_fwd_fp8_noalloc(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, float eps,
    at::Tensor scale, at::Tensor ln_out, at::Tensor amax, at::Tensor scale_inv,
    transformer_engine::DType otype, const int sm_margin, const bool zero_centered_gamma,
    const int scale_offset = 0, const int amax_offset = 0, const int scale_inv_offset = 0);

at::Tensor layernorm_fwd_fp8_inf(const at::Tensor &input, const at::Tensor &weight,
                                 const at::Tensor &bias, float eps, at::Tensor scale,
                                 at::Tensor amax, at::Tensor scale_inv,
                                 transformer_engine::DType otype, const int sm_margin,
                                 const bool zero_centered_gamma, const int scale_offset = 0,
                                 const int amax_offset = 0, const int scale_inv_offset = 0);

std::vector<at::Tensor> layernorm_fwd(const at::Tensor &input, const at::Tensor &weight,
                                      const at::Tensor &bias, float eps, const int sm_margin,
                                      const bool zero_centered_gamma);

std::vector<at::Tensor> layernorm_fwd_noalloc(const at::Tensor &input, const at::Tensor &weight,
                                              const at::Tensor &bias, at::Tensor ln_out, float eps,
                                              const int sm_margin, const bool zero_centered_gamma);

at::Tensor layernorm_fwd_inf(const at::Tensor &input, const at::Tensor &weight,
                             const at::Tensor &bias, float eps, const int sm_margin,
                             const bool zero_centered_gamma);

/***************************************************************************************************
 * RMSNorm
 **************************************************************************************************/

std::vector<at::Tensor> rmsnorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                    const at::Tensor &rsigma, const at::Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma);

std::vector<at::Tensor> rmsnorm_fwd_fp8(const at::Tensor &input, const at::Tensor &weight,
                                        float eps, at::Tensor scale, at::Tensor amax,
                                        at::Tensor scale_inv, transformer_engine::DType otype,
                                        const int sm_margin, const bool zero_centered_gamma,
                                        const int scale_offset = 0, const int amax_offset = 0,
                                        const int scale_inv_offset = 0);

std::vector<at::Tensor> rmsnorm_fwd_fp8_noalloc(
    const at::Tensor &input, const at::Tensor &weight, float eps, at::Tensor scale,
    at::Tensor ln_out, at::Tensor amax, at::Tensor scale_inv, transformer_engine::DType otype,
    const int sm_margin, const bool zero_centered_gamma, const int scale_offset = 0,
    const int amax_offset = 0, const int scale_inv_offset = 0);

at::Tensor rmsnorm_fwd_fp8_inf(const at::Tensor &input, const at::Tensor &weight, float eps,
                               at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                               transformer_engine::DType otype, const int sm_margin,
                               const bool zero_centered_gamma, const int scale_offset = 0,
                               const int amax_offset = 0, const int scale_inv_offset = 0);

std::vector<at::Tensor> rmsnorm_fwd(const at::Tensor &input, const at::Tensor &weight, float eps,
                                    const int sm_margin, const bool zero_centered_gamma);

std::vector<at::Tensor> rmsnorm_fwd_noalloc(const at::Tensor &input, const at::Tensor &weight,
                                            at::Tensor ln_out, float eps, const int sm_margin,
                                            const bool zero_centered_gamma);

at::Tensor rmsnorm_fwd_inf(const at::Tensor &input, const at::Tensor &weight, float eps,
                           const int sm_margin, const bool zero_centered_gamma);

/***************************************************************************************************
 * Cast
 **************************************************************************************************/

at::Tensor cast_to_fp8(const at::Tensor &input, const at::Tensor &scale, at::Tensor amax,
                       at::Tensor scale_inv, transformer_engine::DType otype,
                       const int scale_offset = 0, const int amax_offset = 0,
                       const int scale_inv_offset = 0);

void cast_to_fp8_noalloc(const at::Tensor &input, const at::Tensor &scale, at::Tensor output,
                         at::Tensor amax, at::Tensor scale_inv, transformer_engine::DType otype,
                         const int scale_offset = 0, const int amax_offset = 0,
                         const int scale_inv_offset = 0);

at::Tensor cast_from_fp8(const at::Tensor &input, const at::Tensor &scale_inv,
                         transformer_engine::DType itype, transformer_engine::DType otype,
                         const int scale_inv_offset = 0);

/***************************************************************************************************
 * Softmax
 **************************************************************************************************/

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

at::Tensor scaled_aligned_causal_masked_softmax_forward(at::Tensor input, float scale_factor);

at::Tensor scaled_aligned_causal_masked_softmax_backward(at::Tensor output_grads_,
                                                         at::Tensor softmax_results_,
                                                         float scale_factor);

/***************************************************************************************************
 * FP8 recipe
 **************************************************************************************************/

void fused_amax_and_scale_update_after_reduction(const at::Tensor &amax_reduction_buffer,
                                                 std::vector<at::Tensor> amax_histories,
                                                 std::vector<at::Tensor> scales,
                                                 std::vector<at::Tensor> scale_invs,
                                                 const std::string &amax_compute_algo,
                                                 transformer_engine::DType fp8_dtype, float margin);

/***************************************************************************************************
 * Rotary positional embedding
 **************************************************************************************************/

at::Tensor fused_rope_forward(const at::Tensor &input, const at::Tensor &freqs,
                              const bool transpose_output_memory);

at::Tensor fused_rope_backward(const at::Tensor &output_grads, const at::Tensor &freqs,
                               const bool transpose_output_memory);

at::Tensor fused_rope_thd_forward(const at::Tensor &input, const at::Tensor &cu_seqlens,
                                  const at::Tensor &freqs, const int cp_size, const int cp_rank);

at::Tensor fused_rope_thd_backward(const at::Tensor &output_grads, const at::Tensor &cu_seqlens,
                                   const at::Tensor &freqs, const int cp_size, const int cp_rank);

/***************************************************************************************************
 * Miscellaneous
 **************************************************************************************************/

size_t get_cublasLt_version();

size_t get_cudnn_version();

/***************************************************************************************************
 * Support THD format for Context Parallel
 **************************************************************************************************/

at::Tensor thd_read_half_tensor(const at::Tensor &tensor, const at::Tensor &cu_seqlens,
                                int half_idx);

void thd_second_half_lse_correction(at::Tensor lse, const at::Tensor &lse_per_step,
                                    const at::Tensor &cu_seqlens, bool lse_packed);

at::Tensor thd_read_second_half_lse(const at::Tensor &lse, const at::Tensor &cu_seqlens,
                                    bool lse_packed);

void thd_out_correction(at::Tensor out, const at::Tensor &out_per_step, const at::Tensor &lse,
                        const at::Tensor &lse_per_step, const at::Tensor &cu_seqlens,
                        bool only_second_half, bool lse_packed);

void thd_grad_correction(at::Tensor grad, const at::Tensor &grad_per_step,
                         const at::Tensor &cu_seqlens, const std::string &first_half,
                         const std::string &second_half);

at::Tensor thd_get_partitioned_indices(const at::Tensor &cu_seqlens, int total_tokens,
                                       int world_size, int rank);

/***************************************************************************************************
 * multi_tensor_* kernels
 **************************************************************************************************/

void multi_tensor_scale_cuda(int chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> tensor_lists, float scale);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_unscale_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::Tensor inv_scale, at::optional<bool> per_tensor_python);

using transformer_engine::DType;
void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay);

void multi_tensor_adam_fp8_cuda(int chunk_size, at::Tensor noop_flag,
                                std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, DType fp8_dtype);

void multi_tensor_adam_capturable_cuda(int chunk_size, at::Tensor noop_flag,
                                       std::vector<std::vector<at::Tensor>> tensor_lists,
                                       at::Tensor lr, const float beta1, const float beta2,
                                       const float epsilon, at::Tensor step, const int mode,
                                       const int bias_correction, const float weight_decay,
                                       at::Tensor inv_scale);

void multi_tensor_adam_capturable_master_cuda(int chunk_size, at::Tensor noop_flag,
                                              std::vector<std::vector<at::Tensor>> tensor_lists,
                                              at::Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, at::Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              at::Tensor inv_scale);

void multi_tensor_sgd_cuda(int chunk_size, at::Tensor noop_flag,
                           std::vector<std::vector<at::Tensor>> tensor_lists, float wd,
                           float momentum, float dampening, float lr, bool nesterov, bool first_run,
                           bool wd_after_momentum, float scale);

/***************************************************************************************************
 * padding
 **************************************************************************************************/

void fused_multi_row_padding(at::Tensor input, at::Tensor output,
                             std::vector<size_t> input_row_list,
                             std::vector<size_t> padded_input_row_list);

/***************************************************************************************************
 * Comm+GEMM Overlap Wrappers
 **************************************************************************************************/

class CommOverlapHelper : torch::CustomClassHolder {
 private:
  bool initialized{false};
  bool backend_is_nccl{false};
  std::map<std::string, c10d::ProcessGroup *> pgs;

 public:
  int myrank = -1;
  int numranks = -1;
  int mylocal = -1;
  int numlocal = -1;
  int mynode = -1;
  int numnodes = -1;

  CommOverlapHelper();

  CommOverlapHelper(c10d::ProcessGroup *world_group,
                    std::optional<c10d::ProcessGroup *> intra_node_group,
                    std::optional<c10d::ProcessGroup *> inter_node_group);

  ~CommOverlapHelper();

  void ub_allgather(void *globaldata, size_t globalbytes, void *localdata, size_t localbytes,
                    ExtComm comm);

  void ub_barrier(ExtComm comm);
};

class CommOverlap : torch::CustomClassHolder, public transformer_engine::CommOverlapBase {
 private:
  torch::Tensor _ubuf_torch;
  torch::Tensor _ubuf_counter;

 public:
  CommOverlap(const std::vector<size_t> &buffer_shape, at::ScalarType buffer_dtype,
              CommOverlapHelper *helper, int tp_size, int num_splits = 3,
              int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 2,
              int num_comm_sm = 16, bool set_sm_margin = true, bool atomic_gemm = false);

  void set_ubuf_scale_inv(torch::Tensor scale_inv) {
    assert(scale_inv.numel());
    assert(scale_inv.scalar_type() == torch::kFloat32);
    transformer_engine::CommOverlapBase::set_ubuf_scale_inv(
        reinterpret_cast<float *>(scale_inv.data_ptr()));
  }

  void copy_input_to_ubuf(torch::Tensor input, int comm_type);

  torch::Tensor get_ubuf_output(int comm_type);

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf
  */
  std::vector<at::Tensor> bulk_overlap(
      at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
      transformer_engine::DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
      int64_t B_fp8_tensor, transformer_engine::DType B_type, bool transb, at::Tensor D,
      at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
      transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
      size_t workspaceSize, bool accumulate, bool use_split_accumulator,
      transformer_engine::CommOverlapType comm_type, at::Tensor rs_output);

  /*
  ** Split FPROP GEMM + ReduceScatter
  */
  void atomic_gemm_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                              transformer_engine::DType A_type, bool transa, at::Tensor B,
                              at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                              transformer_engine::DType B_type, bool transb, at::Tensor D,
                              at::Tensor D_scale, transformer_engine::DType D_type,
                              at::Tensor D_amax, at::Tensor bias,
                              transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                              bool grad, at::Tensor workspace, size_t workspaceSize,
                              bool accumulate, bool use_split_accumulator, bool gemm_overlap,
                              at::Tensor rs_output);

  /*
  ** Split FPROP GEMM + ReduceScatter
  */
  void split_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        transformer_engine::DType A_type, bool transa, at::Tensor B,
                        at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                        transformer_engine::DType B_type, bool transb, at::Tensor D,
                        at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                        at::Tensor bias, transformer_engine::DType bias_type,
                        at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                        size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                        bool gemm_overlap, at::Tensor rs_output);
};  // CommOverlap

class CommOverlapP2P : torch::CustomClassHolder, public transformer_engine::CommOverlapP2PBase {
 private:
  torch::Tensor _ubuf_torch;
  torch::Tensor _ubuf_counter;

 public:
  CommOverlapP2P(const std::vector<size_t> &buffer_shape, at::ScalarType buffer_dtype,
                 CommOverlapHelper *helper, int tp_size,
                 transformer_engine::CommOverlapType comm_type,
                 int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 2,
                 int num_comm_sm = 3, bool set_sm_margin = true, bool atomic_gemm = false,
                 bool use_ce = true, bool aggregate = false);

  void set_ubuf_scale_inv(torch::Tensor scale_inv) {
    assert(scale_inv.numel());
    assert(scale_inv.scalar_type() == torch::kFloat32);
    transformer_engine::CommOverlapP2PBase::set_ubuf_scale_inv(
        reinterpret_cast<float *>(scale_inv.data_ptr()));
  }

  void copy_input_to_ubuf(torch::Tensor input, bool chunk);

  torch::Tensor get_ubuf_output(int comm_type);

  /*
  ** Split AllGather + AtomicGEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is
  *needed to have AG outputs
  ** in each rank to be in the contiguous memory space after all ring exchange
  *phases.
  */
  void atomic_gemm_overlap_ag(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                              transformer_engine::DType A_type, bool transa, at::Tensor B,
                              at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                              transformer_engine::DType B_type, bool transb, at::Tensor D,
                              at::Tensor D_scale, transformer_engine::DType D_type,
                              at::Tensor D_amax, at::Tensor bias,
                              transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                              bool grad, at::Tensor workspace, size_t workspaceSize,
                              bool accumulate, bool use_split_accumulator, at::Tensor B_copy);

  /*
  ** Split AllGather + GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is
  *needed to have AG outputs
  ** in each rank to be in the contiguous memory space after all ring exchange
  *phases.
  */
  void split_overlap_ag(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        transformer_engine::DType A_type, bool transa, at::Tensor B,
                        at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                        transformer_engine::DType B_type, bool transb, at::Tensor D,
                        at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                        at::Tensor bias, transformer_engine::DType bias_type,
                        at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                        size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                        at::Tensor B_copy);

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void atomic_gemm_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                              transformer_engine::DType A_type, bool transa, at::Tensor B,
                              at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                              transformer_engine::DType B_type, bool transb, at::Tensor D,
                              at::Tensor D_scale, transformer_engine::DType D_type,
                              at::Tensor D_amax, at::Tensor bias,
                              transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                              bool grad, at::Tensor workspace, size_t workspaceSize,
                              bool accumulate, bool use_split_accumulator, at::Tensor rs_output);

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void split_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        transformer_engine::DType A_type, bool transa, at::Tensor B,
                        at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                        transformer_engine::DType B_type, bool transb, at::Tensor D,
                        at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                        at::Tensor bias, transformer_engine::DType bias_type,
                        at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                        size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                        at::Tensor rs_output);
};  // CommOverlapP2P

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
