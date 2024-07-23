/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_

#include "common.h"
#include "common/common.h"

/***************************************************************************************************
 * Attention
 **************************************************************************************************/

NVTE_Fused_Attn_Backend get_fused_attn_backend(
    const transformer_engine::DType q_dtype, const transformer_engine::DType kv_dtype,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    float p_dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim, int64_t window_size_left, int64_t window_size_right);

std::vector<at::Tensor> fused_attn_fwd_qkvpacked(
    size_t max_seqlen, bool is_training, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const std::vector<int64_t> window_size, const at::Tensor cu_seqlens, const at::Tensor QKV,
    const transformer_engine::DType qkv_type, const c10::optional<at::Tensor> cu_seqlens_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> scale_S, const c10::optional<at::Tensor> scale_O,
    c10::optional<at::Tensor> amax_S, c10::optional<at::Tensor> amax_O,
    const c10::optional<at::Tensor> Bias, const c10::optional<at::Generator> rng_gen,
    size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd_qkvpacked(
    size_t max_seqlen, float attn_scale, float p_dropout, bool set_zero, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, const std::vector<int64_t> window_size,
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
    NVTE_Mask_Type attn_mask_type, const std::vector<int64_t> window_size,
    const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv, const at::Tensor Q,
    const at::Tensor KV, const transformer_engine::DType qkv_type,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> scale_S, const c10::optional<at::Tensor> scale_O,
    c10::optional<at::Tensor> amax_S, c10::optional<at::Tensor> amax_O,
    const c10::optional<at::Tensor> Bias, const c10::optional<at::Generator> rng_gen,
    size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd_kvpacked(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const std::vector<int64_t> window_size, bool deterministic, const at::Tensor cu_seqlens_q,
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
    NVTE_Mask_Type attn_mask_type, const std::vector<int64_t> window_size,
    const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv, const at::Tensor Q,
    const at::Tensor K, const at::Tensor V, const transformer_engine::DType qkv_type,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> scale_S, const c10::optional<at::Tensor> scale_O,
    c10::optional<at::Tensor> amax_S, c10::optional<at::Tensor> amax_O,
    const c10::optional<at::Tensor> Bias, const c10::optional<at::Generator> rng_gen,
    size_t rng_elts_per_thread);

std::vector<at::Tensor> fused_attn_bwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const std::vector<int64_t> window_size, bool deterministic, const at::Tensor cu_seqlens_q,
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

void fused_multi_cast_transpose(std::vector<at::Tensor> input_list,
                                std::vector<at::Tensor> scale_list,
                                std::vector<at::Tensor> cast_output_list,
                                std::vector<at::Tensor> transposed_output_list,
                                std::vector<at::Tensor> amax_output_list,
                                std::vector<at::Tensor> scale_inv_output_list,
                                transformer_engine::DType otype);

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
                       at::Tensor scale_inv, transformer_engine::DType otype);

void cast_to_fp8_noalloc(const at::Tensor &input, const at::Tensor &scale, at::Tensor output,
                         at::Tensor amax, at::Tensor scale_inv, transformer_engine::DType otype);

at::Tensor cast_from_fp8(const at::Tensor &input, const at::Tensor &scale_inv,
                         transformer_engine::DType itype, transformer_engine::DType otype);

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
                                  const at::Tensor &freqs);

at::Tensor fused_rope_thd_backward(const at::Tensor &output_grads, const at::Tensor &cu_seqlens,
                                   const at::Tensor &freqs);

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
                                    const at::Tensor &cu_seqlens, int total_tokens);

at::Tensor thd_read_second_half_lse(const at::Tensor &lse, const at::Tensor &cu_seqlens,
                                    int total_tokens);

void thd_out_correction(at::Tensor out, const at::Tensor &out_per_step, const at::Tensor &lse,
                        const at::Tensor &lse_per_step, const at::Tensor &cu_seqlens,
                        bool only_second_half);

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

void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay);

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
 * Comm+GEMM overlap
 **************************************************************************************************/

void set_comm_overlap_callbacks(
    std::function<void(at::Tensor &, at::Tensor &, const std::string &)> allgather_callback,
    std::function<void(at::Tensor &, int64_t, const std::string &)> bcast_callback,
    std::function<void(const std::string &)> barrier_callback);

namespace transformer_engine_torch {

class CommGemmOverlap : public torch::CustomClassHolder,
                        public transformer_engine::common::CommGemmOverlap {
 private:
  torch::Tensor _counters;
  torch::Tensor _ubuf;
  int _ubuf_bytes;
  transformer_engine::DType _ubuf_dtype;
  void *_ubuf_scale_inv_ptr;
  bool _ubuf_scale_inv_initialized{false};

 public:
  CommGemmOverlap(torch::Tensor sample, int world_rank, int world_size, int local_rank,
                  int local_size, int node_id, int num_nodes, int tp_size, int num_splits,
                  int num_max_streams, int cga_size, int num_comm_sm, bool set_sm_margin,
                  bool use_ce, bool atomic_gemm);

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf.
  */
  std::vector<at::Tensor> bulk_overlap(
      at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
      transformer_engine::DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
      int64_t B_fp8_tensor, transformer_engine::DType B_type, bool transb, at::Tensor D,
      at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
      transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
      size_t workspaceSize, bool accumulate, bool use_split_accumulator,
      NVTE_Comm_Overlap_Type comm_type, at::Tensor rs_output);

  /*
  ** Atomic GEMM + Split Reduce-Scatter
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
  ** Pipelined GEMM + Split Reduce-Scatter
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

  /*
  ** Helper function to copy input to _ubuf.
  */
  void copy_input_to_ubuf(torch::Tensor input, bool chunk);

  /*
  ** Helper function to export _ubuf output.
  */
  torch::Tensor get_ubuf_output(NVTE_Comm_Overlap_Type comm_type);

  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
  */
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv);

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // CommGemmOverlap

class CommGemmOverlapP2P : public torch::CustomClassHolder,
                           public transformer_engine::common::CommGemmOverlapP2P {
 protected:
  torch::Tensor _counters;
  torch::Tensor _ubuf;
  std::vector<torch::Tensor> _ubufs;
  transformer_engine::DType _ubuf_dtype;

  void *_ubuf_scale_inv_ptr;
  bool _ubuf_scale_inv_initialized{false};
  int _ubuf_bytes, _ubuf_chunk_bytes;

 public:
  CommGemmOverlapP2P(torch::Tensor sample, int world_rank, int world_size, int local_rank,
                     int local_size, int node_id, int num_nodes, int tp_size, int num_max_streams,
                     int cga_size, int num_comm_sms, bool set_sm_margin, bool use_ce,
                     bool atomic_gemm, bool aggregate, bool is_reduce_scatter);

  /*
  ** Split AllGather + Atomic GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  torch::Tensor atomic_gemm_overlap_ag(
      at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
      transformer_engine::DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
      int64_t B_fp8_tensor, transformer_engine::DType B_type, bool transb, at::Tensor D,
      at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
      transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
      size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor B_copy);

  /*
  ** Split AllGather + Pipelined GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  torch::Tensor split_overlap_ag(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                                 transformer_engine::DType A_type, bool transa, at::Tensor B,
                                 at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                                 transformer_engine::DType B_type, bool transb, at::Tensor D,
                                 at::Tensor D_scale, transformer_engine::DType D_type,
                                 at::Tensor D_amax, at::Tensor bias,
                                 transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                                 bool grad, at::Tensor workspace, size_t workspaceSize,
                                 bool accumulate, bool use_split_accumulator, at::Tensor B_copy);

  /*
  ** Atomic GEMM + Split Reduce-Scatter using P2P communication
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
  ** Pipelined GEMM + Split Reduce+Scatter using P2P communication
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

  /*
  ** Helper function to copy input to _ubuf or _ubufs chunks.
  */
  void copy_input_to_ubuf(torch::Tensor input, bool chunk);

  /*
  ** Helper function to export _ubuf output.
  */
  torch::Tensor get_ubuf_output(NVTE_Comm_Overlap_Type comm_type);

  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
  */
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv);

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // CommGemmOverlapP2P

}  // namespace transformer_engine_torch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
