/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_

#include <optional>

#include "common.h"

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

std::vector<py::object> fused_attn_fwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, const std::vector<int64_t> window_size,
    const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv, const py::handle Q,
    const py::handle K, const py::handle V, const at::ScalarType fake_dtype,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded, py::handle s_quantizer,
    py::handle o_quantizer, const c10::optional<at::Tensor> Bias,
    const c10::optional<at::Generator> rng_gen, size_t rng_elts_per_thread);

std::vector<py::object> fused_attn_bwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const std::vector<int64_t> window_size, bool deterministic, const at::Tensor cu_seqlens_q,
    const at::Tensor cu_seqlens_kv, const py::handle Q, const py::handle K, const py::handle V,
    const py::handle O, const py::handle dO, const at::ScalarType fake_dtype,
    const transformer_engine::DType dqkv_type, const std::vector<at::Tensor> Aux_CTX_Tensors,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded, py::handle s_quantizer,
    py::handle dp_quantizer, py::handle dqkv_quantizer);

at::Tensor fa_prepare_fwd(at::Tensor qkvi);
at::Tensor fa_prepare_bwd(at::Tensor q, at::Tensor k, at::Tensor v);

/***************************************************************************************************
 * GEMM
 **************************************************************************************************/

using MaybeTensor = std::optional<at::Tensor>;

void te_atomic_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
                    std::vector<int64_t> A_scaling_mode, bool transa, at::Tensor B,
                    at::Tensor B_scale_inverse, transformer_engine::DType B_type,
                    std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D,
                    at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                    bool grad, at::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, at::Tensor counter);

std::optional<std::vector<at::Tensor>> te_general_grouped_gemm(
    std::vector<py::handle> A, bool transa, std::vector<py::handle> B, bool transb,
    std::optional<std::vector<at::Tensor>> D, transformer_engine::DType D_type,
    std::vector<int64_t> m_splits, std::vector<at::Tensor> bias,
    transformer_engine::DType bias_type, bool single_output, std::vector<at::Tensor> pre_gelu_out,
    bool grad, std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, int math_sm_count);

/***************************************************************************************************
 * Transpose
 **************************************************************************************************/

std::vector<py::object> fused_multi_quantize(std::vector<py::handle> input_list,
                                             std::optional<std::vector<py::handle>> output_list,
                                             std::vector<py::handle> quantizer_list,
                                             transformer_engine::DType otype);

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype,
                         std::optional<at::Tensor> output = std::nullopt);

namespace transformer_engine::pytorch {

/***************************************************************************************************
 * Activations
 **************************************************************************************************/

py::object gelu(const at::Tensor &input, py::handle quantizer);

py::object relu(const at::Tensor &input, py::handle quantizer);

py::object geglu(const at::Tensor &input, py::handle quantizer);

py::object qgeglu(const at::Tensor &input, py::handle quantizer);

py::object reglu(const at::Tensor &input, py::handle quantizer);

py::object swiglu(const at::Tensor &input, py::handle quantizer);

py::object qgelu(const at::Tensor &input, py::handle quantizer);

py::object srelu(const at::Tensor &input, py::handle quantizer);

py::object dgelu(const at::Tensor &grad, const at::Tensor &input, py::handle quantizer);

py::object drelu(const at::Tensor &grad, const at::Tensor &input, py::handle quantizer);

py::object dgeglu(const at::Tensor &grad, const at::Tensor &input, py::handle quantizer);

py::object dqgeglu(const at::Tensor &grad, const at::Tensor &input, py::handle quantizer);

py::object dreglu(const at::Tensor &grad, const at::Tensor &input, py::handle quantizer);

py::object dswiglu(const at::Tensor &grad, const at::Tensor &input, py::handle quantizer);

py::object dqgelu(const at::Tensor &grad, const at::Tensor &input, py::handle quantizer);

py::object dsrelu(const at::Tensor &grad, const at::Tensor &input, py::handle quantizer);

}  // namespace transformer_engine::pytorch

/***************************************************************************************************
 * LayerNorm
 **************************************************************************************************/

std::vector<py::object> layernorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                      const at::Tensor &mu, const at::Tensor &rsigma,
                                      const at::Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma);

std::vector<py::object> layernorm_fwd(py::handle input, py::handle weight, MaybeTensor bias,
                                      float eps, py::object ln_out, py::handle quantizer,
                                      transformer_engine::DType out_dtype, const int sm_margin,
                                      const bool zero_centered_gamma);

/***************************************************************************************************
 * RMSNorm
 **************************************************************************************************/

std::vector<py::object> rmsnorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                    const at::Tensor &rsigma, const at::Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma);

std::vector<py::object> rmsnorm_fwd(const py::handle &input, const py::handle &weight, float eps,
                                    py::object ln_out, py::handle quantizer,
                                    transformer_engine::DType otype, const int sm_margin,
                                    const bool zero_centered_gamma);

/***************************************************************************************************
 * Cast
 **************************************************************************************************/

namespace transformer_engine::pytorch {

py::object quantize(const at::Tensor &tensor, py::handle quantizer, const py::object &output,
                    std::optional<at::Tensor> noop);

py::object dequantize(const py::handle &input, transformer_engine::DType otype);

std::vector<py::object> bgrad_quantize(const at::Tensor &input, py::handle py_quantizer);

std::vector<py::object> gemm(py::handle A, bool transa, py::handle B, bool transb, py::object D,
                             py::handle quantizer, std::optional<DType> out_dtype, MaybeTensor bias,
                             DType bias_type, bool gelu, MaybeTensor gelu_in, bool grad,
                             at::Tensor workspace, size_t workspaceSize, bool accumulate,
                             bool use_split_accumulator, CommOverlapCore *comm_overlap = nullptr,
                             std::optional<CommOverlapType> comm_type = std::nullopt,
                             MaybeTensor extra_output = std::nullopt, bool bulk_overlap = false);

/***************************************************************************************************
 * Cast fusions
 **************************************************************************************************/

std::vector<py::object> dbias_dgelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer);

std::vector<py::object> dbias_dsilu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer);

std::vector<py::object> dbias_drelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer);

std::vector<py::object> dbias_dqgelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                     py::handle quantizer);

std::vector<py::object> dbias_dsrelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                     py::handle quantizer);

}  // namespace transformer_engine::pytorch

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
                                    bool lse_packed, int second_half_lse_seqlen);

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

void multi_tensor_adam_param_remainder_cuda(int chunk_size, at::Tensor noop_flag,
                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay);

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
 * swizzle
 **************************************************************************************************/

void swizzle_scaling_factors(transformer_engine::TensorWrapper &input, bool trans);

at::Tensor rowwise_swizzle(at::Tensor input, at::Tensor scale_inv);

at::Tensor columnwise_swizzle(at::Tensor input, at::Tensor scale_inv);

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
 public:
  CommOverlap(const std::vector<size_t> &buffer_shape, at::ScalarType buffer_dtype,
              CommOverlapHelper *helper, int tp_size, int num_splits = 3,
              int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 2,
              int gemm_priority = 0, int comm_priority = 0, int num_comm_sm = 16,
              bool set_sm_margin = true, bool atomic_gemm = false,
              bool rs_overlap_first_gemm = false);

  ~CommOverlap() {}

  void set_buffer_params(py::handle quantizer);

  void copy_into_buffer(py::handle input, py::handle quantizer, bool local_chunk = false);

  py::object get_buffer(py::handle quantizer, bool local_chunk = false,
                        std::optional<const std::vector<int64_t>> shape = std::nullopt);

};  // CommOverlap

class CommOverlapP2P : torch::CustomClassHolder, public transformer_engine::CommOverlapP2PBase {
 public:
  CommOverlapP2P(const std::vector<size_t> &buffer_shape, at::ScalarType buffer_dtype,
                 CommOverlapHelper *helper, int tp_size,
                 transformer_engine::CommOverlapType comm_type,
                 int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 2,
                 int gemm_priority = 0, int comm_priority = 0, int num_comm_sm = 3,
                 bool set_sm_margin = true, bool atomic_gemm = false, bool use_ce = true,
                 bool aggregate = false);

  ~CommOverlapP2P() {}

  void set_buffer_params(py::handle quantizer);

  void copy_into_buffer(py::handle input, py::handle quantizer, bool local_chunk = false);

  py::object get_buffer(py::handle quantizer, bool local_chunk = false,
                        std::optional<const std::vector<int64_t>> shape = std::nullopt);

};  // CommOverlapP2P

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
