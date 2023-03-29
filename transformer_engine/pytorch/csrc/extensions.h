/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"

class cudnnExecutionPlanManager {
 public:
    static cudnnExecutionPlanManager &Instance() {
        static thread_local cudnnExecutionPlanManager instance;
	printf("----------- instance ----------- \n");
        return instance;
    }

    cudnnHandle_t GetCudnnHandle() {
        std::once_flag flag;
        std::call_once(flag, [&] { cudnnCreate(&handle_); printf("----------- create handle ----\n");});
        return handle_;
    }

 private:
    cudnnHandle_t handle_;
};


std::vector<at::Tensor> fused_attn_fwd(
                int64_t b, int64_t max_seq_len,
                int64_t total_seqs, int64_t h, int64_t d,
                float attn_scale, float p_dropout,
                int qkv_layout, bool is_training, bool set_zero,
                at::Tensor &QKV,
                transformer_engine::DType QKV_type,
                at::Tensor &descaleQKV,
                //at::Tensor descaleS,
                //at::Tensor descaleO,
                at::Tensor &scaleS,
                at::Tensor &scaleO,
                at::Tensor amaxS,
                at::Tensor amaxO,
                at::Tensor &QKVRaggedOffset,
                at::Tensor &ORaggedOffset,
                at::Tensor &Seqlens,
                c10::optional<at::Generator> &rng_gen);

at::Tensor fused_attn_bwd(
                int64_t b, int64_t max_seq_len,
                int64_t total_seqs, int64_t h, int64_t d,
                float attn_scale, float p_dropout,
                int qkv_layout, bool set_zero,
                at::Tensor &QKV,
                at::Tensor &O,
                at::Tensor &dO,
                at::Tensor &M,
                at::Tensor &ZInv,
                transformer_engine::DType QKV_type,
                at::Tensor &descaleQKV,
                at::Tensor &descaleS,
                at::Tensor &descaleO,
                at::Tensor &descale_dO,
                //at::Tensor descale_dS,
                //at::Tensor descale_dQKV,
                at::Tensor &scaleS,
                at::Tensor &scale_dS,
                at::Tensor &scale_dQKV,
                at::Tensor amax_dS,
                at::Tensor amax_dQKV,
                at::Tensor &QKVRaggedOffset,
                at::Tensor &ORaggedOffset,
                at::Tensor &Seqlens,
                at::Tensor &rng_state);

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
             bool use_split_accumulator
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
