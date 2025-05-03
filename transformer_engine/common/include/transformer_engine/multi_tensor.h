/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file multi_tensor.h
 *  \brief Functions handling multi tensor kernels.
 */

#ifndef TRANSFORMER_ENGINE_MULTI_TENSOR_H_
#define TRANSFORMER_ENGINE_MULTI_TENSOR_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

void nvte_multi_tensor_l2norm_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                   const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                   NVTETensor output, NVTETensor output_per_tensor, NVTETensor ret,
                                   NVTETensor ret_per_tensor, int per_tensor,
                                   int max_chunks_per_tensor, const int device_id,
                                   cudaStream_t stream);

void nvte_multi_tensor_unscale_l2norm_cuda(int chunk_size, NVTETensor noop_flag,
                                           NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                           const size_t num_tensors_per_list, NVTETensor output,
                                           NVTETensor output_per_tensor, NVTETensor ret,
                                           NVTETensor ret_per_tensor, NVTETensor inv_scale,
                                           int per_tensor, int max_chunks_per_tensor,
                                           const int device_id, cudaStream_t stream);

void nvte_multi_tensor_adam_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                 const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                 const float lr, const float beta1, const float beta2,
                                 const float epsilon, const int step, const int mode,
                                 const int bias_correction, const float weight_decay,
                                 const int device_id, cudaStream_t stream);

void nvte_multi_tensor_adam_param_remainder_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, const float lr, const float beta1, const float beta2,
    const float epsilon, const int step, const int mode, const int bias_correction,
    const float weight_decay, const int device_id, cudaStream_t stream);

void nvte_multi_tensor_adam_fp8_cuda(int chunk_size, NVTETensor noop_flag,
                                     NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                     const size_t num_tensors_per_list, const float lr,
                                     const float beta1, const float beta2, const float epsilon,
                                     const int step, const int mode, const int bias_correction,
                                     const float weight_decay, const NVTEDType fp8_dtype,
                                     const int device_id, cudaStream_t stream);

void nvte_multi_tensor_adam_capturable_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, const int device_id, cudaStream_t stream);

void nvte_multi_tensor_adam_capturable_master_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, const int device_id, cudaStream_t stream);

void nvte_multi_tensor_sgd_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                float wd, float momentum, float dampening, float lr, int nesterov,
                                int first_run, int wd_after_momentum, float scale,
                                const int device_id, cudaStream_t stream);

void nvte_multi_tensor_scale_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                  const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                  float scale, const int device_id, cudaStream_t stream);

void nvte_multi_tensor_compute_scale_and_scale_inv_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, float max_fp8, int force_pow_2_scales, float epsilon,
    const int device_id, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_MULTI_TENSOR_H_
