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

/*!  \brief Computes L2 norm for a list of tensors.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]     noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in]     tensor_lists            2D array of input tensors.
 *  \param[in]     num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]     num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]     output                  Scratch space. Required size grows with number of inputs.
 *  \param[in]     output_per_tensor       Fixed size auxilliary scratch space.
 *  \param[out]    ret                     L2 norm of all inputs.
 *  \param[out]    ret_per_tensor          L2 norm for each tensor.
 *  \param[in]     per_tensor              Whether to calculate per tensor or cumulative norm.
 *  \param[in]     max_chunks_per_tensor   Maximum number of chunks in any input tensor.
 *  \param[in]     stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_l2norm_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                   const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                   NVTETensor output, NVTETensor output_per_tensor, NVTETensor ret,
                                   NVTETensor ret_per_tensor, int per_tensor,
                                   int max_chunks_per_tensor, cudaStream_t stream);

/*!  \brief Computes L2 norm for a list of tensors after unscaling.
 *
 * Unscaling is only done for computing the L2 norm. The tensors themselves are not updated.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]     noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in]     tensor_lists            2D array of input tensors.
 *  \param[in]     num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]     num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]     output                  Scratch space. Required size grows with number of inputs.
 *  \param[in]     output_per_tensor       Fixed size auxilliary scratch space.
 *  \param[out]    ret                     L2 norm of all inputs.
 *  \param[out]    ret_per_tensor          L2 norm for each tensor.
 *  \param[in]     inv_scale               Scalar for the unscaling operation.
 *  \param[in]     per_tensor              Whether to calculate per tensor or cumulative norm.
 *  \param[in]     max_chunks_per_tensor   Maximum number of chunks in any input tensor.
 *  \param[in]     stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_unscale_l2norm_cuda(int chunk_size, NVTETensor noop_flag,
                                           NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                           const size_t num_tensors_per_list, NVTETensor output,
                                           NVTETensor output_per_tensor, NVTETensor ret,
                                           NVTETensor ret_per_tensor, NVTETensor inv_scale,
                                           int per_tensor, int max_chunks_per_tensor,
                                           cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                 const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                 const float lr, const float beta1, const float beta2,
                                 const float epsilon, const int step, const int mode,
                                 const int bias_correction, const float weight_decay,
                                 cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer
 *          where the master parameters only store the remainder bits.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_param_remainder_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, const float lr, const float beta1, const float beta2,
    const float epsilon, const int step, const int mode, const int bias_correction,
    const float weight_decay, cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer
 *          when model parameters are in Float8 precision.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      fp8_dtype               FP8 data type for model parameters.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_fp8_cuda(int chunk_size, NVTETensor noop_flag,
                                     NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                     const size_t num_tensors_per_list, const float lr,
                                     const float beta1, const float beta2, const float epsilon,
                                     const int step, const int mode, const int bias_correction,
                                     const float weight_decay, const NVTEDType fp8_dtype,
                                     cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer
 *          with CUDA graph support and LR scheduling.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      inv_scale               Scalar for the unscaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_capturable_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for Adam optimizer
 *          with CUDA graph support, LR scheduling, and FP32 master weights.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      beta1                   Coefficient for first moment of gradient.
 *  \param[in]      beta2                   Coefficient for second moment of gradient.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      step                    Iteration counter.
 *  \param[in]      mode                    Whether to use AdamW (L2 penalty applied to params).
 *  \param[in]      bias_correction         Whether to apply correction factor for moment estimates.
 *  \param[in]      weight_decay            L2 penalty for weight decay.
 *  \param[in]      inv_scale               Scalar for the unscaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_adam_capturable_master_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, cudaStream_t stream);

/*!  \brief Compute and apply gradient update to parameters for SGD optimizer.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      wd                      Weight decay (L2 penalty).
 *  \param[in]      momentum                Momentum factor.
 *  \param[in]      dampening               Dampening factor.
 *  \param[in]      lr                      Learning rate.
 *  \param[in]      nesterov                Whether or not to enable nesterov momentum.
 *  \param[in]      first_run               Whether momentum buffers have been initialized.
 *  \param[in]      wd_after_momentum       Whether to applied weight decay after momentum update.
 *  \param[in]      scale                   Scalar for the scaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_sgd_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                float wd, float momentum, float dampening, float lr, int nesterov,
                                int first_run, int wd_after_momentum, float scale,
                                cudaStream_t stream);

/*!  \brief Check overflow and scale a list of tensors.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      scale                   Scalar for the scaling operation.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_scale_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                  const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                  float scale, cudaStream_t stream);

/*!  \brief Check overflow and scale a list of tensors.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]      chunk_size              Number of tensor elements processed by a CUDA block.
 *  \param[in]      noop_flag               If this single element tensor has non-zero value, kernel will exit immediately.
 *  \param[in,out]  tensor_lists            2D array of input tensors.
 *  \param[in]      num_tensor_lists        Size (dim0) of tensor_lists.
 *  \param[in]      num_tensors_per_list    Size (dim1) of tensor_lists.
 *  \param[in]      max_fp8                 Maximum representible value in underlying FP8 format.
 *  \param[in]      force_pow_2_scales      Ensure scaling factors are a power of 2.
 *  \param[in]      epsilon                 Term added to the denominator for numerical stability.
 *  \param[in]      stream                  CUDA stream used for this operation.
 */
void nvte_multi_tensor_compute_scale_and_scale_inv_cuda(int chunk_size, NVTETensor noop_flag,
                                                        NVTETensor **tensor_lists,
                                                        const size_t num_tensor_lists,
                                                        const size_t num_tensors_per_list,
                                                        float max_fp8, int force_pow_2_scales,
                                                        float epsilon, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_MULTI_TENSOR_H_
