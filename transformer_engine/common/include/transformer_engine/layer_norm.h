/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file layer_norm.h
 *  \brief LayerNorm functions.
 */

#ifndef TRANSFORMER_ENGINE_LAYER_NORM_H_
#define TRANSFORMER_ENGINE_LAYER_NORM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute LayerNorm on the input.
 *
 * The formula used:
 * @f[
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}\gamma + \beta
 * @f]
 *
 * Calling this function with workspace and barrier set to empty tensor will not
 * perform the operation, but instead set the shape and type of the workspace
 * and barrier tensors to the required values.
 *
 *  \param[in]     x                   Input tensor of shape [N, H].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[in]     beta                Beta tensor of shape [H].
 *  \param[in]     epsilon             Value added to denominator for numerical stability.
 *  \param[in,out] z                   Output tensor of shape [N, H].
 *  \param[out]    mu                  Mean of the input calculated over the last dimension.
 *                                     Shape: [N].
 *  \param[out]    rsigma              Inverse of the variance of the input calculated over
 *                                     the last dimension. Shape: [N].
 *  \param[in]     stream              CUDA stream used for the operation.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[out]    barrier             Barrier tensor.
 */
void nvte_layernorm_fwd(const NVTETensor x, const NVTETensor gamma, const NVTETensor beta,
                        const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma,
                        cudaStream_t stream, const int multiprocessorCount, NVTETensor workspace,
                        NVTETensor barrier);

/*! \brief Compute LayerNorm with zero-centered gamma on the input.
 *
 * The formula used:
 * @f[
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}(1 + \gamma) + \beta
 * @f]
 *
 * Calling this function with workspace and barrier set to empty tensor will not
 * perform the operation, but instead set the shape and type of the workspace
 * and barrier tensors to the required values.
 *
 *  \param[in]     x                   Input tensor of shape [N, H].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[in]     beta                Beta tensor of shape [H].
 *  \param[in]     epsilon             Value added to denominator for numerical stability.
 *  \param[in,out] z                   Output tensor of shape [N, H].
 *  \param[out]    mu                  Mean of the input calculated over the last dimension.
 *                                     Shape: [N].
 *  \param[out]    rsigma              Inverse of the variance of the input calculated over
 *                                     the last dimension. Shape: [N].
 *  \param[in]     stream              CUDA stream used for the operation.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[out]    barrier             Barrier tensor.
 */
void nvte_layernorm1p_fwd(const NVTETensor x, const NVTETensor gamma, const NVTETensor beta,
                          const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma,
                          cudaStream_t stream, const int multiprocessorCount, NVTETensor workspace,
                          NVTETensor barrier);

/*! \brief Compute backward of LayerNorm.
 *
 * This function computes the gradient of function:
 * @f[
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}\gamma + \beta
 * @f]
 * with respect to \f$x\f$, \f$\gamma\f$ and \f$\beta\f$.
 *
 * Calling this function with workspace, barrier, dgamma_part and dbeta_part set
 * to empty tensor will not perform the operation, but instead set the shape and type
 * of these tensors to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [N, H].
 *  \param[in]     x                   Forward input tensor of shape [N, H].
 *  \param[in]     mu                  Mean of the input calculated over the last dimension.
 *                                     Shape: [N].
 *  \param[in]     rsigma              Inverse of the variance of the input calculated over
 *                                     the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[out]    dx                  Output gradient of shape [N, H].
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H].
 *  \param[out]    dbeta               Gradient for beta tensor of shape [H].
 *  \param[out]    dgamma_part         Storage for partial gamma gradient.
 *  \param[out]    dbeta_part          Storage for partial bias gradient.
 *  \param[in]     stream              CUDA stream used for the operation.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[out]    barrier             Barrier tensor.
 */
void nvte_layernorm_bwd(const NVTETensor dz,      // BxSxhidden_size
                        const NVTETensor x,       // BxSxhidden_size
                        const NVTETensor mu,      // BxS, FP32!
                        const NVTETensor rsigma,  // BxS, FP32!
                        const NVTETensor gamma,   // hidden_size
                        NVTETensor dx, NVTETensor dgamma, NVTETensor dbeta, NVTETensor dgamma_part,
                        NVTETensor dbeta_part, cudaStream_t stream, const int multiprocessorCount,
                        NVTETensor workspace, NVTETensor barrier);

/*! \brief Compute backward of LayerNorm with zero-centered gamma.
 *
 * This function computes the gradient of function:
 * @f[
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}(1 + \gamma) + \beta
 * @f]
 * with respect to \f$x\f$, \f$\gamma\f$ and \f$\beta\f$.
 *
 * Calling this function with workspace, barrier, dgamma_part and dbeta_part set
 * to empty tensor will not perform the operation, but instead set the shape and type
 * of these tensors to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [N, H].
 *  \param[in]     x                   Forward input tensor of shape [N, H].
 *  \param[in]     mu                  Mean of the input calculated over the last dimension.
 *                                     Shape: [N].
 *  \param[in]     rsigma              Inverse of the variance of the input calculated over
 *                                     the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[out]    dx                  Output gradient of shape [N, H].
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H].
 *  \param[out]    dbeta               Gradient for beta tensor of shape [H].
 *  \param[out]    dgamma_part         Storage for partial gamma gradient.
 *  \param[out]    dbeta_part          Storage for partial bias gradient.
 *  \param[in]     stream              CUDA stream used for the operation.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[out]    barrier             Barrier tensor.
 */
void nvte_layernorm1p_bwd(const NVTETensor dz,      // BxSxhidden_size
                          const NVTETensor x,       // BxSxhidden_size
                          const NVTETensor mu,      // BxS, FP32!
                          const NVTETensor rsigma,  // BxS, FP32!
                          const NVTETensor gamma,   // hidden_size
                          NVTETensor dx, NVTETensor dgamma, NVTETensor dbeta,
                          NVTETensor dgamma_part, NVTETensor dbeta_part, cudaStream_t stream,
                          const int multiprocessorCount, NVTETensor workspace, NVTETensor barrier);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_LAYER_NORM_H_
