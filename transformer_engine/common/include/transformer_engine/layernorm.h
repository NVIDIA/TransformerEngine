/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file layernorm.h
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
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}(1 + \gamma) + \beta   if zero_centered_gamma
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}\gamma) + \beta        otherwise
 * @f]
 *
 * Calling this function with workspace set to empty tensor will not perform the operation, 
 * but instead set the shape and type of the workspace tensor to the required values.
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
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma If zero_centered_gamma is enabled 
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_layernorm_fwd(const NVTETensor x, const NVTETensor gamma, const NVTETensor beta,
                        const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma,
                        NVTETensor workspace,
                        const int multiprocessorCount, 
                        const bool zero_centered_gamma,
                        cudaStream_t stream);

/*! \brief Compute backward of LayerNorm.
 *
 * This function computes the gradient of function:
 * @f[
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}(1 + \gamma) + \beta     if zero_centered_gamma
 * y = \frac{x - E[x]}{\sqrt{Var[x] + \varepsilon}}\gamma + \beta           otherwise
 * @f]
 * else
 * with respect to \f$x\f$, \f$\gamma\f$ and \f$\beta\f$.
 *
 * Calling this function with workspace set to empty tensor will not perform the operation, 
 * but instead set the shape and type of these tensors to the required values.
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
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma If zero_centered_gamma is enabled 
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_layernorm_bwd(const NVTETensor dz,      // BxSxhidden_size
                        const NVTETensor x,       // BxSxhidden_size
                        const NVTETensor mu,      // BxS, FP32!
                        const NVTETensor rsigma,  // BxS, FP32!
                        const NVTETensor gamma,   // hidden_size
                        NVTETensor dx, NVTETensor dgamma, NVTETensor dbeta, 
                        NVTETensor workspace, 
                        const int multiprocessorCount,
                        const bool zero_centered_gamma,
                        cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_LAYER_NORM_H_
