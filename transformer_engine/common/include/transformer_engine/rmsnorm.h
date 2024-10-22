/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file rmsnorm.h
 *  \brief RMSNorm functions.
 */

#ifndef TRANSFORMER_ENGINE_RMSNORM_H_
#define TRANSFORMER_ENGINE_RMSNORM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif


/*! \brief Compute RMSNorm. 
 *
 * The formula used:
 * @f[
 * y = \frac{x}{RMS_\varepsilon(x)}(1 + \gamma)       if zero_centered_gamma
 * y = \frac{x}{RMS_\varepsilon(x)}\gamma             otherwise
 * @f]
 * where
 * @f[
 * RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} x_i^2 + \varepsilon}
 * @f]
 *
 * Calling this function with workspace and barrier set to empty tensor will not
 * perform the operation, but instead set the shape and type of the workspace
 * and barrier tensors to the required values.
 *
 *  \param[in]     x                   Input tensor of shape [N, H].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[in]     epsilon             Value added to denominator for numerical stability.
 *  \param[in,out] z                   Output tensor of shape [N, H].
 *  \param[out]    rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma If zero_centered_gamma is enabled 
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_rmsnorm_fwd(const NVTETensor x, const NVTETensor gamma, const float epsilon,
                        NVTETensor z, NVTETensor rsigma, NVTETensor workspace,
                        const int multiprocessorCount, const bool zero_centered_gamma,
                        cudaStream_t stream);


/*! \brief Compute backward of RMSNorm.
 *
 * This function computes the gradient of function:
 * @f[
 * y = \frac{x}{RMS_\varepsilon(x)}(1 + \gamma)         if zero_centered_gamma
 * y = \frac{x}{RMS_\varepsilon(x)}\gamma               otherwise
 * @f]
 * where
 * @f[
 * RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} x_i^2 + \varepsilon}
 * @f]
 * with respect to \f$x\f$ and \f$gamma\f$.
 *
 * Calling this function with workspace, barrier, dgamma_part set
 * to empty tensor will not perform the operation, but instead set the shape and type
 * of these tensors to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [N, H].
 *  \param[in]     x                   Forward input tensor of shape [N, H].
 *  \param[in]     rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[out]    dx                  Output gradient of shape [N, H].
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[in]     zero_centered_gamma If zero_centered_gamma is enabled 
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_rmsnorm_bwd(const NVTETensor dz, const NVTETensor x, const NVTETensor rsigma,
                      const NVTETensor gamma, NVTETensor dx, NVTETensor dgamma,
                      NVTETensor workspace, const int multiprocessorCount, const bool zero_centered_gamma,
                      cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_RMSNORM_H_
