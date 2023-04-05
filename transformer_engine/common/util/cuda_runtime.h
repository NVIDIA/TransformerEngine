/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_

#include <cuda_runtime_api.h>

namespace transformer_engine {

namespace cuda {

/* \brief Number of accessible CUDA devices */
int num_devices();

/* \brief Which CUDA device is currently being used */
int current_device();

/* \brief Compute capability of CUDA device
 *
 * \return Compute capability as int. Last digit is minor revision,
 *         remaining digits are major revision.
 */
int sm_arch(int device_id);

}  // namespace cuda

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_
